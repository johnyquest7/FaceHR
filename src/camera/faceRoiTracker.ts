import type { FaceLandmarker } from '@mediapipe/tasks-vision';
import type {
  FrameRoiExtraction,
  NormalizedPoint,
  RgbSample,
  RoiBox,
  TrackingMode,
} from '../types/rppg';

const FACE_LANDMARKER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// Aspect-preserving offscreen readback target (longest side ~360px preserves portrait mobile resolution)
const OFFSCREEN_MAX_DIM = 360;
const OFFSCREEN_MIN_DIM = 180;

const MIN_PIXEL_VALUE = 15;
const MAX_PIXEL_VALUE = 245;

// YCbCr skin color bounds (Chai & Ngan / standard rPPG skin mask)
const CB_MIN = 77;
const CB_MAX = 127;
const CR_MIN = 133;
const CR_MAX = 173;

// Minimum fraction of non-clipped pixels passing strict YCbCr before falling back under tinted lighting
const STRICT_SKIN_MIN_RATIO = 0.08;

/**
 * Computes aspect-preserving offscreen canvas dimensions so portrait mobile feeds (e.g. 480x640)
 * are never horizontally squashed into a landscape buffer.
 */
export function computeOffscreenDimensions(
  videoWidth: number,
  videoHeight: number
): { width: number; height: number } {
  if (videoWidth <= 0 || videoHeight <= 0) {
    return { width: 320, height: 240 };
  }
  const longest = Math.max(videoWidth, videoHeight);
  const scale = Math.min(1, OFFSCREEN_MAX_DIM / longest);
  const width = Math.max(OFFSCREEN_MIN_DIM, Math.round(videoWidth * scale));
  const height = Math.max(OFFSCREEN_MIN_DIM, Math.round(videoHeight * scale));
  return { width, height };
}

/**
 * Canonical MediaPipe 468-landmark indices for clean vascular skin ROIs.
 * Carefully chosen to avoid eyebrows, eyes, glasses rims, nostrils, and mouth.
 */
const FOREHEAD_LANDMARKS = [10, 338, 297, 332, 333, 299, 337, 151, 108, 69, 104, 103, 67, 109];
// Subject's left cheek (malar region, image-right on unmirrored frame)
const LEFT_CHEEK_LANDMARKS = [330, 347, 346, 352, 411, 425, 266];
// Subject's right cheek (malar region, image-left on unmirrored frame)
const RIGHT_CHEEK_LANDMARKS = [101, 118, 117, 123, 187, 205, 36];
// Outer face oval for stable face bounding box
const FACE_OVAL_LANDMARKS = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
  148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

// Centered face guide geometry for graceful skin-fallback mode
const FALLBACK_FACE_BOX: RoiBox = {
  x: 0.28,
  y: 0.14,
  width: 0.44,
  height: 0.68,
  label: 'face',
};

const FALLBACK_FOREHEAD_BOX: RoiBox = {
  x: 0.37,
  y: 0.21,
  width: 0.26,
  height: 0.12,
  label: 'forehead',
};

const FALLBACK_RIGHT_CHEEK_BOX: RoiBox = {
  x: 0.33,
  y: 0.45,
  width: 0.13,
  height: 0.15,
  label: 'right-cheek',
};

const FALLBACK_LEFT_CHEEK_BOX: RoiBox = {
  x: 0.54,
  y: 0.45,
  width: 0.13,
  height: 0.15,
  label: 'left-cheek',
};

export interface SkinExtractionResult {
  rgb: RgbSample | null;
  brightness: number;
  saturationFraction: number;
  /** Fraction of total ROI pixels matching strict YCbCr skin bounds (0..1) */
  skinPixelRatio: number;
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

function pointInPolygon(px: number, py: number, polygon: NormalizedPoint[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;

    const intersects =
      yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi + 1e-12) + xi;
    if (intersects) {
      inside = !inside;
    }
  }
  return inside;
}

/**
 * Pure helper that extracts mean skin RGB, luminance brightness, and clipping fraction
 * from an RGBA ImageData buffer within a normalized RoiBox (and optional polygon mask).
 *
 * Uses YCbCr skin bounds (77 <= Cb <= 127, 133 <= Cr <= 173) and rejects clipped pixels
 * (< 15 or > 245). Falls back gracefully to non-clipped pixels when strict YCbCr rejects
 * too many pixels (e.g., under strongly tinted ambient lighting).
 */
export function extractSkinRgbFromImageData(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  box: RoiBox
): SkinExtractionResult {
  if (width <= 0 || height <= 0 || data.length < width * height * 4) {
    return { rgb: null, brightness: 0, saturationFraction: 1, skinPixelRatio: 0 };
  }

  const x0 = clamp(Math.floor(box.x * width), 0, width);
  const y0 = clamp(Math.floor(box.y * height), 0, height);
  const x1 = clamp(Math.ceil((box.x + box.width) * width), 0, width);
  const y1 = clamp(Math.ceil((box.y + box.height) * height), 0, height);

  if (x1 <= x0 || y1 <= y0) {
    return { rgb: null, brightness: 0, saturationFraction: 1, skinPixelRatio: 0 };
  }

  const polygon = box.polygon && box.polygon.length >= 3 ? box.polygon : null;
  const invW = 1 / width;
  const invH = 1 / height;

  let totalCount = 0;
  let clippedCount = 0;
  let lumSum = 0;

  let skinR = 0;
  let skinG = 0;
  let skinB = 0;
  let skinLumSum = 0;
  let skinLumSqSum = 0;
  let skinCount = 0;

  // Store skin pixel coordinates/values for specular glare & deep shadow outlier trimming
  const maxBoxPixels = (x1 - x0) * (y1 - y0);
  const skinBufR = new Float32Array(maxBoxPixels);
  const skinBufG = new Float32Array(maxBoxPixels);
  const skinBufB = new Float32Array(maxBoxPixels);
  const skinBufLum = new Float32Array(maxBoxPixels);

  let nonClippedR = 0;
  let nonClippedG = 0;
  let nonClippedB = 0;
  let nonClippedCount = 0;

  for (let py = y0; py < y1; py++) {
    const ny = (py + 0.5) * invH;
    const rowOffset = py * width * 4;

    for (let px = x0; px < x1; px++) {
      if (polygon) {
        const nx = (px + 0.5) * invW;
        if (!pointInPolygon(nx, ny, polygon)) {
          continue;
        }
      }

      const idx = rowOffset + px * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];

      totalCount += 1;
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      lumSum += lum;

      if (
        r < MIN_PIXEL_VALUE ||
        g < MIN_PIXEL_VALUE ||
        b < MIN_PIXEL_VALUE ||
        r > MAX_PIXEL_VALUE ||
        g > MAX_PIXEL_VALUE ||
        b > MAX_PIXEL_VALUE
      ) {
        clippedCount += 1;
        continue;
      }

      nonClippedR += r;
      nonClippedG += g;
      nonClippedB += b;
      nonClippedCount += 1;

      const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
      const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;

      if (cb >= CB_MIN && cb <= CB_MAX && cr >= CR_MIN && cr <= CR_MAX) {
        skinBufR[skinCount] = r;
        skinBufG[skinCount] = g;
        skinBufB[skinCount] = b;
        skinBufLum[skinCount] = lum;
        skinR += r;
        skinG += g;
        skinB += b;
        skinLumSum += lum;
        skinLumSqSum += lum * lum;
        skinCount += 1;
      }
    }
  }

  if (totalCount === 0) {
    return { rgb: null, brightness: 0, saturationFraction: 1, skinPixelRatio: 0 };
  }

  const brightness = lumSum / totalCount;
  const saturationFraction = clippedCount / totalCount;
  const skinPixelRatio = skinCount / totalCount;

  if (nonClippedCount === 0) {
    return { rgb: null, brightness, saturationFraction, skinPixelRatio: 0 };
  }

  // Prefer strict YCbCr skin pixels when enough are present; trim bright specular glare
  // and dark shadow/eyebrow outliers (> 1.5 sigma from ROI skin luminance mean) to boost SNR.
  const minStrictCount = Math.max(1, Math.floor(nonClippedCount * STRICT_SKIN_MIN_RATIO));
  if (skinCount >= minStrictCount) {
    if (skinCount >= 20) {
      const meanSkinLum = skinLumSum / skinCount;
      const variance = Math.max(0, skinLumSqSum / skinCount - meanSkinLum * meanSkinLum);
      const stdSkinLum = Math.sqrt(variance);
      if (stdSkinLum > 3.0) {
        const lowCut = meanSkinLum - 1.5 * stdSkinLum;
        const highCut = meanSkinLum + 1.5 * stdSkinLum;
        let trimR = 0;
        let trimG = 0;
        let trimB = 0;
        let trimCount = 0;
        for (let i = 0; i < skinCount; i++) {
          const l = skinBufLum[i];
          if (l >= lowCut && l <= highCut) {
            trimR += skinBufR[i];
            trimG += skinBufG[i];
            trimB += skinBufB[i];
            trimCount += 1;
          }
        }
        if (trimCount >= 12) {
          return {
            rgb: {
              r: trimR / trimCount,
              g: trimG / trimCount,
              b: trimB / trimCount,
              pixelCount: trimCount,
            },
            brightness,
            saturationFraction,
            skinPixelRatio,
          };
        }
      }
    }

    return {
      rgb: {
        r: skinR / skinCount,
        g: skinG / skinCount,
        b: skinB / skinCount,
        pixelCount: skinCount,
      },
      brightness,
      saturationFraction,
      skinPixelRatio,
    };
  }

  return {
    rgb: {
      r: nonClippedR / nonClippedCount,
      g: nonClippedG / nonClippedCount,
      b: nonClippedB / nonClippedCount,
      pixelCount: nonClippedCount,
    },
    brightness,
    saturationFraction,
    skinPixelRatio,
  };
}

function buildPolygonRoiBox(
  landmarks: Array<{ x: number; y: number }>,
  indices: number[],
  label: RoiBox['label'],
  includePolygon = true
): RoiBox | null {
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  const polygon: NormalizedPoint[] = [];

  for (const idx of indices) {
    const pt = landmarks[idx];
    if (!pt || !Number.isFinite(pt.x) || !Number.isFinite(pt.y)) continue;
    const x = clamp(pt.x, 0, 1);
    const y = clamp(pt.y, 0, 1);
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    polygon.push({ x, y });
  }

  const width = maxX - minX;
  const height = maxY - minY;
  if (polygon.length < 3 || width < 0.015 || height < 0.015) {
    return null;
  }

  return includePolygon
    ? { x: minX, y: minY, width, height, polygon, label }
    : { x: minX, y: minY, width, height, label };
}

function fuseRegionSamples(entries: Array<{ sample: RgbSample | null; weight: number }>): RgbSample | null {
  let weightSum = 0;
  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let totalPixels = 0;

  for (const entry of entries) {
    if (!entry.sample || entry.sample.pixelCount <= 0) continue;
    // Combine anatomical priority weight with gentle pixel-area confidence
    const effectiveWeight = entry.weight * Math.min(1.5, Math.sqrt(entry.sample.pixelCount / 100));
    weightSum += effectiveWeight;
    rSum += entry.sample.r * effectiveWeight;
    gSum += entry.sample.g * effectiveWeight;
    bSum += entry.sample.b * effectiveWeight;
    totalPixels += entry.sample.pixelCount;
  }

  if (weightSum <= 0 || totalPixels <= 0) {
    return null;
  }

  return {
    r: rSum / weightSum,
    g: gSum / weightSum,
    b: bSum / weightSum,
    pixelCount: totalPixels,
  };
}

function resolveWasmBasePath(): string {
  const rawBase =
    typeof import.meta !== 'undefined' && import.meta.env && typeof import.meta.env.BASE_URL === 'string'
      ? import.meta.env.BASE_URL
      : './';
  const normalizedBase = rawBase.endsWith('/') ? rawBase : `${rawBase}/`;
  const relativePath = `${normalizedBase}mediapipe/wasm`;

  if (typeof window !== 'undefined' && window.location?.href) {
    try {
      return new URL(relativePath, window.location.href).href.replace(/\/$/, '');
    } catch {
      // Fall back to relative path
    }
  }
  return relativePath.replace(/\/$/, '');
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`MediaPipe initialization timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      }
    );
  });
}

interface PreviousFaceGeometry {
  cx: number;
  cy: number;
  scale: number;
  timestampMs: number;
}

export class FaceRoiTracker {
  private landmarker: FaceLandmarker | null = null;
  private initPromise: Promise<TrackingMode> | null = null;
  private mode: TrackingMode = 'skin-fallback';
  private mediapipeFailed = false;
  private disposed = false;

  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  private lastMpTimestampMs = -1;
  private prevGeometry: PreviousFaceGeometry | null = null;
  private smoothedPolygons: Partial<Record<RoiBox['label'], NormalizedPoint[]>> = {};
  private prevFallbackLuma: Float32Array | null = null;
  private prevFallbackTimestampMs = -1;

  public getTrackingMode(): TrackingMode {
    return this.mode;
  }

  /**
   * Lazily initializes MediaPipe FaceLandmarker using the bundled local WASM assets.
   * Automatically falls back to 'skin-fallback' if offline, unsupported, or GPU/CPU init fails.
   */
  public async init(): Promise<TrackingMode> {
    if (this.disposed) {
      return 'skin-fallback';
    }
    if (this.landmarker) {
      this.mode = 'mediapipe';
      return 'mediapipe';
    }
    if (this.mediapipeFailed) {
      this.mode = 'skin-fallback';
      return 'skin-fallback';
    }
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = this.loadMediaPipe();
    const resolvedMode = await this.initPromise;
    this.initPromise = null;
    return resolvedMode;
  }

  private async loadMediaPipe(): Promise<TrackingMode> {
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      this.mediapipeFailed = true;
      this.mode = 'skin-fallback';
      return 'skin-fallback';
    }

    try {
      const vision = await withTimeout(import('@mediapipe/tasks-vision'), 8000);
      if (this.disposed) return 'skin-fallback';

      const wasmPath = resolveWasmBasePath();
      const filesetResolver = await withTimeout(
        vision.FilesetResolver.forVisionTasks(wasmPath),
        10000
      );
      if (this.disposed) return 'skin-fallback';

      const createWithDelegate = async (delegate: 'GPU' | 'CPU') =>
        withTimeout(
          vision.FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
              modelAssetPath: FACE_LANDMARKER_MODEL_URL,
              delegate,
            },
            runningMode: 'VIDEO',
            numFaces: 1,
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
          }),
          15000
        );

      let instance: FaceLandmarker;
      try {
        instance = await createWithDelegate('GPU');
      } catch {
        if (this.disposed) return 'skin-fallback';
        instance = await createWithDelegate('CPU');
      }

      if (this.disposed) {
        instance.close();
        return 'skin-fallback';
      }

      this.landmarker = instance;
      this.mode = 'mediapipe';
      this.mediapipeFailed = false;
      return 'mediapipe';
    } catch {
      this.mediapipeFailed = true;
      this.mode = 'skin-fallback';
      return 'skin-fallback';
    }
  }

  private ensureCanvas(videoWidth: number, videoHeight: number): CanvasRenderingContext2D | null {
    if (typeof document === 'undefined') return null;

    const { width, height } = computeOffscreenDimensions(videoWidth, videoHeight);
    if (!this.canvas) {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return null;
      this.canvas = canvas;
      this.ctx = ctx;
      return ctx;
    }

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    return this.ctx;
  }

  /**
   * Light adaptive EMA smoothing on ROI polygon vertices to eliminate 1–2 px MediaPipe
   * landmark jitter across eyebrows/hair edges while staying responsive to real head motion.
   */
  private smoothRoiBox(box: RoiBox | null, motionScore: number): RoiBox | null {
    if (!box || !box.polygon || box.polygon.length < 3) {
      if (box?.label) {
        delete this.smoothedPolygons[box.label];
      }
      return box;
    }

    const prev = this.smoothedPolygons[box.label];
    if (!prev || prev.length !== box.polygon.length) {
      this.smoothedPolygons[box.label] = box.polygon.map((p) => ({ x: p.x, y: p.y }));
      return box;
    }

    const alpha = motionScore < 0.2 ? 0.55 : motionScore < 0.5 ? 0.8 : 1.0;
    let minX = 1;
    let minY = 1;
    let maxX = 0;
    let maxY = 0;
    const smoothed: NormalizedPoint[] = new Array(box.polygon.length);

    for (let i = 0; i < box.polygon.length; i++) {
      const x = prev[i].x + alpha * (box.polygon[i].x - prev[i].x);
      const y = prev[i].y + alpha * (box.polygon[i].y - prev[i].y);
      smoothed[i] = { x, y };
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }

    this.smoothedPolygons[box.label] = smoothed;
    return {
      ...box,
      x: minX,
      y: minY,
      width: Math.max(0.015, maxX - minX),
      height: Math.max(0.015, maxY - minY),
      polygon: smoothed,
    };
  }

  /**
   * Extracts facial skin ROIs, fused RGB sample, luminance, clipping, and motion
   * from the current HTMLVideoElement frame.
   */
  public extractFrame(video: HTMLVideoElement, timestampMs: number): FrameRoiExtraction {
    const emptyResult = (trackingMode: TrackingMode): FrameRoiExtraction => ({
      timestampMs,
      faceDetected: false,
      fusedRgb: null,
      regions: {},
      faceBox: null,
      roiBoxes: [],
      motionScore: 0,
      brightness: 0,
      saturationFraction: 0,
      trackingMode,
    });

    if (this.disposed || video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0) {
      return emptyResult(this.mode);
    }

    const ctx = this.ensureCanvas(video.videoWidth, video.videoHeight);
    if (!ctx || !this.canvas) {
      return emptyResult(this.mode);
    }

    const cw = this.canvas.width;
    const ch = this.canvas.height;
    ctx.drawImage(video, 0, 0, cw, ch);
    const imageData = ctx.getImageData(0, 0, cw, ch);

    if (this.landmarker && this.mode === 'mediapipe') {
      try {
        return this.extractWithMediaPipe(video, imageData.data, cw, ch, timestampMs);
      } catch {
        // If MediaPipe throws unexpectedly on a specific GPU/driver, degrade cleanly to skin-fallback
        this.mode = 'skin-fallback';
      }
    }

    return this.extractWithSkinFallback(imageData.data, cw, ch, timestampMs);
  }

  private extractWithMediaPipe(
    video: HTMLVideoElement,
    data: Uint8ClampedArray,
    width: number,
    height: number,
    timestampMs: number
  ): FrameRoiExtraction {
    const mpTimestamp = Math.max(Math.round(timestampMs), this.lastMpTimestampMs + 1);
    this.lastMpTimestampMs = mpTimestamp;

    const detection = this.landmarker!.detectForVideo(video, mpTimestamp);
    const landmarks = detection.faceLandmarks?.[0];

    if (!landmarks || landmarks.length < 468) {
      this.prevGeometry = null;
      return {
        timestampMs,
        faceDetected: false,
        fusedRgb: null,
        regions: {},
        faceBox: null,
        roiBoxes: [],
        motionScore: 0,
        brightness: 0,
        saturationFraction: 0,
        trackingMode: 'mediapipe',
      };
    }

    const faceBox = buildPolygonRoiBox(landmarks, FACE_OVAL_LANDMARKS, 'face', false);
    const motionScore = this.computeLandmarkMotion(faceBox, timestampMs);

    const foreheadBox = this.smoothRoiBox(
      buildPolygonRoiBox(landmarks, FOREHEAD_LANDMARKS, 'forehead', true),
      motionScore
    );
    const leftCheekBox = this.smoothRoiBox(
      buildPolygonRoiBox(landmarks, LEFT_CHEEK_LANDMARKS, 'left-cheek', true),
      motionScore
    );
    const rightCheekBox = this.smoothRoiBox(
      buildPolygonRoiBox(landmarks, RIGHT_CHEEK_LANDMARKS, 'right-cheek', true),
      motionScore
    );

    const roiBoxes: RoiBox[] = [];
    if (foreheadBox) roiBoxes.push(foreheadBox);
    if (leftCheekBox) roiBoxes.push(leftCheekBox);
    if (rightCheekBox) roiBoxes.push(rightCheekBox);

    const foreheadRes = foreheadBox
      ? extractSkinRgbFromImageData(data, width, height, foreheadBox)
      : null;
    const leftCheekRes = leftCheekBox
      ? extractSkinRgbFromImageData(data, width, height, leftCheekBox)
      : null;
    const rightCheekRes = rightCheekBox
      ? extractSkinRgbFromImageData(data, width, height, rightCheekBox)
      : null;

    const regions: FrameRoiExtraction['regions'] = {};
    if (foreheadRes?.rgb) regions.forehead = foreheadRes.rgb;
    if (leftCheekRes?.rgb) regions.leftCheek = leftCheekRes.rgb;
    if (rightCheekRes?.rgb) regions.rightCheek = rightCheekRes.rgb;

    const fusedRgb = fuseRegionSamples([
      { sample: foreheadRes?.rgb ?? null, weight: 0.4 },
      { sample: leftCheekRes?.rgb ?? null, weight: 0.3 },
      { sample: rightCheekRes?.rgb ?? null, weight: 0.3 },
    ]);

    const activeResults = [foreheadRes, leftCheekRes, rightCheekRes].filter(
      (r): r is SkinExtractionResult => r !== null
    );
    const brightness =
      activeResults.length > 0
        ? activeResults.reduce((acc, r) => acc + r.brightness, 0) / activeResults.length
        : 0;
    const saturationFraction =
      activeResults.length > 0
        ? activeResults.reduce((acc, r) => acc + r.saturationFraction, 0) / activeResults.length
        : 0;

    return {
      timestampMs,
      faceDetected: Boolean(faceBox && fusedRgb && fusedRgb.pixelCount >= 20),
      fusedRgb,
      regions,
      faceBox,
      roiBoxes,
      motionScore,
      brightness,
      saturationFraction,
      trackingMode: 'mediapipe',
    };
  }

  private computeLandmarkMotion(faceBox: RoiBox | null, timestampMs: number): number {
    if (!faceBox) {
      this.prevGeometry = null;
      return 0;
    }

    const cx = faceBox.x + faceBox.width * 0.5;
    const cy = faceBox.y + faceBox.height * 0.5;
    const scale = Math.hypot(faceBox.width, faceBox.height);

    const prev = this.prevGeometry;
    this.prevGeometry = { cx, cy, scale, timestampMs };

    if (!prev) {
      return 0;
    }

    const dtMs = Math.max(8, timestampMs - prev.timestampMs);
    const dtScale = clamp(33.33 / dtMs, 0.35, 2.5);
    const normDenom = Math.max(scale, 0.15);

    const deltaPos = Math.hypot(cx - prev.cx, cy - prev.cy) / normDenom;
    const deltaScale = Math.abs(scale - prev.scale) / normDenom;

    return clamp((deltaPos * 18 + deltaScale * 14) * dtScale, 0, 5);
  }

  private extractWithSkinFallback(
    data: Uint8ClampedArray,
    width: number,
    height: number,
    timestampMs: number
  ): FrameRoiExtraction {
    const foreheadRes = extractSkinRgbFromImageData(data, width, height, FALLBACK_FOREHEAD_BOX);
    const leftCheekRes = extractSkinRgbFromImageData(data, width, height, FALLBACK_LEFT_CHEEK_BOX);
    const rightCheekRes = extractSkinRgbFromImageData(data, width, height, FALLBACK_RIGHT_CHEEK_BOX);

    const avgSkinRatio =
      (foreheadRes.skinPixelRatio * 0.4 +
        leftCheekRes.skinPixelRatio * 0.3 +
        rightCheekRes.skinPixelRatio * 0.3);

    const fusedRgb = fuseRegionSamples([
      { sample: foreheadRes.rgb, weight: 0.4 },
      { sample: leftCheekRes.rgb, weight: 0.3 },
      { sample: rightCheekRes.rgb, weight: 0.3 },
    ]);

    const brightness =
      foreheadRes.brightness * 0.4 +
      leftCheekRes.brightness * 0.3 +
      rightCheekRes.brightness * 0.3;
    const saturationFraction =
      foreheadRes.saturationFraction * 0.4 +
      leftCheekRes.saturationFraction * 0.3 +
      rightCheekRes.saturationFraction * 0.3;

    const motionScore = this.computeFallbackMotion(data, width, height, timestampMs);

    // Require meaningful skin presence inside the guide oval to report faceDetected
    const faceDetected = Boolean(
      fusedRgb && fusedRgb.pixelCount >= 60 && avgSkinRatio >= 0.14 && brightness >= 25
    );

    const regions: FrameRoiExtraction['regions'] = {};
    if (foreheadRes.rgb) regions.forehead = foreheadRes.rgb;
    if (leftCheekRes.rgb) regions.leftCheek = leftCheekRes.rgb;
    if (rightCheekRes.rgb) regions.rightCheek = rightCheekRes.rgb;

    return {
      timestampMs,
      faceDetected,
      fusedRgb: faceDetected ? fusedRgb : null,
      regions: faceDetected ? regions : {},
      faceBox: faceDetected ? FALLBACK_FACE_BOX : null,
      roiBoxes: faceDetected
        ? [FALLBACK_FOREHEAD_BOX, FALLBACK_LEFT_CHEEK_BOX, FALLBACK_RIGHT_CHEEK_BOX]
        : [],
      motionScore,
      brightness,
      saturationFraction,
      trackingMode: 'skin-fallback',
    };
  }

  private computeFallbackMotion(
    data: Uint8ClampedArray,
    width: number,
    height: number,
    timestampMs: number
  ): number {
    // Sample a 12x12 sparse luminance grid inside the centered face guide box
    const gridN = 12;
    const current = new Float32Array(gridN * gridN);

    for (let gy = 0; gy < gridN; gy++) {
      const py = clamp(
        Math.floor((FALLBACK_FACE_BOX.y + ((gy + 0.5) / gridN) * FALLBACK_FACE_BOX.height) * height),
        0,
        height - 1
      );
      for (let gx = 0; gx < gridN; gx++) {
        const px = clamp(
          Math.floor((FALLBACK_FACE_BOX.x + ((gx + 0.5) / gridN) * FALLBACK_FACE_BOX.width) * width),
          0,
          width - 1
        );
        const idx = (py * width + px) * 4;
        current[gy * gridN + gx] =
          0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      }
    }

    const prev = this.prevFallbackLuma;
    const prevTs = this.prevFallbackTimestampMs;
    this.prevFallbackLuma = current;
    this.prevFallbackTimestampMs = timestampMs;

    if (!prev || prevTs < 0) {
      return 0;
    }

    let diffSum = 0;
    for (let i = 0; i < current.length; i++) {
      diffSum += Math.abs(current[i] - prev[i]);
    }
    const meanAbsDiff = diffSum / current.length; // 0..255
    const dtMs = Math.max(8, timestampMs - prevTs);
    const dtScale = clamp(33.33 / dtMs, 0.35, 2.5);

    // ~15 luma levels/frame average change across face oval ≈ motionScore 1.0
    return clamp((meanAbsDiff / 15) * dtScale, 0, 5);
  }

  public reset(): void {
    this.prevGeometry = null;
    this.smoothedPolygons = {};
    this.prevFallbackLuma = null;
    this.prevFallbackTimestampMs = -1;
  }

  public dispose(): void {
    this.disposed = true;
    this.reset();
    if (this.landmarker) {
      try {
        this.landmarker.close();
      } catch {
        // Ignore cleanup errors
      }
      this.landmarker = null;
    }
    this.ctx = null;
    this.canvas = null;
  }
}
