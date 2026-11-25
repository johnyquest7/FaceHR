import type { FrameRoiExtraction, RgbSample, RoiBox } from '../types/rppg';

export interface SyntheticPulseOptions {
  /** Initial target heart rate in BPM (default: 72) */
  targetBpm?: number;
  /** Relative sensor noise std-dev as a fraction of channel intensity (default: 0.0006) */
  noiseLevel?: number;
  /** Optional deterministic PRNG seed */
  seed?: number;
}

const DEFAULT_BPM = 72;
const MIN_BPM = 40;
const MAX_BPM = 200;

// Baseline warm human skin RGB values (in 0..255)
const BASE_SKIN_R = 192;
const BASE_SKIN_G = 144;
const BASE_SKIN_B = 116;

// Physiologically realistic rPPG AC/DC modulation depths (hemoglobin absorption profile)
const AC_DEPTH_R = 0.0025; // ~0.25%
const AC_DEPTH_G = 0.012; // ~1.20% (strongest oxyhemoglobin absorption)
const AC_DEPTH_B = 0.006; // ~0.60%

const SIM_FACE_BOX: RoiBox = {
  x: 0.27,
  y: 0.14,
  width: 0.46,
  height: 0.66,
  label: 'face',
};

const SIM_FOREHEAD_BOX: RoiBox = {
  x: 0.37,
  y: 0.21,
  width: 0.26,
  height: 0.12,
  label: 'forehead',
};

const SIM_RIGHT_CHEEK_BOX: RoiBox = {
  x: 0.33,
  y: 0.45,
  width: 0.13,
  height: 0.14,
  label: 'right-cheek',
};

const SIM_LEFT_CHEEK_BOX: RoiBox = {
  x: 0.54,
  y: 0.45,
  width: 0.13,
  height: 0.14,
  label: 'left-cheek',
};

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

/**
 * Generates realistic facial rPPG RGB samples and an optional stylized portrait canvas
 * modulated by a physiological blood-volume pulse wave at `targetBpm`.
 */
export class SyntheticPulseGenerator {
  private targetBpm: number;
  private noiseLevel: number;
  private rngState: number;
  private initialSeed: number;

  private cardiacPhaseRad = 0;
  private respPhaseRad = 0;
  private lastTimestampMs: number | null = null;

  constructor(options: SyntheticPulseOptions = {}) {
    this.targetBpm = clamp(options.targetBpm ?? DEFAULT_BPM, MIN_BPM, MAX_BPM);
    this.noiseLevel = Math.max(0, options.noiseLevel ?? 0.0006);
    this.initialSeed = (options.seed ?? 0x51a7c3d1) >>> 0;
    this.rngState = this.initialSeed;
  }

  public getTargetBpm(): number {
    return this.targetBpm;
  }

  public setTargetBpm(bpm: number): void {
    if (!Number.isFinite(bpm)) return;
    this.targetBpm = clamp(bpm, MIN_BPM, MAX_BPM);
  }

  public reset(): void {
    this.cardiacPhaseRad = 0;
    this.respPhaseRad = 0;
    this.lastTimestampMs = null;
    this.rngState = this.initialSeed;
  }

  /** Fast deterministic PRNG (Mulberry32) yielding approximately zero-mean [-1, 1] noise */
  private nextNoise(): number {
    let t = (this.rngState += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    const u1 = ((t ^ (t >>> 14)) >>> 0) / 4294967296;

    t = (this.rngState += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    const u2 = ((t ^ (t >>> 14)) >>> 0) / 4294967296;

    // Triangular distribution in [-1, 1] with zero mean
    return u1 - u2;
  }

  /**
   * Advances cardiac & respiratory oscillators smoothly and computes a normalized
   * physiological blood-volume pulse waveform value in roughly [-1, 1].
   */
  private advancePulseWave(timestampMs: number): number {
    if (this.lastTimestampMs !== null) {
      const rawDtSec = (timestampMs - this.lastTimestampMs) / 1000;
      const dtSec = clamp(rawDtSec, 0, 0.25);
      const freqHz = this.targetBpm / 60;
      this.cardiacPhaseRad = (this.cardiacPhaseRad + 2 * Math.PI * freqHz * dtSec) % (2 * Math.PI);
      // Gentle ~0.22 Hz respiratory baseline modulation
      this.respPhaseRad = (this.respPhaseRad + 2 * Math.PI * 0.22 * dtSec) % (2 * Math.PI);
    }
    this.lastTimestampMs = timestampMs;

    // Fundamental systolic wave + gentle dicrotic 2nd harmonic (fundamental remains strongly dominant)
    const cardiacWave =
      (Math.sin(this.cardiacPhaseRad) + 0.2 * Math.sin(2 * this.cardiacPhaseRad - 0.65)) / 1.15;

    return cardiacWave;
  }

  private createRegionSample(
    pulseWave: number,
    respDrift: number,
    lumaScale: number,
    pixelCount: number
  ): RgbSample {
    const noiseR = this.nextNoise() * this.noiseLevel;
    const noiseG = this.nextNoise() * this.noiseLevel;
    const noiseB = this.nextNoise() * this.noiseLevel;

    // Green has strongest BVP modulation (~1.2%), blue moderate (~0.6%), red weakest (~0.25%)
    const r = clamp(
      BASE_SKIN_R * lumaScale * (1 + AC_DEPTH_R * pulseWave + respDrift + noiseR),
      15,
      245
    );
    const g = clamp(
      BASE_SKIN_G * lumaScale * (1 + AC_DEPTH_G * pulseWave + respDrift + noiseG),
      15,
      245
    );
    const b = clamp(
      BASE_SKIN_B * lumaScale * (1 + AC_DEPTH_B * pulseWave + respDrift + noiseB),
      15,
      245
    );

    return { r, g, b, pixelCount };
  }

  /**
   * Produces a synthetic FrameRoiExtraction without touching any DOM APIs.
   */
  public generateFrame(timestampMs: number): FrameRoiExtraction {
    const pulseWave = this.advancePulseWave(timestampMs);
    const respDrift = 0.0012 * Math.sin(this.respPhaseRad);

    const forehead = this.createRegionSample(pulseWave, respDrift, 1.03, 980);
    const leftCheek = this.createRegionSample(pulseWave, respDrift, 0.99, 640);
    const rightCheek = this.createRegionSample(pulseWave, respDrift, 0.98, 640);

    const fusedRgb: RgbSample = {
      r: forehead.r * 0.4 + leftCheek.r * 0.3 + rightCheek.r * 0.3,
      g: forehead.g * 0.4 + leftCheek.g * 0.3 + rightCheek.g * 0.3,
      b: forehead.b * 0.4 + leftCheek.b * 0.3 + rightCheek.b * 0.3,
      pixelCount: forehead.pixelCount + leftCheek.pixelCount + rightCheek.pixelCount,
    };

    const brightness = 0.299 * fusedRgb.r + 0.587 * fusedRgb.g + 0.114 * fusedRgb.b;

    return {
      timestampMs,
      faceDetected: true,
      fusedRgb,
      regions: {
        forehead,
        leftCheek,
        rightCheek,
      },
      faceBox: SIM_FACE_BOX,
      roiBoxes: [SIM_FOREHEAD_BOX, SIM_LEFT_CHEEK_BOX, SIM_RIGHT_CHEEK_BOX],
      motionScore: 0.02 + Math.abs(Math.sin(this.respPhaseRad)) * 0.015,
      brightness,
      saturationFraction: 0,
      trackingMode: 'simulation',
    };
  }

  /**
   * Renders a soft stylized face silhouette with live physiological skin-tone pulse modulation
   * onto `canvas`, and returns the corresponding FrameRoiExtraction for the rPPG pipeline.
   */
  public renderToCanvas(canvas: HTMLCanvasElement, timestampMs: number): FrameRoiExtraction {
    const extraction = this.generateFrame(timestampMs);
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return extraction;
    }

    if (canvas.width === 0 || canvas.height === 0) {
      canvas.width = 480;
      canvas.height = 640;
    }

    const w = canvas.width;
    const h = canvas.height;
    const rgb = extraction.fusedRgb ?? {
      r: BASE_SKIN_R,
      g: BASE_SKIN_G,
      b: BASE_SKIN_B,
      pixelCount: 1,
    };

    // Subtle visual amplification of pulse warmth so the simulation feels alive on screen
    const pulseDelta = (rgb.g - BASE_SKIN_G) / (BASE_SKIN_G * AC_DEPTH_G);
    const visR = Math.round(clamp(rgb.r + pulseDelta * 2.5, 0, 255));
    const visG = Math.round(clamp(rgb.g + pulseDelta * 3.5, 0, 255));
    const visB = Math.round(clamp(rgb.b + pulseDelta * 1.5, 0, 255));

    // Studio dark radial background
    const bgGrad = ctx.createRadialGradient(w * 0.5, h * 0.42, w * 0.06, w * 0.5, h * 0.48, w * 0.85);
    bgGrad.addColorStop(0, '#182438');
    bgGrad.addColorStop(0.55, '#0f1726');
    bgGrad.addColorStop(1, '#080c15');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    // Subtle pulsing biometric halo behind head
    const haloAlpha = 0.08 + 0.04 * clamp(pulseDelta * 0.5 + 0.5, 0, 1);
    const haloGrad = ctx.createRadialGradient(w * 0.5, h * 0.46, w * 0.12, w * 0.5, h * 0.46, w * 0.46);
    haloGrad.addColorStop(0, `rgba(56, 189, 248, ${haloAlpha.toFixed(3)})`);
    haloGrad.addColorStop(1, 'rgba(56, 189, 248, 0)');
    ctx.fillStyle = haloGrad;
    ctx.fillRect(0, 0, w, h);

    const cx = (SIM_FACE_BOX.x + SIM_FACE_BOX.width * 0.5) * w;
    const cy = (SIM_FACE_BOX.y + SIM_FACE_BOX.height * 0.5) * h;
    const rx = SIM_FACE_BOX.width * 0.48 * w;
    const ry = SIM_FACE_BOX.height * 0.49 * h;

    // Shoulders silhouette
    ctx.save();
    const shoulderGrad = ctx.createLinearGradient(cx, cy + ry * 0.85, cx, h);
    shoulderGrad.addColorStop(0, '#1e293b');
    shoulderGrad.addColorStop(1, '#0f172a');
    ctx.fillStyle = shoulderGrad;
    ctx.beginPath();
    ctx.ellipse(cx, h * 1.03, w * 0.44, h * 0.24, 0, Math.PI, 2 * Math.PI);
    ctx.fill();

    // Neck
    const neckGrad = ctx.createLinearGradient(cx, cy + ry * 0.65, cx, cy + ry * 1.25);
    neckGrad.addColorStop(0, `rgb(${Math.round(visR * 0.78)}, ${Math.round(visG * 0.76)}, ${Math.round(visB * 0.76)})`);
    neckGrad.addColorStop(1, `rgb(${Math.round(visR * 0.65)}, ${Math.round(visG * 0.64)}, ${Math.round(visB * 0.65)})`);
    ctx.fillStyle = neckGrad;
    ctx.beginPath();
    ctx.roundRect(cx - rx * 0.34, cy + ry * 0.68, rx * 0.68, ry * 0.52, rx * 0.18);
    ctx.fill();

    // Head oval with soft 3D skin shading
    const faceGrad = ctx.createRadialGradient(
      cx - rx * 0.14,
      cy - ry * 0.22,
      rx * 0.1,
      cx,
      cy,
      ry * 1.05
    );
    faceGrad.addColorStop(0, `rgb(${Math.min(242, visR + 14)}, ${Math.min(235, visG + 12)}, ${Math.min(225, visB + 10)})`);
    faceGrad.addColorStop(0.65, `rgb(${visR}, ${visG}, ${visB})`);
    faceGrad.addColorStop(1, `rgb(${Math.round(visR * 0.72)}, ${Math.round(visG * 0.70)}, ${Math.round(visB * 0.70)})`);

    ctx.fillStyle = faceGrad;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();

    // Soft forehead highlight
    const foreheadGlow = ctx.createRadialGradient(
      cx,
      cy - ry * 0.48,
      rx * 0.05,
      cx,
      cy - ry * 0.45,
      rx * 0.55
    );
    foreheadGlow.addColorStop(0, 'rgba(255, 244, 234, 0.16)');
    foreheadGlow.addColorStop(1, 'rgba(255, 244, 234, 0)');
    ctx.fillStyle = foreheadGlow;
    ctx.beginPath();
    ctx.ellipse(cx, cy - ry * 0.46, rx * 0.55, ry * 0.22, 0, 0, Math.PI * 2);
    ctx.fill();

    // Soft rosy vascular cheek tint (pulses subtly with BVP)
    const cheekAlpha = 0.09 + 0.05 * clamp(pulseDelta * 0.5 + 0.5, 0, 1);
    for (const dir of [-1, 1]) {
      const cheekX = cx + dir * rx * 0.46;
      const cheekY = cy + ry * 0.12;
      const cheekGrad = ctx.createRadialGradient(cheekX, cheekY, 2, cheekX, cheekY, rx * 0.34);
      cheekGrad.addColorStop(0, `rgba(244, 114, 128, ${cheekAlpha.toFixed(3)})`);
      cheekGrad.addColorStop(1, 'rgba(244, 114, 128, 0)');
      ctx.fillStyle = cheekGrad;
      ctx.beginPath();
      ctx.arc(cheekX, cheekY, rx * 0.34, 0, Math.PI * 2);
      ctx.fill();
    }

    // Minimalist soft facial landmarks (brows, calm eyes, nose bridge, subtle mouth)
    ctx.strokeStyle = 'rgba(58, 42, 36, 0.24)';
    ctx.lineCap = 'round';
    ctx.lineWidth = Math.max(2, w * 0.005);

    // Subtle eyebrows
    for (const dir of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(cx + dir * rx * 0.2, cy - ry * 0.16);
      ctx.quadraticCurveTo(cx + dir * rx * 0.42, cy - ry * 0.22, cx + dir * rx * 0.62, cy - ry * 0.14);
      ctx.stroke();
    }

    // Soft eyes
    ctx.fillStyle = 'rgba(42, 32, 30, 0.28)';
    for (const dir of [-1, 1]) {
      ctx.beginPath();
      ctx.ellipse(cx + dir * rx * 0.38, cy - ry * 0.04, rx * 0.11, ry * 0.032, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // Soft nose shadow
    ctx.strokeStyle = 'rgba(92, 62, 52, 0.18)';
    ctx.lineWidth = Math.max(1.5, w * 0.004);
    ctx.beginPath();
    ctx.moveTo(cx, cy - ry * 0.03);
    ctx.lineTo(cx - rx * 0.04, cy + ry * 0.2);
    ctx.quadraticCurveTo(cx, cy + ry * 0.24, cx + rx * 0.06, cy + ry * 0.2);
    ctx.stroke();

    // Soft calm mouth curve
    ctx.strokeStyle = 'rgba(122, 68, 64, 0.26)';
    ctx.lineWidth = Math.max(2, w * 0.005);
    ctx.beginPath();
    ctx.moveTo(cx - rx * 0.22, cy + ry * 0.43);
    ctx.quadraticCurveTo(cx, cy + ry * 0.48, cx + rx * 0.22, cy + ry * 0.43);
    ctx.stroke();

    ctx.restore();
    return extraction;
  }
}
