import { describe, expect, it } from 'vitest';
import { RppgProcessor } from '../../rppg';
import type { RoiBox } from '../../types/rppg';
import { computeOffscreenDimensions, extractSkinRgbFromImageData } from '../faceRoiTracker';
import { SyntheticPulseGenerator } from '../syntheticFeed';
import { classifyCameraError } from '../useCamera';

function createRgbaBuffer(
  width: number,
  height: number,
  pixelAt: (x: number, y: number) => [number, number, number, number?]
): Uint8ClampedArray {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b, a = 255] = pixelAt(x, y);
      const idx = (y * width + x) * 4;
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = a;
    }
  }
  return data;
}

function dftMagnitudeAtFrequency(signal: number[], sampleRateHz: number, freqHz: number): number {
  const n = signal.length;
  const mean = signal.reduce((acc, v) => acc + v, 0) / n;
  let re = 0;
  let im = 0;
  for (let i = 0; i < n; i++) {
    const centered = signal[i] - mean;
    const angle = (2 * Math.PI * freqHz * i) / sampleRateHz;
    re += centered * Math.cos(angle);
    im -= centered * Math.sin(angle);
  }
  return (2 * Math.hypot(re, im)) / n;
}

describe('extractSkinRgbFromImageData', () => {
  const fullBox: RoiBox = { x: 0, y: 0, width: 1, height: 1, label: 'forehead' };

  it('extracts accurate mean RGB, brightness, and zero saturation for a pure skin ROI', () => {
    const width = 20;
    const height = 20;
    const skin: [number, number, number] = [190, 140, 110];
    const data = createRgbaBuffer(width, height, () => skin);

    const res = extractSkinRgbFromImageData(data, width, height, fullBox);

    expect(res.rgb).not.toBeNull();
    expect(res.rgb!.r).toBeCloseTo(190, 3);
    expect(res.rgb!.g).toBeCloseTo(140, 3);
    expect(res.rgb!.b).toBeCloseTo(110, 3);
    expect(res.rgb!.pixelCount).toBe(width * height);

    const expectedLuma = 0.299 * 190 + 0.587 * 140 + 0.114 * 110;
    expect(res.brightness).toBeCloseTo(expectedLuma, 3);
    expect(res.saturationFraction).toBe(0);
    expect(res.skinPixelRatio).toBe(1);
  });

  it('filters out non-skin background pixels via YCbCr bounds when skin pixels are present', () => {
    const width = 20;
    const height = 20;
    const skin: [number, number, number] = [196, 145, 115];
    const greenScreen: [number, number, number] = [25, 205, 30];
    const blueWall: [number, number, number] = [30, 50, 210];

    // Top half is skin (200 pixels), bottom-left is green (100 pixels), bottom-right is blue (100 pixels)
    const data = createRgbaBuffer(width, height, (x, y) => {
      if (y < 10) return skin;
      return x < 10 ? greenScreen : blueWall;
    });

    const res = extractSkinRgbFromImageData(data, width, height, fullBox);

    expect(res.rgb).not.toBeNull();
    expect(res.rgb!.pixelCount).toBe(200);
    expect(res.rgb!.r).toBeCloseTo(skin[0], 3);
    expect(res.rgb!.g).toBeCloseTo(skin[1], 3);
    expect(res.rgb!.b).toBeCloseTo(skin[2], 3);
    expect(res.saturationFraction).toBe(0);
    expect(res.skinPixelRatio).toBeCloseTo(0.5, 4);
  });

  it('rejects clipped highlights (> 245) and shadows (< 15) and reports saturationFraction', () => {
    const width = 20;
    const height = 10; // 200 pixels
    const skin: [number, number, number] = [185, 136, 108];

    // 50 clipped highlights (y < 5, x < 10), 50 clipped shadows (y >= 5, x < 10), 100 valid skin (x >= 10)
    const data = createRgbaBuffer(width, height, (x, y) => {
      if (x < 10 && y < 5) return [252, 250, 248];
      if (x < 10 && y >= 5) return [8, 10, 12];
      return skin;
    });

    const res = extractSkinRgbFromImageData(data, width, height, fullBox);

    expect(res.rgb).not.toBeNull();
    expect(res.rgb!.pixelCount).toBe(100);
    expect(res.rgb!.r).toBeCloseTo(skin[0], 3);
    expect(res.rgb!.g).toBeCloseTo(skin[1], 3);
    expect(res.rgb!.b).toBeCloseTo(skin[2], 3);
    expect(res.saturationFraction).toBeCloseTo(0.5, 4);
  });

  it('returns null rgb and saturationFraction === 1 when every pixel is clipped', () => {
    const width = 10;
    const height = 10;
    const data = createRgbaBuffer(width, height, (x) =>
      x < 5 ? [255, 255, 255] : [0, 0, 0]
    );

    const res = extractSkinRgbFromImageData(data, width, height, fullBox);

    expect(res.rgb).toBeNull();
    expect(res.saturationFraction).toBe(1);
    expect(res.skinPixelRatio).toBe(0);
  });

  it('gracefully falls back to non-clipped pixels under strongly tinted lighting', () => {
    const width = 16;
    const height = 16;
    // Cool cyan/blue-tinted skin where strict YCbCr rejects all pixels, but none are clipped
    const tinted: [number, number, number] = [95, 130, 165];
    const data = createRgbaBuffer(width, height, () => tinted);

    const res = extractSkinRgbFromImageData(data, width, height, fullBox);

    expect(res.skinPixelRatio).toBe(0);
    expect(res.saturationFraction).toBe(0);
    expect(res.rgb).not.toBeNull();
    expect(res.rgb!.pixelCount).toBe(width * height);
    expect(res.rgb!.r).toBeCloseTo(tinted[0], 3);
    expect(res.rgb!.g).toBeCloseTo(tinted[1], 3);
    expect(res.rgb!.b).toBeCloseTo(tinted[2], 3);
  });

  it('respects optional polygon masking inside RoiBox', () => {
    const width = 40;
    const height = 40;
    const insideColor: [number, number, number] = [195, 145, 115];
    const outsideColor: [number, number, number] = [160, 112, 86];

    // Inner square [0.25..0.75] has insideColor, outer border has outsideColor (both valid skin)
    const data = createRgbaBuffer(width, height, (x, y) => {
      const nx = (x + 0.5) / width;
      const ny = (y + 0.5) / height;
      return nx >= 0.25 && nx <= 0.75 && ny >= 0.25 && ny <= 0.75 ? insideColor : outsideColor;
    });

    const polyBox: RoiBox = {
      x: 0,
      y: 0,
      width: 1,
      height: 1,
      label: 'left-cheek',
      polygon: [
        { x: 0.25, y: 0.25 },
        { x: 0.75, y: 0.25 },
        { x: 0.75, y: 0.75 },
        { x: 0.25, y: 0.75 },
      ],
    };

    const res = extractSkinRgbFromImageData(data, width, height, polyBox);

    expect(res.rgb).not.toBeNull();
    expect(res.rgb!.pixelCount).toBe(400);
    expect(res.rgb!.r).toBeCloseTo(insideColor[0], 3);
    expect(res.rgb!.g).toBeCloseTo(insideColor[1], 3);
    expect(res.rgb!.b).toBeCloseTo(insideColor[2], 3);
  });

  it('trims bright specular forehead glare outliers from skin ROI averages', () => {
    const width = 20;
    const height = 20; // 400 pixels
    const skin: [number, number, number] = [182, 134, 106];
    // Bright skin-ish specular hotspot still within YCbCr & < 245, representing oily forehead glare
    const glare: [number, number, number] = [240, 212, 190];

    // 90% clean skin, 10% bright specular glare (top 2 rows)
    const data = createRgbaBuffer(width, height, (_x, y) => (y < 2 ? glare : skin));

    const res = extractSkinRgbFromImageData(data, width, height, fullBox);

    expect(res.rgb).not.toBeNull();
    // Trimmed mean should closely match true skin (134), rejecting the 212 glare pixels
    expect(res.rgb!.g).toBeCloseTo(skin[1], 1);
    expect(res.rgb!.pixelCount).toBe(360);
  });
});

describe('computeOffscreenDimensions', () => {
  it('preserves portrait mobile and landscape laptop aspect ratios cleanly', () => {
    expect(computeOffscreenDimensions(480, 640)).toEqual({ width: 270, height: 360 });
    expect(computeOffscreenDimensions(640, 480)).toEqual({ width: 360, height: 270 });
    expect(computeOffscreenDimensions(720, 1280)).toEqual({ width: 203, height: 360 });
  });
});

describe('SyntheticPulseGenerator', () => {
  it('generates valid FrameRoiExtraction objects with periodic green-dominant modulation at target BPM', () => {
    const targetBpm = 72;
    const sampleRateHz = 30;
    const durationSec = 12;
    const totalFrames = sampleRateHz * durationSec;

    const gen = new SyntheticPulseGenerator({ targetBpm, noiseLevel: 0.0004, seed: 42 });

    const rSeries: number[] = [];
    const gSeries: number[] = [];
    const bSeries: number[] = [];

    for (let i = 0; i < totalFrames; i++) {
      const ts = (i * 1000) / sampleRateHz;
      const frame = gen.generateFrame(ts);

      if (i === 0) {
        expect(frame.faceDetected).toBe(true);
        expect(frame.trackingMode).toBe('simulation');
        expect(frame.faceBox).not.toBeNull();
        expect(frame.roiBoxes).toHaveLength(3);
        expect(frame.regions.forehead).toBeDefined();
        expect(frame.regions.leftCheek).toBeDefined();
        expect(frame.regions.rightCheek).toBeDefined();
        expect(frame.saturationFraction).toBe(0);
        expect(frame.brightness).toBeGreaterThan(80);
        expect(frame.brightness).toBeLessThan(210);
      }

      expect(frame.fusedRgb).not.toBeNull();
      rSeries.push(frame.fusedRgb!.r);
      gSeries.push(frame.fusedRgb!.g);
      bSeries.push(frame.fusedRgb!.b);
    }

    const targetHz = targetBpm / 60;
    // Scan physiological band 0.7..3.5 Hz in 0.01 Hz steps to find dominant frequency in green channel
    let peakHz = 0;
    let peakMag = -1;
    for (let hz = 0.7; hz <= 3.5; hz += 0.01) {
      const mag = dftMagnitudeAtFrequency(gSeries, sampleRateHz, hz);
      if (mag > peakMag) {
        peakMag = mag;
        peakHz = hz;
      }
    }

    const estimatedBpm = peakHz * 60;
    expect(Math.abs(estimatedBpm - targetBpm)).toBeLessThan(1.0);

    // Check physiological AC/DC ratios: green (~1.2%) > blue (~0.6%) > red (~0.25%)
    const meanR = rSeries.reduce((a, b) => a + b, 0) / rSeries.length;
    const meanG = gSeries.reduce((a, b) => a + b, 0) / gSeries.length;
    const meanB = bSeries.reduce((a, b) => a + b, 0) / bSeries.length;

    const acDcR = dftMagnitudeAtFrequency(rSeries, sampleRateHz, targetHz) / meanR;
    const acDcG = dftMagnitudeAtFrequency(gSeries, sampleRateHz, targetHz) / meanG;
    const acDcB = dftMagnitudeAtFrequency(bSeries, sampleRateHz, targetHz) / meanB;

    expect(acDcG).toBeGreaterThan(acDcB * 1.6);
    expect(acDcB).toBeGreaterThan(acDcR * 1.8);
    expect(acDcG).toBeGreaterThan(0.008);
    expect(acDcG).toBeLessThan(0.015);
  });

  it('tracks dynamically updated BPM cleanly via setTargetBpm', () => {
    const sampleRateHz = 30;
    const gen = new SyntheticPulseGenerator({ targetBpm: 60, noiseLevel: 0.0002, seed: 7 });

    // Run 3 seconds at 60 BPM, then switch to 108 BPM and sample 10 seconds
    for (let i = 0; i < 90; i++) {
      gen.generateFrame((i * 1000) / sampleRateHz);
    }

    gen.setTargetBpm(108);
    expect(gen.getTargetBpm()).toBe(108);

    const gAfterSwitch: number[] = [];
    for (let i = 90; i < 390; i++) {
      const frame = gen.generateFrame((i * 1000) / sampleRateHz);
      gAfterSwitch.push(frame.fusedRgb!.g);
    }

    const magAt108 = dftMagnitudeAtFrequency(gAfterSwitch, sampleRateHz, 108 / 60);
    const magAt60 = dftMagnitudeAtFrequency(gAfterSwitch, sampleRateHz, 60 / 60);

    expect(magAt108).toBeGreaterThan(magAt60 * 6);
  });

  it('integrates end-to-end with RppgProcessor to estimate BPM accurately', () => {
    const targetBpm = 72;
    const sampleRateHz = 30;
    const gen = new SyntheticPulseGenerator({ targetBpm, noiseLevel: 0.0005, seed: 123 });
    const processor = new RppgProcessor();

    let latest = processor.getEstimate();
    for (let i = 0; i < sampleRateHz * 13; i++) {
      const frame = gen.generateFrame((i * 1000) / sampleRateHz);
      latest = processor.pushSample(frame);
    }

    expect(latest.status).toBe('measuring');
    expect(latest.bpm).not.toBeNull();
    expect(Math.abs(latest.bpm! - targetBpm)).toBeLessThan(2.5);
    expect(latest.confidence).toBeGreaterThanOrEqual(70);
  });
});

describe('classifyCameraError', () => {
  it('maps standard browser MediaDevices error names to clear CameraErrorInfo codes', () => {
    expect(classifyCameraError({ name: 'NotAllowedError' }).code).toBe('permission-denied');
    expect(classifyCameraError({ name: 'NotFoundError' }).code).toBe('no-camera');
    expect(classifyCameraError({ name: 'NotReadableError' }).code).toBe('camera-in-use');
    expect(classifyCameraError(new Error('weird failure')).code).toBe('unknown');
  });
});
