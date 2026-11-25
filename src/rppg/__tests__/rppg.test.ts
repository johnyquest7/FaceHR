import { describe, expect, it } from 'vitest';
import type { FrameRoiExtraction, RgbSample } from '../../types/rppg';
import {
  BpmTemporalTracker,
  RppgProcessor,
  applyHannWindow,
  biquadFrequencyResponseMagnitude,
  butterworthBandpassZeroPhase,
  computeSpectrum,
  createIdleEstimate,
  despikeRgbSeries,
  designButterworthHighpass2,
  designButterworthLowpass2,
  estimateSpectralBpm,
  extractOmitPulse,
  extractPosPulse,
  getBiquadPoleRadius,
  linearDetrend,
  parabolicInterpolatePeak,
  resampleRgbBuffer,
} from '../index';

/** Deterministic PRNG (Mulberry32) for zero-flake tests */
function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianNoise(rng: () => number): number {
  const u1 = Math.max(1e-9, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

interface SyntheticStreamOptions {
  bpm: number;
  durationSec: number;
  fps?: number;
  jitterFpsRange?: [number, number];
  /** Relative AC/DC modulation depths */
  greenAcDepth?: number;
  blueAcDepth?: number;
  redAcDepth?: number;
  /** Optional 2nd harmonic relative amplitude (multiplier of fundamental) */
  secondHarmonicRatio?: number;
  /** Additive linear + low-freq lighting drift amplitude (in 0..255 intensity units) */
  lightingDriftAmplitude?: number;
  /** In-band specular motion artifact (multiplicative, equal across RGB) */
  specularMotionAmplitude?: number;
  specularMotionFreqHz?: number;
  /** Sensor Gaussian noise std in intensity units */
  noiseStd?: number;
  motionScore?: number;
  brightness?: number;
  saturationFraction?: number;
  faceDetected?: boolean;
  seed?: number;
}

function generateSyntheticExtractions(
  options: SyntheticStreamOptions,
): FrameRoiExtraction[] {
  const {
    bpm,
    durationSec,
    fps = 30,
    jitterFpsRange,
    greenAcDepth = 0.011,
    blueAcDepth = 0.0055,
    redAcDepth = 0.0025,
    secondHarmonicRatio = 0,
    lightingDriftAmplitude = 0,
    specularMotionAmplitude = 0,
    specularMotionFreqHz = 1.9,
    noiseStd = 0.08,
    motionScore = 0.04,
    brightness = 135,
    saturationFraction = 0.01,
    faceDetected = true,
    seed = 42,
  } = options;

  const rng = createRng(seed);
  const f0 = bpm / 60;
  const frames: FrameRoiExtraction[] = [];

  // Realistic skin baseline scaled to requested brightness
  const lumScale = brightness / 135;
  const baseR = 176 * lumScale;
  const baseG = 130 * lumScale;
  const baseB = 104 * lumScale;

  let tMs = 1000;
  const endMs = 1000 + durationSec * 1000;

  while (tMs <= endMs) {
    const tSec = (tMs - 1000) / 1000;

    // Physiological pulse wave (fundamental + optional 2nd harmonic)
    const pulse =
      Math.sin(2 * Math.PI * f0 * tSec) +
      secondHarmonicRatio * Math.sin(2 * Math.PI * 2 * f0 * tSec + 0.4);

    // Slow additive lighting drift (linear ramp + slow ambient oscillation)
    const drift =
      lightingDriftAmplitude > 0
        ? lightingDriftAmplitude * (0.6 * (tSec / durationSec) + 0.4 * Math.sin(2 * Math.PI * 0.14 * tSec))
        : 0;

    // Specular / motion-induced intensity modulation (shared across R/G/B)
    const specular =
      specularMotionAmplitude > 0
        ? 1 + specularMotionAmplitude * Math.sin(2 * Math.PI * specularMotionFreqHz * tSec + 0.8)
        : 1;

    const makeRegion = (offset: number, pixelCount: number): RgbSample => ({
      r: Math.max(1, (baseR * (1 + redAcDepth * pulse) * specular) + drift + offset + noiseStd * gaussianNoise(rng)),
      g: Math.max(1, (baseG * (1 + greenAcDepth * pulse) * specular) + drift + offset + noiseStd * gaussianNoise(rng)),
      b: Math.max(1, (baseB * (1 + blueAcDepth * pulse) * specular) + drift + offset + noiseStd * gaussianNoise(rng)),
      pixelCount,
    });

    const forehead = makeRegion(2.5, 420);
    const leftCheek = makeRegion(-1.5, 340);
    const rightCheek = makeRegion(-1.0, 350);

    const fusedRgb: RgbSample = {
      r: 0.4 * forehead.r + 0.3 * leftCheek.r + 0.3 * rightCheek.r,
      g: 0.4 * forehead.g + 0.3 * leftCheek.g + 0.3 * rightCheek.g,
      b: 0.4 * forehead.b + 0.3 * leftCheek.b + 0.3 * rightCheek.b,
      pixelCount: forehead.pixelCount + leftCheek.pixelCount + rightCheek.pixelCount,
    };

    frames.push({
      timestampMs: tMs,
      faceDetected,
      fusedRgb: faceDetected ? fusedRgb : null,
      regions: faceDetected ? { forehead, leftCheek, rightCheek } : {},
      faceBox: faceDetected
        ? { x: 0.25, y: 0.15, width: 0.5, height: 0.65, label: 'face' }
        : null,
      roiBoxes: [],
      motionScore,
      brightness,
      saturationFraction,
      trackingMode: 'mediapipe',
    });

    if (jitterFpsRange) {
      const [minFps, maxFps] = jitterFpsRange;
      const instFps = minFps + rng() * (maxFps - minFps);
      tMs += 1000 / instFps;
    } else {
      tMs += 1000 / fps;
    }
  }

  return frames;
}

describe('rPPG DSP primitives', () => {
  it('designs numerically stable Butterworth bandpass sections at 30 Hz with expected gain', () => {
    const fs = 30;
    const hp = designButterworthHighpass2(0.7, fs);
    const lp = designButterworthLowpass2(3.5, fs);

    // Strictly inside unit circle
    expect(getBiquadPoleRadius(hp)).toBeLessThan(0.95);
    expect(getBiquadPoleRadius(lp)).toBeLessThan(0.95);

    const combinedGain = (freqHz: number) => {
      const gHp = biquadFrequencyResponseMagnitude(hp, freqHz, fs);
      const gLp = biquadFrequencyResponseMagnitude(lp, freqHz, fs);
      // Forward-backward filtfilt squares magnitude response
      return (gHp * gLp) ** 2;
    };

    // Passband flatness
    expect(combinedGain(1.2)).toBeGreaterThan(0.8);
    expect(combinedGain(1.6)).toBeGreaterThan(0.85);
    expect(combinedGain(2.2)).toBeGreaterThan(0.8);

    // Strong stopband rejection at low drift and high-frequency noise
    expect(combinedGain(0.15)).toBeLessThan(0.01);
    expect(combinedGain(8.0)).toBeLessThan(0.05);

    // Zero-phase check on a 1.5 Hz sinusoid
    const n = 300;
    const input = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      input[i] = Math.sin((2 * Math.PI * 1.5 * i) / fs);
    }
    const filtered = butterworthBandpassZeroPhase(input, fs, 0.7, 3.5);
    // Check middle segment correlation is near +1 (zero phase shift)
    let dot = 0;
    let normIn = 0;
    let normOut = 0;
    for (let i = 60; i < 240; i++) {
      dot += input[i] * filtered[i];
      normIn += input[i] * input[i];
      normOut += filtered[i] * filtered[i];
    }
    const corr = dot / Math.sqrt(normIn * normOut);
    expect(corr).toBeGreaterThan(0.995);
  });

  it('achieves sub-0.2 BPM frequency precision with radix-2 FFT + parabolic interpolation', () => {
    const fs = 30;
    const trueBpm = 77.35; // intentionally between FFT bins
    const trueHz = trueBpm / 60;
    const n = 360; // 12 s
    const signal = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      signal[i] = Math.sin((2 * Math.PI * trueHz * i) / fs);
    }

    const windowed = applyHannWindow(linearDetrend(signal));
    const spectrum = computeSpectrum(windowed, fs, 4096);

    let peakIdx = 0;
    let maxMag = 0;
    for (let i = 1; i < spectrum.magnitudes.length - 1; i++) {
      if (spectrum.magnitudes[i] > maxMag) {
        maxMag = spectrum.magnitudes[i];
        peakIdx = i;
      }
    }

    const refined = parabolicInterpolatePeak(spectrum.magnitudes, peakIdx);
    const estBpm = refined.index * spectrum.binResolutionHz * 60;
    expect(Math.abs(estBpm - trueBpm)).toBeLessThan(0.2);
  });
});

describe('RppgProcessor end-to-end estimation', () => {
  it.each([52, 72, 96, 132])(
    'estimates synthetic rPPG heart rate at %i BPM within ±2.5 BPM',
    (targetBpm) => {
      const processor = new RppgProcessor();
      const frames = generateSyntheticExtractions({
        bpm: targetBpm,
        durationSec: 12,
        fps: 30,
        seed: 100 + targetBpm,
      });

      let latest = createIdleEstimate();
      for (const frame of frames) {
        latest = processor.pushSample(frame);
      }

      expect(latest.status).toBe('measuring');
      expect(latest.bpm).not.toBeNull();
      expect(Math.abs((latest.bpm ?? 0) - targetBpm)).toBeLessThanOrEqual(2.5);
      expect(latest.rawPosBpm).not.toBeNull();
      expect(latest.rawChromBpm).not.toBeNull();
      expect(Math.abs((latest.rawPosBpm ?? 0) - targetBpm)).toBeLessThanOrEqual(2.5);
      expect(Math.abs((latest.rawChromBpm ?? 0) - targetBpm)).toBeLessThanOrEqual(2.5);
      expect(latest.confidence).toBeGreaterThanOrEqual(70);
      expect(['good', 'excellent']).toContain(latest.quality);

      // Verify normalized waveform payload (~120 samples in [-1, 1])
      expect(latest.waveform.length).toBe(120);
      for (const v of latest.waveform) {
        expect(v).toBeGreaterThanOrEqual(-1);
        expect(v).toBeLessThanOrEqual(1);
      }
    },
  );

  it('handles variable mobile frame rate jitter (22–30 FPS) within ±3 BPM', () => {
    const targetBpm = 84;
    const processor = new RppgProcessor();
    const frames = generateSyntheticExtractions({
      bpm: targetBpm,
      durationSec: 13,
      jitterFpsRange: [22, 30],
      seed: 777,
    });

    let latest = createIdleEstimate();
    for (const frame of frames) {
      latest = processor.pushSample(frame);
    }

    expect(latest.status).toBe('measuring');
    expect(latest.bpm).not.toBeNull();
    expect(Math.abs((latest.bpm ?? 0) - targetBpm)).toBeLessThanOrEqual(3.0);
    expect(latest.diagnostics.fps).toBeGreaterThanOrEqual(21);
    expect(latest.diagnostics.fps).toBeLessThanOrEqual(31);
  });

  it('recovers heart rate with POS under additive lighting drift and specular motion noise', () => {
    const targetBpm = 72;
    const frames = generateSyntheticExtractions({
      bpm: targetBpm,
      durationSec: 13,
      fps: 30,
      lightingDriftAmplitude: 24,
      specularMotionAmplitude: 0.04, // ~4% specular motion at 114 BPM (1.9 Hz)
      specularMotionFreqHz: 1.9,
      noiseStd: 0.12,
      seed: 901,
    });

    // Verify POS directly on the resampled RGB series
    const uniform = resampleRgbBuffer(
      frames.map((f) => ({
        timestampMs: f.timestampMs,
        r: f.fusedRgb!.r,
        g: f.fusedRgb!.g,
        b: f.fusedRgb!.b,
      })),
      30,
    );
    const posPulse = extractPosPulse(uniform);
    const posSpectral = estimateSpectralBpm(posPulse, 30);
    expect(posSpectral.bpm).not.toBeNull();
    expect(Math.abs((posSpectral.bpm ?? 0) - targetBpm)).toBeLessThanOrEqual(2.5);

    // Also verify end-to-end processor
    const processor = new RppgProcessor();
    let latest = createIdleEstimate();
    for (const frame of frames) {
      latest = processor.pushSample(frame);
    }
    expect(latest.bpm).not.toBeNull();
    expect(Math.abs((latest.bpm ?? 0) - targetBpm)).toBeLessThanOrEqual(2.5);
  });

  it('disambiguates strong 2nd harmonic without producing 2x BPM error', () => {
    const targetBpm = 68; // 2nd harmonic at 136 BPM
    const processor = new RppgProcessor();
    const frames = generateSyntheticExtractions({
      bpm: targetBpm,
      durationSec: 13,
      fps: 30,
      // 2nd harmonic slightly stronger than fundamental!
      secondHarmonicRatio: 1.2,
      seed: 515,
    });

    let latest = createIdleEstimate();
    for (const frame of frames) {
      latest = processor.pushSample(frame);
    }

    expect(latest.bpm).not.toBeNull();
    expect(Math.abs((latest.bpm ?? 0) - targetBpm)).toBeLessThanOrEqual(2.5);
    expect(latest.rawPosBpm).not.toBeNull();
    expect(Math.abs((latest.rawPosBpm ?? 0) - targetBpm)).toBeLessThanOrEqual(2.5);
  });

  it('rejects unphysiological instantaneous BPM jumps while allowing sustained high-SNR shifts', () => {
    const tracker = new BpmTemporalTracker();

    // Initialize at 72 BPM
    expect(tracker.update(72, 6000, 10, 85)).toBeCloseTo(72, 1);
    expect(tracker.update(72.5, 6200, 10, 85)).toBeCloseTo(72.1, 0);

    // Brief +45 BPM spike (< 1 s) must be rejected
    const afterSpike1 = tracker.update(118, 6400, 8, 80);
    const afterSpike2 = tracker.update(120, 6600, 8, 80);
    expect(Math.abs((afterSpike1 ?? 0) - 72)).toBeLessThan(3);
    expect(Math.abs((afterSpike2 ?? 0) - 72)).toBeLessThan(3);

    // Sustained high-SNR shift over > 1.2 s should eventually be accepted
    let tracked = afterSpike2;
    for (let t = 6800; t <= 8400; t += 200) {
      tracked = tracker.update(118, t, 10, 85);
    }
    expect(tracked).not.toBeNull();
    expect(tracked!).toBeGreaterThan(95);
  });

  it('reports warm-up, no-face, excessive motion, and dark lighting with degraded confidence and clear guidance', () => {
    const processor = new RppgProcessor();

    // 1. Warm-up (< 6 seconds)
    const shortFrames = generateSyntheticExtractions({
      bpm: 72,
      durationSec: 3.0,
      fps: 30,
      seed: 11,
    });
    let est = createIdleEstimate();
    for (const f of shortFrames) {
      est = processor.pushSample(f);
    }
    expect(est.status).toBe('warming-up');
    expect(est.bpm).toBeNull();
    expect(est.diagnostics.bufferProgress).toBeGreaterThan(0.15);
    expect(est.diagnostics.bufferProgress).toBeLessThan(0.4);

    // 2. No-face
    const noFaceFrame: FrameRoiExtraction = {
      timestampMs: shortFrames[shortFrames.length - 1].timestampMs + 33,
      faceDetected: false,
      fusedRgb: null,
      regions: {},
      faceBox: null,
      roiBoxes: [],
      motionScore: 0,
      brightness: 120,
      saturationFraction: 0,
      trackingMode: 'mediapipe',
    };
    const noFaceEst = processor.pushSample(noFaceFrame);
    expect(noFaceEst.status).toBe('no-face');
    expect(noFaceEst.confidence).toBe(0);
    expect(noFaceEst.quality).toBe('poor');
    expect(noFaceEst.bpm).toBeNull();
    expect(noFaceEst.guidance.toLowerCase()).toContain('face');

    // 3. Excessive motion
    processor.reset();
    expect(processor.getEstimate().status).toBe('idle');

    const shakyFrames = generateSyntheticExtractions({
      bpm: 75,
      durationSec: 8,
      fps: 30,
      motionScore: 0.92,
      seed: 22,
    });
    for (const f of shakyFrames) {
      est = processor.pushSample(f);
    }
    expect(est.status).toBe('low-signal');
    expect(est.confidence).toBeLessThanOrEqual(35);
    expect(est.quality).toBe('poor');
    expect(est.diagnostics.motionStability).toBeLessThan(25);
    expect(est.guidance.toLowerCase()).toContain('still');

    // 4. Dark lighting
    processor.reset();
    const darkFrames = generateSyntheticExtractions({
      bpm: 75,
      durationSec: 8,
      fps: 30,
      brightness: 24,
      motionScore: 0.05,
      seed: 33,
    });
    for (const f of darkFrames) {
      est = processor.pushSample(f);
    }
    expect(est.status).toBe('low-signal');
    expect(est.confidence).toBeLessThanOrEqual(35);
    expect(est.quality).toBe('poor');
    expect(est.diagnostics.lightingScore).toBeLessThan(25);
    expect(est.guidance.toLowerCase()).toMatch(/dark|bright|light/);
  });

  it('recovers accurate BPM with OMIT across light and dark skin tones under specular motion', () => {
    const targetBpm = 78;
    const f0 = targetBpm / 60;
    const fs = 30;
    const n = fs * 12;

    const skinProfiles = [
      { label: 'dark', baseR: 95, baseG: 68, baseB: 52 },
      { label: 'light', baseR: 215, baseG: 175, baseB: 150 },
    ];

    for (const profile of skinProfiles) {
      const r = new Float64Array(n);
      const g = new Float64Array(n);
      const b = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const t = i / fs;
        const pulse = Math.sin(2 * Math.PI * f0 * t);
        // Strong 8% multiplicative specular/AE oscillation at 1.95 Hz (117 BPM)
        const specular = 1 + 0.08 * Math.sin(2 * Math.PI * 1.95 * t + 0.5);
        r[i] = profile.baseR * (1 + 0.0025 * pulse) * specular;
        g[i] = profile.baseG * (1 + 0.011 * pulse) * specular;
        b[i] = profile.baseB * (1 + 0.0055 * pulse) * specular;
      }

      const omitPulse = extractOmitPulse({
        sampleRateHz: fs,
        r,
        g,
        b,
        durationSec: (n - 1) / fs,
        startTimestampMs: 0,
      });
      const est = estimateSpectralBpm(omitPulse, fs);
      expect(est.bpm).not.toBeNull();
      expect(Math.abs((est.bpm ?? 0) - targetBpm)).toBeLessThanOrEqual(2.0);
    }
  });

  it('suppresses sudden mobile auto-exposure step jumps via despikeRgbSeries', () => {
    const targetBpm = 74;
    const baseFrames = generateSyntheticExtractions({
      bpm: targetBpm,
      durationSec: 13,
      fps: 30,
      seed: 2026,
    });

    // Inject abrupt smartphone AE steps (+14 at 4 s, -12 at 8.5 s)
    const steppedFrames = baseFrames.map((f) => {
      const elapsedSec = (f.timestampMs - 1000) / 1000;
      const aeOffset = elapsedSec >= 8.5 ? 2 : elapsedSec >= 4.0 ? 14 : 0;
      const shiftRgb = (rgb: RgbSample): RgbSample => ({
        ...rgb,
        r: rgb.r + aeOffset,
        g: rgb.g + aeOffset,
        b: rgb.b + aeOffset,
      });
      return {
        ...f,
        fusedRgb: f.fusedRgb ? shiftRgb(f.fusedRgb) : null,
        regions: {
          forehead: f.regions.forehead ? shiftRgb(f.regions.forehead) : undefined,
          leftCheek: f.regions.leftCheek ? shiftRgb(f.regions.leftCheek) : undefined,
          rightCheek: f.regions.rightCheek ? shiftRgb(f.regions.rightCheek) : undefined,
        },
      };
    });

    // Verify despikeRgbSeries removes the step jump cleanly
    const uniform = resampleRgbBuffer(
      steppedFrames.map((f) => ({
        timestampMs: f.timestampMs,
        r: f.fusedRgb!.r,
        g: f.fusedRgb!.g,
        b: f.fusedRgb!.b,
      })),
      30,
    );
    const despiked = despikeRgbSeries(uniform);
    const maxStep = Math.max(
      ...Array.from({ length: despiked.g.length - 1 }, (_, i) =>
        Math.abs(despiked.g[i + 1] - despiked.g[i]),
      ),
    );
    expect(maxStep).toBeLessThan(3.0);

    // End-to-end processor should still lock accurately onto 74 BPM
    const processor = new RppgProcessor();
    let latest = createIdleEstimate();
    for (const frame of steppedFrames) {
      latest = processor.pushSample(frame);
    }
    expect(latest.status).toBe('measuring');
    expect(latest.bpm).not.toBeNull();
    expect(Math.abs((latest.bpm ?? 0) - targetBpm)).toBeLessThanOrEqual(2.5);
  });

  it('down-weights noisy/occluded cheeks via Multi-ROI SNR fusion to preserve clean forehead BPM', () => {
    const targetBpm = 66;
    const rng = createRng(808);
    const cleanFrames = generateSyntheticExtractions({
      bpm: targetBpm,
      durationSec: 13,
      fps: 30,
      seed: 809,
    });

    // Corrupt leftCheek and rightCheek with strong broadband + 124 BPM artifact (beard/shadow/glasses)
    // while keeping forehead clean at 66 BPM.
    const occludedFrames = cleanFrames.map((f) => {
      const tSec = (f.timestampMs - 1000) / 1000;
      const noiseCheek = (orig: RgbSample): RgbSample => {
        const artifact = 12 * Math.sin(2 * Math.PI * (124 / 60) * tSec);
        return {
          r: Math.max(5, orig.r + artifact + 6 * gaussianNoise(rng)),
          g: Math.max(5, orig.g - artifact * 0.8 + 6 * gaussianNoise(rng)),
          b: Math.max(5, orig.b + artifact * 0.5 + 6 * gaussianNoise(rng)),
          pixelCount: orig.pixelCount,
        };
      };
      const forehead = f.regions.forehead!;
      const leftCheek = noiseCheek(f.regions.leftCheek!);
      const rightCheek = noiseCheek(f.regions.rightCheek!);
      return {
        ...f,
        regions: { forehead, leftCheek, rightCheek },
        fusedRgb: {
          r: 0.4 * forehead.r + 0.3 * leftCheek.r + 0.3 * rightCheek.r,
          g: 0.4 * forehead.g + 0.3 * leftCheek.g + 0.3 * rightCheek.g,
          b: 0.4 * forehead.b + 0.3 * leftCheek.b + 0.3 * rightCheek.b,
          pixelCount: forehead.pixelCount + leftCheek.pixelCount + rightCheek.pixelCount,
        },
      };
    });

    const processor = new RppgProcessor();
    let latest = createIdleEstimate();
    for (const frame of occludedFrames) {
      latest = processor.pushSample(frame);
    }

    expect(latest.status).toBe('measuring');
    expect(latest.bpm).not.toBeNull();
    expect(Math.abs((latest.bpm ?? 0) - targetBpm)).toBeLessThanOrEqual(2.5);
  });
});
