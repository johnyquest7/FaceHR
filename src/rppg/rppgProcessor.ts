/**
 * Stateful sliding-window rPPG processor coordinating multi-ROI fusion,
 * uniform resampling, POS & CHROM extraction, spectral BPM estimation,
 * temporal tracking, and signal quality assessment.
 *
 * Pure TypeScript — zero DOM dependencies.
 */

import type {
  FrameRoiExtraction,
  HeartRateEstimate,
  RgbSample,
} from '../types/rppg';
import { extractChromPulse, extractPosPulse } from './algorithms';
import { BpmTemporalTracker, estimateSpectralBpm } from './bpmEstimator';
import {
  DEFAULT_FFT_SIZE,
  DEFAULT_MAX_FREQ_HZ,
  DEFAULT_MIN_FREQ_HZ,
  DEFAULT_SAMPLE_RATE_HZ,
  TimedRgbSample,
  mean,
  resampleRgbBuffer,
  std,
} from './dsp';
import { assessSignalQuality, computeSpectralSnrDb } from './signalQuality';

export interface RppgProcessorOptions {
  /** Sliding analysis window duration in seconds (default 14 s) */
  windowSec: number;
  /** Minimum buffer duration before emitting first BPM reading (default 6 s) */
  warmUpSec: number;
  /** Target uniform resampling rate in Hz (default 30 Hz) */
  sampleRateHz: number;
  /** Minimum physiological heart rate frequency in Hz (default 0.70 Hz = 42 BPM) */
  minFreqHz: number;
  /** Maximum physiological heart rate frequency in Hz (default 3.50 Hz = 210 BPM) */
  maxFreqHz: number;
  /** Zero-padded FFT size (default 4096) */
  fftSize: number;
  /** Number of normalized pulse waveform points exposed in estimate.waveform */
  waveformLength: number;
  /** Minimum interval between full POS/CHROM/FFT recomputations (ms) */
  minAnalysisIntervalMs: number;
  /** Face-loss / timestamp gap threshold before clearing stale buffer (ms) */
  maxGapResetMs: number;
}

const DEFAULT_OPTIONS: RppgProcessorOptions = {
  windowSec: 14.0,
  warmUpSec: 6.0,
  sampleRateHz: DEFAULT_SAMPLE_RATE_HZ,
  minFreqHz: DEFAULT_MIN_FREQ_HZ,
  maxFreqHz: DEFAULT_MAX_FREQ_HZ,
  fftSize: DEFAULT_FFT_SIZE,
  waveformLength: 120,
  minAnalysisIntervalMs: 150,
  maxGapResetMs: 1500,
};

const REGION_WEIGHTS = {
  forehead: 0.40,
  leftCheek: 0.30,
  rightCheek: 0.30,
} as const;

/**
 * Creates a clean idle HeartRateEstimate suitable for initial UI/hook state.
 */
export function createIdleEstimate(updatedAtMs = 0): HeartRateEstimate {
  return {
    bpm: null,
    rawPosBpm: null,
    rawChromBpm: null,
    confidence: 0,
    quality: 'poor',
    status: 'idle',
    guidance: 'Position your face in the frame and press Start',
    diagnostics: {
      snrDb: 0,
      algorithmAgreementBpm: null,
      motionStability: 100,
      lightingScore: 100,
      fps: 0,
      bufferProgress: 0,
    },
    waveform: [],
    updatedAtMs,
  };
}

function isValidRgb(sample: RgbSample | null | undefined): sample is RgbSample {
  return (
    sample !== null &&
    sample !== undefined &&
    sample.pixelCount > 0 &&
    Number.isFinite(sample.r) &&
    Number.isFinite(sample.g) &&
    Number.isFinite(sample.b) &&
    sample.r > 0 &&
    sample.g > 0 &&
    sample.b > 0
  );
}

/**
 * Multi-ROI fusion helper: combines forehead, left cheek, and right cheek
 * with anatomical weights scaled by pixel reliability, falling back to fusedRgb.
 */
export function fuseRoiRgb(extraction: FrameRoiExtraction): RgbSample | null {
  const { regions, fusedRgb } = extraction;
  const entries: Array<{ rgb: RgbSample; weight: number }> = [];

  if (isValidRgb(regions.forehead)) {
    const reliability = Math.min(1, Math.max(0.25, regions.forehead.pixelCount / 100));
    entries.push({ rgb: regions.forehead, weight: REGION_WEIGHTS.forehead * reliability });
  }
  if (isValidRgb(regions.leftCheek)) {
    const reliability = Math.min(1, Math.max(0.25, regions.leftCheek.pixelCount / 100));
    entries.push({ rgb: regions.leftCheek, weight: REGION_WEIGHTS.leftCheek * reliability });
  }
  if (isValidRgb(regions.rightCheek)) {
    const reliability = Math.min(1, Math.max(0.25, regions.rightCheek.pixelCount / 100));
    entries.push({ rgb: regions.rightCheek, weight: REGION_WEIGHTS.rightCheek * reliability });
  }

  if (entries.length > 0) {
    let totalWeight = 0;
    let r = 0;
    let g = 0;
    let b = 0;
    let pixelCount = 0;

    for (const item of entries) {
      totalWeight += item.weight;
      r += item.rgb.r * item.weight;
      g += item.rgb.g * item.weight;
      b += item.rgb.b * item.weight;
      pixelCount += item.rgb.pixelCount;
    }

    if (totalWeight > 0) {
      return {
        r: r / totalWeight,
        g: g / totalWeight,
        b: b / totalWeight,
        pixelCount,
      };
    }
  }

  if (isValidRgb(fusedRgb)) {
    return fusedRgb;
  }

  return null;
}

/**
 * Extracts and normalizes the latest `targetLength` pulse samples into [-1, 1].
 */
function buildNormalizedWaveform(pulse: Float64Array, targetLength: number): number[] {
  const n = pulse.length;
  if (n === 0 || targetLength <= 0) return [];

  const count = Math.min(n, targetLength);
  const start = n - count;
  const slice = pulse.subarray(start, n);

  const sliceMean = mean(slice);
  const sliceStd = std(slice, sliceMean);
  let maxAbs = 0;
  for (let i = 0; i < count; i++) {
    const absVal = Math.abs(slice[i] - sliceMean);
    if (absVal > maxAbs) maxAbs = absVal;
  }

  const scale = Math.max(maxAbs, 2.2 * sliceStd, 1e-6);
  const out = new Array<number>(count);
  for (let i = 0; i < count; i++) {
    const normalized = (slice[i] - sliceMean) / scale;
    const clamped = Math.max(-1, Math.min(1, normalized));
    out[i] = Math.round(clamped * 10000) / 10000;
  }
  return out;
}

export class RppgProcessor {
  private readonly options: RppgProcessorOptions;
  private readonly tracker: BpmTemporalTracker;

  private samples: TimedRgbSample[] = [];
  private frameTimestamps: number[] = [];
  private lastSampleTimestampMs: number | null = null;
  private lastAnalysisTimestampMs: number | null = null;
  private noFaceSinceMs: number | null = null;
  private wasWarmedUp = false;

  // Smoothed acquisition telemetry
  private motionEma: number | null = null;
  private brightnessEma: number | null = null;
  private saturationEma: number | null = null;

  // Cached spectral analysis outputs between throttled FFT cycles
  private lastRawPosBpm: number | null = null;
  private lastRawChromBpm: number | null = null;
  private lastSnrDb = -15;
  private lastWaveform: number[] = [];

  private currentEstimate: HeartRateEstimate;

  constructor(options: Partial<RppgProcessorOptions> = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.tracker = new BpmTemporalTracker();
    this.currentEstimate = createIdleEstimate();
  }

  public reset(): void {
    this.samples = [];
    this.frameTimestamps = [];
    this.lastSampleTimestampMs = null;
    this.lastAnalysisTimestampMs = null;
    this.noFaceSinceMs = null;
    this.wasWarmedUp = false;
    this.motionEma = null;
    this.brightnessEma = null;
    this.saturationEma = null;
    this.lastRawPosBpm = null;
    this.lastRawChromBpm = null;
    this.lastSnrDb = -15;
    this.lastWaveform = [];
    this.tracker.reset();
    this.currentEstimate = createIdleEstimate();
  }

  public getEstimate(): HeartRateEstimate {
    return this.currentEstimate;
  }

  private clearSignalBuffer(): void {
    this.samples = [];
    this.lastAnalysisTimestampMs = null;
    this.wasWarmedUp = false;
    this.lastRawPosBpm = null;
    this.lastRawChromBpm = null;
    this.lastSnrDb = -15;
    this.lastWaveform = [];
    this.tracker.reset();
  }

  private updateFps(timestampMs: number): number {
    this.frameTimestamps.push(timestampMs);
    const cutoff = timestampMs - 2500;
    while (this.frameTimestamps.length > 2 && this.frameTimestamps[0] < cutoff) {
      this.frameTimestamps.shift();
    }
    if (this.frameTimestamps.length < 2) {
      return 0;
    }
    const spanMs =
      this.frameTimestamps[this.frameTimestamps.length - 1] - this.frameTimestamps[0];
    if (spanMs <= 0) return 0;
    return ((this.frameTimestamps.length - 1) * 1000) / spanMs;
  }

  private updateTelemetryEma(sample: FrameRoiExtraction): {
    motionScore: number;
    brightness: number;
    saturationFraction: number;
  } {
    const alpha = 0.32;
    if (this.motionEma === null || this.brightnessEma === null || this.saturationEma === null) {
      this.motionEma = sample.motionScore;
      this.brightnessEma = sample.brightness;
      this.saturationEma = sample.saturationFraction;
    } else {
      this.motionEma = this.motionEma + alpha * (sample.motionScore - this.motionEma);
      this.brightnessEma = this.brightnessEma + alpha * (sample.brightness - this.brightnessEma);
      this.saturationEma =
        this.saturationEma + alpha * (sample.saturationFraction - this.saturationEma);
    }

    return {
      motionScore: this.motionEma,
      brightness: this.brightnessEma,
      saturationFraction: this.saturationEma,
    };
  }

  public pushSample(sample: FrameRoiExtraction): HeartRateEstimate {
    const { timestampMs } = sample;
    if (!Number.isFinite(timestampMs)) {
      return this.currentEstimate;
    }

    // Ignore duplicate or out-of-order timestamps.
    if (
      this.frameTimestamps.length > 0 &&
      timestampMs <= this.frameTimestamps[this.frameTimestamps.length - 1]
    ) {
      return this.currentEstimate;
    }

    const fps = this.updateFps(timestampMs);
    const rgb = sample.faceDetected ? fuseRoiRgb(sample) : null;

    if (!sample.faceDetected || rgb === null) {
      if (this.noFaceSinceMs === null) {
        this.noFaceSinceMs = timestampMs;
      } else if (timestampMs - this.noFaceSinceMs >= this.options.maxGapResetMs) {
        this.clearSignalBuffer();
      }

      const bufferDurationSec = this.getBufferDurationSec();
      const quality = assessSignalQuality({
        faceDetected: false,
        bufferDurationSec,
        warmUpSec: this.options.warmUpSec,
        windowSec: this.options.windowSec,
        snrDb: -15,
        posBpm: null,
        chromBpm: null,
        motionScore: sample.motionScore,
        brightness: sample.brightness,
        saturationFraction: sample.saturationFraction,
        fps,
      });

      this.currentEstimate = {
        bpm: null,
        rawPosBpm: null,
        rawChromBpm: null,
        confidence: quality.confidence,
        quality: quality.quality,
        status: quality.status,
        guidance: quality.guidance,
        diagnostics: quality.diagnostics,
        waveform: [],
        updatedAtMs: timestampMs,
      };
      return this.currentEstimate;
    }

    // Face detected with valid RGB sample.
    this.noFaceSinceMs = null;

    // Reset stale signal buffer if there was a large time gap.
    if (
      this.lastSampleTimestampMs !== null &&
      timestampMs - this.lastSampleTimestampMs > this.options.maxGapResetMs
    ) {
      this.clearSignalBuffer();
    }

    this.samples.push({
      timestampMs,
      r: rgb.r,
      g: rgb.g,
      b: rgb.b,
    });
    this.lastSampleTimestampMs = timestampMs;

    // Trim buffer to sliding window (~14 seconds).
    const minWindowMs = timestampMs - this.options.windowSec * 1000;
    while (this.samples.length > 2 && this.samples[0].timestampMs < minWindowMs) {
      this.samples.shift();
    }

    const telemetry = this.updateTelemetryEma(sample);
    const bufferDurationSec = this.getBufferDurationSec();
    const isWarmedUp = bufferDurationSec >= this.options.warmUpSec;
    const justWarmedUp = isWarmedUp && !this.wasWarmedUp;

    const shouldRunAnalysis =
      justWarmedUp ||
      this.lastAnalysisTimestampMs === null ||
      timestampMs - this.lastAnalysisTimestampMs >= this.options.minAnalysisIntervalMs;

    if (shouldRunAnalysis && bufferDurationSec >= 2.0 && this.samples.length >= 20) {
      this.runSpectralAnalysis(timestampMs, isWarmedUp, telemetry, fps, bufferDurationSec);
      this.lastAnalysisTimestampMs = timestampMs;
    }

    this.wasWarmedUp = isWarmedUp;

    const quality = assessSignalQuality({
      faceDetected: true,
      bufferDurationSec,
      warmUpSec: this.options.warmUpSec,
      windowSec: this.options.windowSec,
      snrDb: isWarmedUp ? this.lastSnrDb : -15,
      posBpm: isWarmedUp ? this.lastRawPosBpm : null,
      chromBpm: isWarmedUp ? this.lastRawChromBpm : null,
      motionScore: telemetry.motionScore,
      brightness: telemetry.brightness,
      saturationFraction: telemetry.saturationFraction,
      fps,
    });

    const trackedBpm = this.tracker.getCurrentBpm();
    let exposedBpm: number | null = null;
    if (isWarmedUp && trackedBpm !== null) {
      // Suppress unreliable BPM when signal is critically poor and never stabilized.
      exposedBpm =
        quality.status === 'low-signal' && quality.confidence < 25
          ? null
          : Math.round(trackedBpm * 10) / 10;
    }

    this.currentEstimate = {
      bpm: exposedBpm,
      rawPosBpm:
        isWarmedUp && this.lastRawPosBpm !== null
          ? Math.round(this.lastRawPosBpm * 10) / 10
          : null,
      rawChromBpm:
        isWarmedUp && this.lastRawChromBpm !== null
          ? Math.round(this.lastRawChromBpm * 10) / 10
          : null,
      confidence: quality.confidence,
      quality: quality.quality,
      status: quality.status,
      guidance: quality.guidance,
      diagnostics: quality.diagnostics,
      waveform: this.lastWaveform,
      updatedAtMs: timestampMs,
    };

    return this.currentEstimate;
  }

  private getBufferDurationSec(): number {
    if (this.samples.length < 2) return 0;
    const spanMs =
      this.samples[this.samples.length - 1].timestampMs - this.samples[0].timestampMs;
    return Math.max(0, spanMs / 1000);
  }

  private runSpectralAnalysis(
    timestampMs: number,
    isWarmedUp: boolean,
    telemetry: { motionScore: number; brightness: number; saturationFraction: number },
    fps: number,
    bufferDurationSec: number,
  ): void {
    const uniform = resampleRgbBuffer(this.samples, this.options.sampleRateHz);
    if (uniform.r.length < 30) {
      return;
    }

    const posPulse = extractPosPulse(uniform, {
      windowSec: 1.6,
      bandpass: true,
      minFreqHz: this.options.minFreqHz,
      maxFreqHz: this.options.maxFreqHz,
    });

    // Expose live filtered pulse waveform (~120 points, -1..1) once >= 2s buffered.
    this.lastWaveform = buildNormalizedWaveform(posPulse, this.options.waveformLength);

    if (!isWarmedUp) {
      return;
    }

    const chromPulse = extractChromPulse(uniform, {
      minFreqHz: this.options.minFreqHz,
      maxFreqHz: this.options.maxFreqHz,
    });

    const priorBpm = this.tracker.getCurrentBpm();
    const posEst = estimateSpectralBpm(posPulse, uniform.sampleRateHz, {
      minFreqHz: this.options.minFreqHz,
      maxFreqHz: this.options.maxFreqHz,
      fftSize: this.options.fftSize,
      priorBpm,
    });

    const chromEst = estimateSpectralBpm(chromPulse, uniform.sampleRateHz, {
      minFreqHz: this.options.minFreqHz,
      maxFreqHz: this.options.maxFreqHz,
      fftSize: this.options.fftSize,
      priorBpm,
    });

    const posSnrDb = computeSpectralSnrDb(
      posEst.spectrum,
      posEst.frequencyHz,
      this.options.minFreqHz,
      this.options.maxFreqHz,
    );
    const chromSnrDb = computeSpectralSnrDb(
      chromEst.spectrum,
      chromEst.frequencyHz,
      this.options.minFreqHz,
      this.options.maxFreqHz,
    );

    this.lastRawPosBpm = posEst.bpm;
    this.lastRawChromBpm = chromEst.bpm;

    // Prefer POS as primary (best robustness to lighting drift + motion);
    // blend slightly with CHROM when both agree tightly, or fall back to CHROM if POS is missing.
    let candidateBpm = posEst.bpm;
    let primarySnrDb = posSnrDb;

    if (posEst.bpm !== null && chromEst.bpm !== null) {
      const diff = Math.abs(posEst.bpm - chromEst.bpm);
      if (diff <= 4.0) {
        candidateBpm = 0.7 * posEst.bpm + 0.3 * chromEst.bpm;
        primarySnrDb = Math.max(posSnrDb, (posSnrDb + chromSnrDb) / 2);
      } else if (chromSnrDb > posSnrDb + 4.0 && posSnrDb < 1.5) {
        candidateBpm = chromEst.bpm;
        primarySnrDb = chromSnrDb;
      }
    } else if (posEst.bpm === null && chromEst.bpm !== null) {
      candidateBpm = chromEst.bpm;
      primarySnrDb = chromSnrDb;
    }

    this.lastSnrDb = primarySnrDb;

    // Pre-assess confidence for adaptive tracker weighting.
    const preliminaryQuality = assessSignalQuality({
      faceDetected: true,
      bufferDurationSec,
      warmUpSec: this.options.warmUpSec,
      windowSec: this.options.windowSec,
      snrDb: primarySnrDb,
      posBpm: posEst.bpm,
      chromBpm: chromEst.bpm,
      motionScore: telemetry.motionScore,
      brightness: telemetry.brightness,
      saturationFraction: telemetry.saturationFraction,
      fps,
    });

    // Only feed tracker when signal isn't severely corrupted by extreme motion/darkness.
    if (
      candidateBpm !== null &&
      preliminaryQuality.diagnostics.motionStability >= 25 &&
      preliminaryQuality.diagnostics.lightingScore >= 25
    ) {
      this.tracker.update(
        candidateBpm,
        timestampMs,
        primarySnrDb,
        preliminaryQuality.confidence,
      );
    }
  }
}
