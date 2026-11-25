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
import { extractChromPulse, extractOmitPulse, extractPosPulse } from './algorithms';
import { BpmTemporalTracker, estimateSpectralBpm } from './bpmEstimator';
import {
  DEFAULT_FFT_SIZE,
  DEFAULT_MAX_FREQ_HZ,
  DEFAULT_MIN_FREQ_HZ,
  DEFAULT_SAMPLE_RATE_HZ,
  TimedRgbSample,
  UniformRgbSeries,
  despikeRgbSeries,
  linearDetrend,
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

type RegionKey = 'forehead' | 'leftCheek' | 'rightCheek';
const REGION_KEYS: readonly RegionKey[] = ['forehead', 'leftCheek', 'rightCheek'];

interface RoiPulseCandidate {
  key: RegionKey;
  series: UniformRgbSeries;
  pulse: Float64Array;
  bpm: number | null;
  snrDb: number;
  /** Physiological plausibility score (penalizes non-physiological >3.5% AC/DC motion/glare) */
  plausibility: number;
  effectiveScore: number;
  weight: number;
}

/**
 * Standardizes a pulse trace to zero-mean, unit-std so multi-ROI waveforms
 * can be combined purely by SNR weight without scale bias.
 */
function zNormalizePulse(pulse: Float64Array): Float64Array {
  const n = pulse.length;
  const out = new Float64Array(n);
  if (n === 0) return out;
  const m = mean(pulse);
  const s = std(pulse, m);
  if (s <= 1e-9) return out;
  const inv = 1 / s;
  for (let i = 0; i < n; i++) {
    out[i] = (pulse[i] - m) * inv;
  }
  return out;
}

export class RppgProcessor {
  private readonly options: RppgProcessorOptions;
  private readonly tracker: BpmTemporalTracker;

  private samples: TimedRgbSample[] = [];
  private regionSamples: Record<RegionKey, TimedRgbSample[]> = {
    forehead: [],
    leftCheek: [],
    rightCheek: [],
  };
  private frameTimestamps: number[] = [];
  private lastSampleTimestampMs: number | null = null;
  private lastAnalysisTimestampMs: number | null = null;
  private noFaceSinceMs: number | null = null;
  private wasWarmedUp = false;
  private pendingInitialBpm: number | null = null;

  // Smoothed acquisition telemetry
  private motionEma: number | null = null;
  private brightnessEma: number | null = null;
  private saturationEma: number | null = null;

  // Cached spectral analysis outputs between throttled FFT cycles
  private lastRawPosBpm: number | null = null;
  private lastRawChromBpm: number | null = null;
  private lastConsensusRefBpm: number | null = null;
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
    this.regionSamples = { forehead: [], leftCheek: [], rightCheek: [] };
    this.frameTimestamps = [];
    this.lastSampleTimestampMs = null;
    this.lastAnalysisTimestampMs = null;
    this.noFaceSinceMs = null;
    this.wasWarmedUp = false;
    this.pendingInitialBpm = null;
    this.motionEma = null;
    this.brightnessEma = null;
    this.saturationEma = null;
    this.lastRawPosBpm = null;
    this.lastRawChromBpm = null;
    this.lastConsensusRefBpm = null;
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
    this.regionSamples = { forehead: [], leftCheek: [], rightCheek: [] };
    this.lastAnalysisTimestampMs = null;
    this.wasWarmedUp = false;
    this.pendingInitialBpm = null;
    this.lastRawPosBpm = null;
    this.lastRawChromBpm = null;
    this.lastConsensusRefBpm = null;
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
    for (const key of REGION_KEYS) {
      const regionRgb = sample.regions[key];
      if (isValidRgb(regionRgb)) {
        this.regionSamples[key].push({
          timestampMs,
          r: regionRgb.r,
          g: regionRgb.g,
          b: regionRgb.b,
        });
      }
    }
    this.lastSampleTimestampMs = timestampMs;

    // Trim global and per-ROI buffers to sliding window (~14 seconds).
    const minWindowMs = timestampMs - this.options.windowSec * 1000;
    while (this.samples.length > 2 && this.samples[0].timestampMs < minWindowMs) {
      this.samples.shift();
    }
    for (const key of REGION_KEYS) {
      const buf = this.regionSamples[key];
      while (buf.length > 2 && buf[0].timestampMs < minWindowMs) {
        buf.shift();
      }
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
      chromBpm: isWarmedUp ? (this.lastConsensusRefBpm ?? this.lastRawChromBpm) : null,
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

  /**
   * Runs independent POS extraction per anatomical ROI (forehead, left cheek, right cheek)
   * and fuses their pulses via dynamic SNR-squared weighting so shadowed, glared, or occluded
   * regions are automatically down-weighted. Also returns an SNR-weighted RGB series for
   * CHROM and OMIT consensus estimation.
   */
  private extractMultiRoiSignals(
    fallbackUniform: UniformRgbSeries,
    priorBpm: number | null,
    isWarmedUp: boolean,
  ): { posPulse: Float64Array; weightedRgb: UniformRgbSeries } {
    if (!isWarmedUp) {
      const posPulse = extractPosPulse(fallbackUniform, {
        windowSec: 1.6,
        bandpass: true,
        minFreqHz: this.options.minFreqHz,
        maxFreqHz: this.options.maxFreqHz,
      });
      return { posPulse, weightedRgb: fallbackUniform };
    }

    const candidates: RoiPulseCandidate[] = [];
    const targetLen = fallbackUniform.r.length;
    const minCoverageSec = fallbackUniform.durationSec * 0.78;

    for (const key of REGION_KEYS) {
      const rawBuf = this.regionSamples[key];
      if (rawBuf.length < 30) continue;
      const spanSec = (rawBuf[rawBuf.length - 1].timestampMs - rawBuf[0].timestampMs) / 1000;
      if (spanSec < minCoverageSec) continue;

      const resampled = despikeRgbSeries(resampleRgbBuffer(rawBuf, this.options.sampleRateHz));
      if (resampled.r.length < Math.floor(targetLen * 0.78)) continue;

      const pulse = extractPosPulse(resampled, {
        windowSec: 1.6,
        bandpass: true,
        minFreqHz: this.options.minFreqHz,
        maxFreqHz: this.options.maxFreqHz,
      });

      const specEst = estimateSpectralBpm(pulse, resampled.sampleRateHz, {
        minFreqHz: this.options.minFreqHz,
        maxFreqHz: this.options.maxFreqHz,
        fftSize: this.options.fftSize,
        priorBpm,
      });
      const snrDb = computeSpectralSnrDb(
        specEst.spectrum,
        specEst.frequencyHz,
        this.options.minFreqHz,
        this.options.maxFreqHz,
      );

      // Estimate normalized green-channel AC/DC ratio; physiological facial BVP is ~0.2%–2.5%.
      // Much larger fluctuations indicate motion, glare, or occlusion artifacts rather than pulse.
      const gMean = Math.max(1, mean(resampled.g));
      const gAcDc = std(linearDetrend(resampled.g)) / gMean;
      const plausibility =
        gAcDc <= 0.035 ? 1.0 : Math.max(0.05, Math.exp(-((gAcDc - 0.035) / 0.025)));

      // Effective ROI quality score combines spectral SNR (dB) with physiological amplitude sanity
      const effectiveScore = snrDb - (1 - plausibility) * 14.0;

      candidates.push({
        key,
        series: resampled,
        pulse,
        bpm: specEst.bpm,
        snrDb,
        plausibility,
        effectiveScore,
        weight: 0,
      });
    }

    if (candidates.length === 0) {
      const posPulse = extractPosPulse(fallbackUniform, {
        windowSec: 1.6,
        bandpass: true,
        minFreqHz: this.options.minFreqHz,
        maxFreqHz: this.options.maxFreqHz,
      });
      return { posPulse, weightedRgb: fallbackUniform };
    }

    // Sort strongest effective quality score first so we can phase-align and exclude corrupted ROIs
    candidates.sort((a, b) => b.effectiveScore - a.effectiveScore);
    const best = candidates[0];

    let weightSum = 0;
    for (const c of candidates) {
      const scoreGap = best.effectiveScore - c.effectiveScore;
      const bpmDisagrees =
        best.bpm !== null && c.bpm !== null && Math.abs(best.bpm - c.bpm) > 10.0;

      // Hard-exclude corrupted ROIs (shadow, glare, beard, cheek motion) so large raw
      // artifacts cannot leak into the fused RGB trace used by CHROM/OMIT.
      if (
        (best.effectiveScore >= 1.0 && scoreGap > 4.5) ||
        (best.effectiveScore >= 1.0 && bpmDisagrees && scoreGap > 2.0) ||
        (best.plausibility >= 0.8 && c.plausibility < 0.35)
      ) {
        c.weight = 0;
      } else {
        c.weight = Math.exp(-scoreGap * 0.45) * c.plausibility * REGION_WEIGHTS[c.key];
      }
      weightSum += c.weight;
    }

    if (weightSum <= 1e-9) {
      best.weight = 1;
      weightSum = 1;
    }

    const activeCandidates = candidates.filter((c) => c.weight > 0);

    // Align all active ROI series on their overlapping tail length
    let commonLen = activeCandidates[0].pulse.length;
    for (let i = 1; i < activeCandidates.length; i++) {
      if (activeCandidates[i].pulse.length < commonLen) {
        commonLen = activeCandidates[i].pulse.length;
      }
    }

    const refTail = zNormalizePulse(
      activeCandidates[0].pulse.subarray(activeCandidates[0].pulse.length - commonLen),
    );
    const fusedPulse = new Float64Array(commonLen);
    const fusedR = new Float64Array(commonLen);
    const fusedG = new Float64Array(commonLen);
    const fusedB = new Float64Array(commonLen);

    for (const c of activeCandidates) {
      const w = c.weight / weightSum;
      const pTail = zNormalizePulse(c.pulse.subarray(c.pulse.length - commonLen));

      // Ensure polarity alignment with the highest-SNR ROI before summing
      let dot = 0;
      for (let i = 0; i < commonLen; i++) {
        dot += refTail[i] * pTail[i];
      }
      const sign = dot >= 0 ? 1 : -1;

      const rOffset = c.series.r.length - commonLen;
      for (let i = 0; i < commonLen; i++) {
        fusedPulse[i] += sign * w * pTail[i];
        fusedR[i] += w * c.series.r[rOffset + i];
        fusedG[i] += w * c.series.g[rOffset + i];
        fusedB[i] += w * c.series.b[rOffset + i];
      }
    }

    const weightedRgb: UniformRgbSeries = {
      sampleRateHz: this.options.sampleRateHz,
      r: fusedR,
      g: fusedG,
      b: fusedB,
      durationSec: (commonLen - 1) / this.options.sampleRateHz,
      startTimestampMs:
        fallbackUniform.startTimestampMs +
        Math.max(0, (fallbackUniform.r.length - commonLen) / this.options.sampleRateHz) * 1000,
    };

    return {
      posPulse: linearDetrend(fusedPulse),
      weightedRgb,
    };
  }

  private runSpectralAnalysis(
    timestampMs: number,
    isWarmedUp: boolean,
    telemetry: { motionScore: number; brightness: number; saturationFraction: number },
    fps: number,
    bufferDurationSec: number,
  ): void {
    const rawUniform = resampleRgbBuffer(this.samples, this.options.sampleRateHz);
    if (rawUniform.r.length < 30) {
      return;
    }

    const uniform = despikeRgbSeries(rawUniform);
    const priorBpm = this.tracker.getCurrentBpm();
    const { posPulse, weightedRgb } = this.extractMultiRoiSignals(uniform, priorBpm, isWarmedUp);

    // Expose live filtered pulse waveform (~120 points, -1..1) once >= 2s buffered.
    this.lastWaveform = buildNormalizedWaveform(posPulse, this.options.waveformLength);

    if (!isWarmedUp) {
      return;
    }

    const chromPulse = extractChromPulse(weightedRgb, {
      minFreqHz: this.options.minFreqHz,
      maxFreqHz: this.options.maxFreqHz,
    });
    const omitPulse = extractOmitPulse(weightedRgb, {
      windowSec: 1.6,
      bandpass: true,
      minFreqHz: this.options.minFreqHz,
      maxFreqHz: this.options.maxFreqHz,
    });

    const posEst = estimateSpectralBpm(posPulse, weightedRgb.sampleRateHz, {
      minFreqHz: this.options.minFreqHz,
      maxFreqHz: this.options.maxFreqHz,
      fftSize: this.options.fftSize,
      priorBpm,
    });
    const chromEst = estimateSpectralBpm(chromPulse, weightedRgb.sampleRateHz, {
      minFreqHz: this.options.minFreqHz,
      maxFreqHz: this.options.maxFreqHz,
      fftSize: this.options.fftSize,
      priorBpm,
    });
    const omitEst = estimateSpectralBpm(omitPulse, weightedRgb.sampleRateHz, {
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
    const omitSnrDb = computeSpectralSnrDb(
      omitEst.spectrum,
      omitEst.frequencyHz,
      this.options.minFreqHz,
      this.options.maxFreqHz,
    );

    this.lastRawPosBpm = posEst.bpm;
    this.lastRawChromBpm = chromEst.bpm;

    // 3-Algorithm Consensus Fusion (Multi-ROI POS + OMIT + CHROM):
    // Combine agreeing estimators weighted by SNR; if only 2 of 3 agree (e.g., CHROM
    // degrades under tinted indoor mobile LED lighting while POS & OMIT lock tightly),
    // trust the agreeing pair cleanly.
    const algos: Array<{ name: 'pos' | 'omit' | 'chrom'; bpm: number; snrDb: number; baseWeight: number }> = [];
    if (posEst.bpm !== null) {
      algos.push({ name: 'pos', bpm: posEst.bpm, snrDb: posSnrDb, baseWeight: 1.25 });
    }
    if (omitEst.bpm !== null) {
      algos.push({ name: 'omit', bpm: omitEst.bpm, snrDb: omitSnrDb, baseWeight: 1.15 });
    }
    if (chromEst.bpm !== null) {
      algos.push({ name: 'chrom', bpm: chromEst.bpm, snrDb: chromSnrDb, baseWeight: 0.9 });
    }

    let candidateBpm: number | null = posEst.bpm;
    let primarySnrDb = posSnrDb;
    let consensusRefBpm: number | null = chromEst.bpm;

    if (algos.length >= 2) {
      // Find the largest agreeing cluster (within 4.5 BPM) with highest total SNR score
      let bestCluster: typeof algos = [];
      let bestClusterScore = -Infinity;

      for (const anchor of algos) {
        const cluster = algos.filter((a) => Math.abs(a.bpm - anchor.bpm) <= 4.5);
        const clusterScore =
          cluster.length * 10 +
          cluster.reduce((sum, a) => sum + Math.max(-5, a.snrDb) * a.baseWeight, 0);
        if (clusterScore > bestClusterScore) {
          bestClusterScore = clusterScore;
          bestCluster = cluster;
        }
      }

      if (bestCluster.length >= 2) {
        let wSum = 0;
        let bpmSum = 0;
        let snrSum = 0;
        let maxClusterSnr = -15;
        for (const a of bestCluster) {
          const w = a.baseWeight * Math.max(0.25, a.snrDb + 4.0);
          wSum += w;
          bpmSum += w * a.bpm;
          snrSum += w * a.snrDb;
          if (a.snrDb > maxClusterSnr) maxClusterSnr = a.snrDb;
        }
        candidateBpm = bpmSum / wSum;
        primarySnrDb = Math.max(maxClusterSnr, snrSum / wSum);

        // Pick the closest secondary algorithm to POS (or cluster mean) for agreement reporting
        const refCandidate = bestCluster.find((a) => a.name !== 'pos') ?? bestCluster[0];
        consensusRefBpm = refCandidate.bpm;
      } else {
        // No pair agrees within 4.5 BPM: pick the highest-SNR algorithm (preferring POS on ties)
        const best = [...algos].sort(
          (a, b) => (b.snrDb + (b.name === 'pos' ? 1.0 : 0)) - (a.snrDb + (a.name === 'pos' ? 1.0 : 0)),
        )[0];
        candidateBpm = best.bpm;
        primarySnrDb = best.snrDb;
        consensusRefBpm = chromEst.bpm ?? omitEst.bpm;
      }
    } else if (algos.length === 1) {
      candidateBpm = algos[0].bpm;
      primarySnrDb = algos[0].snrDb;
      consensusRefBpm = null;
    }

    this.lastConsensusRefBpm = consensusRefBpm;
    this.lastSnrDb = primarySnrDb;

    // Pre-assess confidence for adaptive tracker weighting.
    const preliminaryQuality = assessSignalQuality({
      faceDetected: true,
      bufferDurationSec,
      warmUpSec: this.options.warmUpSec,
      windowSec: this.options.windowSec,
      snrDb: primarySnrDb,
      posBpm: this.lastRawPosBpm,
      chromBpm: consensusRefBpm,
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
      // Smart initial lock: avoid seeding the very first reading on a noisy sub-0 dB transient;
      // either require SNR >= 0 dB or two consecutive agreeing candidates within 6 BPM.
      if (this.tracker.getCurrentBpm() === null && primarySnrDb < 0) {
        if (
          this.pendingInitialBpm !== null &&
          Math.abs(candidateBpm - this.pendingInitialBpm) <= 6.0
        ) {
          const seeded = 0.5 * (candidateBpm + this.pendingInitialBpm);
          this.pendingInitialBpm = null;
          this.tracker.update(seeded, timestampMs, primarySnrDb, preliminaryQuality.confidence);
        } else {
          this.pendingInitialBpm = candidateBpm;
        }
        return;
      }

      this.pendingInitialBpm = null;
      this.tracker.update(
        candidateBpm,
        timestampMs,
        primarySnrDb,
        preliminaryQuality.confidence,
      );
    }
  }
}
