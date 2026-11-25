/**
 * Spectral BPM estimation with parabolic peak refinement,
 * harmonic/subharmonic disambiguation, and physiological temporal tracking.
 */

import {
  DEFAULT_FFT_SIZE,
  DEFAULT_MAX_FREQ_HZ,
  DEFAULT_MIN_FREQ_HZ,
  DEFAULT_SAMPLE_RATE_HZ,
  SpectrumResult,
  applyHannWindow,
  computeSpectrum,
  linearDetrend,
  parabolicInterpolatePeak,
} from './dsp';

export interface SpectralEstimateOptions {
  minFreqHz?: number;
  maxFreqHz?: number;
  fftSize?: number;
  /** Optional prior tracked BPM to assist borderline harmonic disambiguation */
  priorBpm?: number | null;
}

export interface SpectralBpmEstimate {
  bpm: number | null;
  frequencyHz: number | null;
  peakMagnitude: number;
  medianMagnitude: number;
  spectrum: SpectrumResult;
  harmonicDisambiguated: boolean;
}

function computeMedian(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function findLocalPeakIndex(
  magnitudes: Float64Array,
  frequencies: Float64Array,
  targetFreqHz: number,
  toleranceHz: number,
  minIdx: number,
  maxIdx: number,
): { index: number; magnitude: number; isLocalMax: boolean } | null {
  let bestIdx = -1;
  let bestMag = -1;

  for (let i = minIdx; i <= maxIdx; i++) {
    if (Math.abs(frequencies[i] - targetFreqHz) <= toleranceHz) {
      if (magnitudes[i] > bestMag) {
        bestMag = magnitudes[i];
        bestIdx = i;
      }
    }
  }

  if (bestIdx < 0) return null;

  const left = bestIdx > 0 ? magnitudes[bestIdx - 1] : 0;
  const right = bestIdx < magnitudes.length - 1 ? magnitudes[bestIdx + 1] : 0;
  const isLocalMax = bestMag >= left && bestMag >= right;

  return { index: bestIdx, magnitude: bestMag, isLocalMax };
}

/**
 * Estimates heart rate in BPM from a bandlimited pulse waveform using
 * zero-padded FFT, parabolic peak interpolation, and harmonic/subharmonic
 * disambiguation (checking f/2, f/3, and 2f).
 */
export function estimateSpectralBpm(
  pulseSignal: Float64Array,
  sampleRateHz = DEFAULT_SAMPLE_RATE_HZ,
  options: SpectralEstimateOptions = {},
): SpectralBpmEstimate {
  const minFreqHz = options.minFreqHz ?? DEFAULT_MIN_FREQ_HZ;
  const maxFreqHz = options.maxFreqHz ?? DEFAULT_MAX_FREQ_HZ;
  const fftSize = options.fftSize ?? DEFAULT_FFT_SIZE;
  const priorBpm = options.priorBpm ?? null;

  const windowed = applyHannWindow(linearDetrend(pulseSignal));
  const spectrum = computeSpectrum(windowed, sampleRateHz, fftSize);
  const { frequencies, magnitudes, binResolutionHz } = spectrum;

  const minIdx = Math.max(1, Math.ceil(minFreqHz / binResolutionHz));
  const maxIdx = Math.min(magnitudes.length - 2, Math.floor(maxFreqHz / binResolutionHz));

  if (minIdx >= maxIdx) {
    return {
      bpm: null,
      frequencyHz: null,
      peakMagnitude: 0,
      medianMagnitude: 0,
      spectrum,
      harmonicDisambiguated: false,
    };
  }

  // Find global peak and noise floor (median magnitude) within physiological HR band.
  let peakIdx = minIdx;
  let peakMag = magnitudes[minIdx];
  const bandMags: number[] = [];

  for (let i = minIdx; i <= maxIdx; i++) {
    const m = magnitudes[i];
    bandMags.push(m);
    if (m > peakMag) {
      peakMag = m;
      peakIdx = i;
    }
  }

  const medianMag = computeMedian(bandMags);
  if (peakMag <= 1e-12) {
    return {
      bpm: null,
      frequencyHz: null,
      peakMagnitude: 0,
      medianMagnitude: medianMag,
      spectrum,
      harmonicDisambiguated: false,
    };
  }

  const primaryParabolic = parabolicInterpolatePeak(magnitudes, peakIdx);
  let chosenFreqHz = primaryParabolic.index * binResolutionHz;
  let chosenMag = primaryParabolic.value;
  let harmonicDisambiguated = false;

  const harmonicToleranceHz = Math.max(0.08, 3 * binResolutionHz);

  // 1. Subharmonic check (f/2): prevents strong 2nd harmonics from causing 2x BPM errors.
  const halfFreqHz = chosenFreqHz / 2;
  if (halfFreqHz >= minFreqHz) {
    const subPeak = findLocalPeakIndex(
      magnitudes,
      frequencies,
      halfFreqHz,
      harmonicToleranceHz,
      minIdx,
      maxIdx,
    );
    if (subPeak && subPeak.isLocalMax) {
      const priorSupportsHalf =
        priorBpm !== null &&
        Math.abs(halfFreqHz * 60 - priorBpm) <= 12 &&
        Math.abs(chosenFreqHz * 60 - priorBpm) > 25;

      const minRatio = priorSupportsHalf ? 0.28 : 0.42;
      const minFloorFactor = priorSupportsHalf ? 2.0 : 2.5;

      if (
        subPeak.magnitude >= minRatio * peakMag &&
        subPeak.magnitude >= minFloorFactor * Math.max(medianMag, 1e-12)
      ) {
        const refinedSub = parabolicInterpolatePeak(magnitudes, subPeak.index);
        chosenFreqHz = refinedSub.index * binResolutionHz;
        chosenMag = refinedSub.value;
        harmonicDisambiguated = true;
      }
    }
  }

  // 2. Third-harmonic check (f/3): handles rare cases where the 3rd harmonic dominates.
  if (!harmonicDisambiguated) {
    const thirdFreqHz = chosenFreqHz / 3;
    if (thirdFreqHz >= minFreqHz) {
      const subThird = findLocalPeakIndex(
        magnitudes,
        frequencies,
        thirdFreqHz,
        harmonicToleranceHz,
        minIdx,
        maxIdx,
      );
      const twoThird = findLocalPeakIndex(
        magnitudes,
        frequencies,
        2 * thirdFreqHz,
        harmonicToleranceHz,
        minIdx,
        maxIdx,
      );
      if (
        subThird &&
        subThird.isLocalMax &&
        twoThird &&
        subThird.magnitude >= 0.40 * peakMag &&
        twoThird.magnitude >= 0.35 * peakMag &&
        subThird.magnitude >= 2.5 * Math.max(medianMag, 1e-12)
      ) {
        const refinedThird = parabolicInterpolatePeak(magnitudes, subThird.index);
        chosenFreqHz = refinedThird.index * binResolutionHz;
        chosenMag = refinedThird.value;
        harmonicDisambiguated = true;
      }
    }
  }

  // 3. Double-frequency check (2f): only prefer 2f if a strong prior tracks 2f
  //    and the lower peak is likely low-frequency respiration/motion leakage.
  if (!harmonicDisambiguated && priorBpm !== null) {
    const doubleFreqHz = chosenFreqHz * 2;
    if (doubleFreqHz <= maxFreqHz) {
      const dblPeak = findLocalPeakIndex(
        magnitudes,
        frequencies,
        doubleFreqHz,
        harmonicToleranceHz,
        minIdx,
        maxIdx,
      );
      if (
        dblPeak &&
        dblPeak.isLocalMax &&
        dblPeak.magnitude >= 0.65 * peakMag &&
        Math.abs(doubleFreqHz * 60 - priorBpm) <= 10 &&
        Math.abs(chosenFreqHz * 60 - priorBpm) > 25
      ) {
        const refinedDbl = parabolicInterpolatePeak(magnitudes, dblPeak.index);
        chosenFreqHz = refinedDbl.index * binResolutionHz;
        chosenMag = refinedDbl.value;
        harmonicDisambiguated = true;
      }
    }
  }

  // Clamp refined frequency to physiological band
  chosenFreqHz = Math.min(maxFreqHz, Math.max(minFreqHz, chosenFreqHz));

  return {
    bpm: chosenFreqHz * 60,
    frequencyHz: chosenFreqHz,
    peakMagnitude: chosenMag,
    medianMagnitude: medianMag,
    spectrum,
    harmonicDisambiguated,
  };
}

export interface BpmTrackerOptions {
  /** Maximum instantaneous jump (BPM) accepted without sustained high SNR */
  maxInstantJumpBpm?: number;
  /** How long a large jump must persist at high SNR before being accepted (ms) */
  sustainedJumpDurationMs?: number;
  /** Minimum average SNR (dB) required to accept a sustained large jump */
  sustainedJumpMinSnrDb?: number;
  /** Minimum consistent observations for a sustained jump */
  sustainedJumpMinCount?: number;
  /** Number of recent accepted readings in the median buffer */
  medianWindowSize?: number;
}

interface PendingJumpState {
  bpmSum: number;
  snrSum: number;
  count: number;
  firstSeenMs: number;
  lastSeenMs: number;
}

/**
 * Temporal BPM tracker combining a sliding median filter with adaptive
 * time-constant smoothing and physiological jump rejection.
 */
export class BpmTemporalTracker {
  private readonly maxInstantJumpBpm: number;
  private readonly sustainedJumpDurationMs: number;
  private readonly sustainedJumpMinSnrDb: number;
  private readonly sustainedJumpMinCount: number;
  private readonly medianWindowSize: number;

  private smoothedBpm: number | null = null;
  private lastAcceptedAtMs: number | null = null;
  private recentBpms: number[] = [];
  private pendingJump: PendingJumpState | null = null;

  constructor(options: BpmTrackerOptions = {}) {
    this.maxInstantJumpBpm = options.maxInstantJumpBpm ?? 18;
    this.sustainedJumpDurationMs = options.sustainedJumpDurationMs ?? 1200;
    this.sustainedJumpMinSnrDb = options.sustainedJumpMinSnrDb ?? 5.0;
    this.sustainedJumpMinCount = options.sustainedJumpMinCount ?? 4;
    this.medianWindowSize = options.medianWindowSize ?? 7;
  }

  public getCurrentBpm(): number | null {
    return this.smoothedBpm;
  }

  public reset(): void {
    this.smoothedBpm = null;
    this.lastAcceptedAtMs = null;
    this.recentBpms = [];
    this.pendingJump = null;
  }

  public update(
    rawBpm: number | null,
    timestampMs: number,
    snrDb: number,
    confidence: number,
  ): number | null {
    if (rawBpm === null || !Number.isFinite(rawBpm)) {
      return this.smoothedBpm;
    }

    if (this.smoothedBpm === null || this.lastAcceptedAtMs === null) {
      this.smoothedBpm = rawBpm;
      this.lastAcceptedAtMs = timestampMs;
      this.recentBpms = [rawBpm];
      this.pendingJump = null;
      return this.smoothedBpm;
    }

    const dtSec = Math.max(0.01, (timestampMs - this.lastAcceptedAtMs) / 1000);
    const jumpBpm = Math.abs(rawBpm - this.smoothedBpm);

    // Reject unphysiological instantaneous jumps (> 18 BPM in < 1 s, or persistent large jumps
    // without sustained high SNR).
    const allowedByTime = dtSec >= 4.0 && snrDb >= 3.0;
    if (jumpBpm > this.maxInstantJumpBpm && !allowedByTime) {
      const pending = this.pendingJump;
      if (
        pending &&
        Math.abs(rawBpm - pending.bpmSum / pending.count) <= 10 &&
        timestampMs - pending.lastSeenMs <= 1500
      ) {
        pending.bpmSum += rawBpm;
        pending.snrSum += snrDb;
        pending.count += 1;
        pending.lastSeenMs = timestampMs;

        const sustainedMs = timestampMs - pending.firstSeenMs;
        const avgSnr = pending.snrSum / pending.count;
        if (
          sustainedMs >= this.sustainedJumpDurationMs &&
          pending.count >= this.sustainedJumpMinCount &&
          avgSnr >= this.sustainedJumpMinSnrDb
        ) {
          const targetBpm = pending.bpmSum / pending.count;
          this.recentBpms = [targetBpm];
          this.smoothedBpm = this.smoothedBpm + 0.65 * (targetBpm - this.smoothedBpm);
          this.lastAcceptedAtMs = timestampMs;
          this.pendingJump = null;
        }
      } else {
        this.pendingJump = {
          bpmSum: rawBpm,
          snrSum: snrDb,
          count: 1,
          firstSeenMs: timestampMs,
          lastSeenMs: timestampMs,
        };
      }
      return this.smoothedBpm;
    }

    // Physiological observation: clear pending jump and update median + adaptive EMA.
    this.pendingJump = null;
    this.recentBpms.push(rawBpm);
    if (this.recentBpms.length > this.medianWindowSize) {
      this.recentBpms.shift();
    }

    const medianBpm = computeMedian(this.recentBpms);

    // Adaptive smoothing time constant: faster convergence at high SNR & confidence.
    let tauSec = 3.5;
    if (snrDb >= 6 && confidence >= 65) {
      tauSec = 1.1;
    } else if (snrDb >= 2 && confidence >= 45) {
      tauSec = 2.0;
    }

    const alpha = Math.min(0.6, Math.max(0.08, 1 - Math.exp(-dtSec / tauSec)));
    this.smoothedBpm = this.smoothedBpm + alpha * (medianBpm - this.smoothedBpm);
    this.lastAcceptedAtMs = timestampMs;

    return this.smoothedBpm;
  }
}
