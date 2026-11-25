/**
 * Signal Quality Index (SQI), spectral SNR, POS/CHROM agreement,
 * confidence scoring, measurement status, and real-time user guidance.
 */

import type {
  MeasurementStatus,
  QualityDiagnostics,
  SignalQualityLevel,
} from '../types/rppg';
import {
  DEFAULT_MAX_FREQ_HZ,
  DEFAULT_MIN_FREQ_HZ,
  SpectrumResult,
} from './dsp';

export interface SignalQualityInput {
  faceDetected: boolean;
  bufferDurationSec: number;
  warmUpSec: number;
  windowSec: number;
  snrDb: number;
  posBpm: number | null;
  chromBpm: number | null;
  motionScore: number;
  brightness: number;
  saturationFraction: number;
  fps: number;
}

export interface SignalQualityAssessment {
  confidence: number;
  quality: SignalQualityLevel;
  status: MeasurementStatus;
  guidance: string;
  diagnostics: QualityDiagnostics;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * Computes spectral SNR in dB:
 *   10 * log10(signalPower / noisePower)
 * where signalPower sums bins within +/-0.15 Hz of `peakFreqHz` and +/-0.15 Hz
 * of its 2nd harmonic (`2 * peakFreqHz`), and noisePower sums all remaining bins
 * in `[minFreqHz, maxFreqHz]` (default 0.70–3.50 Hz).
 */
export function computeSpectralSnrDb(
  spectrum: SpectrumResult,
  peakFreqHz: number | null,
  minFreqHz = DEFAULT_MIN_FREQ_HZ,
  maxFreqHz = DEFAULT_MAX_FREQ_HZ,
  bandwidthHz = 0.15,
): number {
  if (peakFreqHz === null || !Number.isFinite(peakFreqHz)) {
    return -15;
  }

  const { frequencies, power } = spectrum;
  const harmonic2Hz = 2 * peakFreqHz;
  let signalPower = 0;
  let noisePower = 0;

  for (let i = 0; i < frequencies.length; i++) {
    const f = frequencies[i];
    if (f < minFreqHz || f > maxFreqHz) continue;

    const inFundamental = Math.abs(f - peakFreqHz) <= bandwidthHz;
    const inSecondHarmonic =
      harmonic2Hz <= maxFreqHz + bandwidthHz && Math.abs(f - harmonic2Hz) <= bandwidthHz;

    if (inFundamental || inSecondHarmonic) {
      signalPower += power[i];
    } else {
      noisePower += power[i];
    }
  }

  if (signalPower <= 1e-18 && noisePower <= 1e-18) {
    return -15;
  }
  if (noisePower <= 1e-18) {
    return 30;
  }
  if (signalPower <= 1e-18) {
    return -15;
  }

  const snrDb = 10 * Math.log10(signalPower / noisePower);
  return clamp(snrDb, -15, 30);
}

/**
 * Computes absolute BPM difference between POS and CHROM estimates.
 */
export function computeAlgorithmAgreement(
  posBpm: number | null,
  chromBpm: number | null,
): number | null {
  if (
    posBpm === null ||
    chromBpm === null ||
    !Number.isFinite(posBpm) ||
    !Number.isFinite(chromBpm)
  ) {
    return null;
  }
  return Math.abs(posBpm - chromBpm);
}

/**
 * Converts mean facial brightness (0..255, ideal 65..210) and clipped-pixel
 * fraction (0..1) into a 0..100 lighting quality score.
 */
export function computeLightingScore(
  brightness: number,
  saturationFraction: number,
): number {
  let brightnessScore: number;
  if (brightness >= 65 && brightness <= 210) {
    brightnessScore = 100;
  } else if (brightness < 65) {
    brightnessScore = clamp((brightness - 20) / 45, 0, 1) * 100;
  } else {
    brightnessScore = clamp((250 - brightness) / 40, 0, 1) * 100;
  }

  const satPenalty = clamp(saturationFraction * 220, 0, 75);
  return Math.round(clamp(brightnessScore - satPenalty, 0, 100));
}

/**
 * Converts frame-to-frame motionScore (0 = still, >= 1 = high motion)
 * into a 0..100 motion stability score.
 */
export function computeMotionStability(motionScore: number): number {
  const normalized = clamp(motionScore, 0, 1.2);
  const stability = clamp(1 - Math.pow(normalized, 0.95), 0, 1) * 100;
  return Math.round(stability);
}

function computeFpsScore(fps: number): number {
  if (fps >= 22) return 100;
  if (fps >= 15) return 65 + ((fps - 15) / 7) * 35;
  if (fps >= 10) return 20 + ((fps - 10) / 5) * 45;
  return clamp((fps / 10) * 20, 0, 20);
}

function computeSnrScore(snrDb: number): number {
  // Maps [-4 dB .. 10 dB] -> [0 .. 100]
  return clamp(((snrDb + 4) / 14) * 100, 0, 100);
}

function computeAgreementScore(agreementBpm: number | null): number {
  if (agreementBpm === null) return 50;
  if (agreementBpm <= 2) return 100;
  if (agreementBpm <= 6) return 100 - ((agreementBpm - 2) / 4) * 25;
  if (agreementBpm <= 16) return 75 - ((agreementBpm - 6) / 10) * 60;
  return clamp(15 - ((agreementBpm - 16) / 10) * 15, 0, 15);
}

export function classifyQualityLevel(confidence: number): SignalQualityLevel {
  if (confidence >= 80) return 'excellent';
  if (confidence >= 60) return 'good';
  if (confidence >= 38) return 'fair';
  return 'poor';
}

function selectGuidance(params: {
  status: MeasurementStatus;
  quality: SignalQualityLevel;
  brightness: number;
  saturationFraction: number;
  motionStability: number;
  lightingScore: number;
  fps: number;
}): string {
  const {
    status,
    quality,
    brightness,
    saturationFraction,
    motionStability,
    lightingScore,
    fps,
  } = params;

  if (status === 'no-face') {
    return 'Center your face in the oval frame';
  }
  if (brightness < 55 || (lightingScore < 45 && brightness < 65)) {
    return 'Too dark — move toward a brighter light source';
  }
  if (brightness > 220 || saturationFraction > 0.25) {
    return 'Too bright — avoid direct glare on your face';
  }
  if (motionStability < 45) {
    return 'Hold still for a steadier heart rate reading';
  }
  if (fps > 0 && fps < 14) {
    return 'Low camera frame rate — close background apps';
  }
  if (status === 'warming-up') {
    return 'Hold still — calibrating pulse signal...';
  }
  if (status === 'low-signal') {
    return 'Weak pulse signal — stay still in even lighting';
  }

  switch (quality) {
    case 'excellent':
      return 'Strong pulse lock — keep holding steady';
    case 'good':
      return 'Good signal — stay still for best accuracy';
    case 'fair':
      return 'Fair signal — face even light and stay still';
    default:
      return 'Improving signal — hold steady';
  }
}

/**
 * Combines spectral SNR, POS/CHROM agreement, motion stability, lighting score,
 * effective FPS, and buffer progress into confidence (0..100), quality level,
 * measurement status, and actionable user guidance.
 */
export function assessSignalQuality(input: SignalQualityInput): SignalQualityAssessment {
  const {
    faceDetected,
    bufferDurationSec,
    warmUpSec,
    windowSec,
    snrDb,
    posBpm,
    chromBpm,
    motionScore,
    brightness,
    saturationFraction,
    fps,
  } = input;

  const bufferProgress = clamp(bufferDurationSec / Math.max(windowSec, 1), 0, 1);
  const lightingScore = computeLightingScore(brightness, saturationFraction);
  const motionStability = computeMotionStability(motionScore);
  const algorithmAgreementBpm = computeAlgorithmAgreement(posBpm, chromBpm);

  const diagnostics: QualityDiagnostics = {
    snrDb: Number(snrDb.toFixed(2)),
    algorithmAgreementBpm:
      algorithmAgreementBpm !== null ? Number(algorithmAgreementBpm.toFixed(2)) : null,
    motionStability,
    lightingScore,
    fps: Number(fps.toFixed(1)),
    bufferProgress: Number(bufferProgress.toFixed(3)),
  };

  if (!faceDetected) {
    const status: MeasurementStatus = 'no-face';
    const quality: SignalQualityLevel = 'poor';
    return {
      confidence: 0,
      quality,
      status,
      guidance: selectGuidance({
        status,
        quality,
        brightness,
        saturationFraction,
        motionStability,
        lightingScore,
        fps,
      }),
      diagnostics,
    };
  }

  const fpsScore = computeFpsScore(fps > 0 ? fps : 30);

  if (bufferDurationSec < warmUpSec) {
    const status: MeasurementStatus = 'warming-up';
    const readiness =
      (lightingScore * 0.45 + motionStability * 0.4 + fpsScore * 0.15) *
      (bufferDurationSec / Math.max(warmUpSec, 1)) *
      0.35;
    const confidence = Math.round(clamp(readiness, 0, 35));
    const quality = classifyQualityLevel(confidence);
    return {
      confidence,
      quality,
      status,
      guidance: selectGuidance({
        status,
        quality,
        brightness,
        saturationFraction,
        motionStability,
        lightingScore,
        fps,
      }),
      diagnostics,
    };
  }

  const snrScore = computeSnrScore(snrDb);
  const agreementScore = computeAgreementScore(algorithmAgreementBpm);

  const weighted =
    snrScore * 0.38 +
    agreementScore * 0.22 +
    motionStability * 0.18 +
    lightingScore * 0.14 +
    fpsScore * 0.08;

  // Gentle maturity ramp from warm-up (~6s) to full window (~14s).
  const maturityFactor = 0.8 + 0.2 * bufferProgress;
  let confidence = weighted * maturityFactor;

  // Hard caps when fundamental acquisition constraints are violated.
  const severeMotion = motionStability < 30;
  const severeLighting = lightingScore < 30;
  const severeSnr = snrDb < -1.0;
  const severeFps = fps > 0 && fps < 10;

  if (severeMotion) {
    confidence = Math.min(confidence, 30);
  }
  if (severeLighting) {
    confidence = Math.min(confidence, 30);
  }
  if (severeSnr) {
    confidence = Math.min(confidence, 32);
  }
  if (severeFps) {
    confidence = Math.min(confidence, 25);
  }

  const finalConfidence = Math.round(clamp(confidence, 0, 100));
  const quality = classifyQualityLevel(finalConfidence);

  const status: MeasurementStatus =
    finalConfidence < 32 || severeMotion || severeLighting || severeSnr || severeFps
      ? 'low-signal'
      : 'measuring';

  return {
    confidence: finalConfidence,
    quality,
    status,
    guidance: selectGuidance({
      status,
      quality,
      brightness,
      saturationFraction,
      motionStability,
      lightingScore,
      fps,
    }),
    diagnostics,
  };
}
