/**
 * Public exports for the PulseCam rPPG DSP engine.
 */

export {
  DEFAULT_FFT_SIZE,
  DEFAULT_MAX_FREQ_HZ,
  DEFAULT_MIN_FREQ_HZ,
  DEFAULT_SAMPLE_RATE_HZ,
  applyHannWindow,
  biquadFrequencyResponseMagnitude,
  butterworthBandpassZeroPhase,
  computeSpectrum,
  computeWelchSpectrum,
  despikeChannel,
  despikeRgbSeries,
  designButterworthHighpass2,
  designButterworthLowpass2,
  fftRadix2,
  filtfiltBiquad,
  getBiquadPoleRadius,
  linearDetrend,
  mean,
  nextPowerOfTwo,
  parabolicInterpolatePeak,
  resampleRgbBuffer,
  std,
  type BiquadCoefficients,
  type ParabolicPeakResult,
  type SpectrumResult,
  type TimedRgbSample,
  type UniformRgbSeries,
} from './dsp';

export {
  extractChromPulse,
  extractOmitPulse,
  extractPosPulse,
  type ChromOptions,
  type OmitOptions,
  type PosOptions,
} from './algorithms';

export {
  BpmTemporalTracker,
  estimateSpectralBpm,
  type BpmTrackerOptions,
  type SpectralBpmEstimate,
  type SpectralEstimateOptions,
} from './bpmEstimator';

export {
  assessSignalQuality,
  classifyQualityLevel,
  computeAlgorithmAgreement,
  computeLightingScore,
  computeMotionStability,
  computeSpectralSnrDb,
  type SignalQualityAssessment,
  type SignalQualityInput,
} from './signalQuality';

export {
  RppgProcessor,
  createIdleEstimate,
  fuseRoiRgb,
  type RppgProcessorOptions,
} from './rppgProcessor';
