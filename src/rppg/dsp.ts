/**
 * Core Digital Signal Processing primitives for on-device facial rPPG.
 * Pure TypeScript, zero DOM dependencies.
 */

export interface TimedRgbSample {
  timestampMs: number;
  r: number;
  g: number;
  b: number;
}

export interface UniformRgbSeries {
  sampleRateHz: number;
  r: Float64Array;
  g: Float64Array;
  b: Float64Array;
  durationSec: number;
  startTimestampMs: number;
}

export interface BiquadCoefficients {
  b0: number;
  b1: number;
  b2: number;
  a1: number;
  a2: number;
}

export interface SpectrumResult {
  frequencies: Float64Array;
  magnitudes: Float64Array;
  power: Float64Array;
  fftSize: number;
  sampleRateHz: number;
  binResolutionHz: number;
}

export interface ParabolicPeakResult {
  index: number;
  value: number;
  delta: number;
}

export const DEFAULT_SAMPLE_RATE_HZ = 30;
export const DEFAULT_MIN_FREQ_HZ = 0.70; // 42 BPM
export const DEFAULT_MAX_FREQ_HZ = 3.50; // 210 BPM
export const DEFAULT_FFT_SIZE = 4096;

export function mean(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += values[i];
  }
  return sum / n;
}

export function std(values: ArrayLike<number>, precomputedMean?: number): number {
  const n = values.length;
  if (n < 2) return 0;
  const m = precomputedMean ?? mean(values);
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = values[i] - m;
    sumSq += d * d;
  }
  return Math.sqrt(sumSq / n);
}

/**
 * Uniformly resamples an irregular sequence of (timestampMs, r, g, b) samples
 * onto a uniform grid at `sampleRateHz` using linear interpolation.
 */
export function resampleRgbBuffer(
  samples: readonly TimedRgbSample[],
  sampleRateHz = DEFAULT_SAMPLE_RATE_HZ,
): UniformRgbSeries {
  if (samples.length < 2 || sampleRateHz <= 0) {
    return {
      sampleRateHz,
      r: new Float64Array(0),
      g: new Float64Array(0),
      b: new Float64Array(0),
      durationSec: 0,
      startTimestampMs: samples.length > 0 ? samples[0].timestampMs : 0,
    };
  }

  // Ensure strictly increasing timestamps (deduplicate equal/out-of-order entries).
  const clean: TimedRgbSample[] = [samples[0]];
  for (let i = 1; i < samples.length; i++) {
    if (samples[i].timestampMs > clean[clean.length - 1].timestampMs) {
      clean.push(samples[i]);
    }
  }

  if (clean.length < 2) {
    return {
      sampleRateHz,
      r: new Float64Array(0),
      g: new Float64Array(0),
      b: new Float64Array(0),
      durationSec: 0,
      startTimestampMs: clean[0].timestampMs,
    };
  }

  const startMs = clean[0].timestampMs;
  const endMs = clean[clean.length - 1].timestampMs;
  const durationSec = (endMs - startMs) / 1000;
  const stepMs = 1000 / sampleRateHz;
  const count = Math.floor(durationSec * sampleRateHz) + 1;

  const rOut = new Float64Array(count);
  const gOut = new Float64Array(count);
  const bOut = new Float64Array(count);

  let srcIdx = 0;
  const lastSrc = clean.length - 1;

  for (let k = 0; k < count; k++) {
    const targetMs = startMs + k * stepMs;
    while (srcIdx < lastSrc - 1 && clean[srcIdx + 1].timestampMs < targetMs) {
      srcIdx++;
    }

    const s0 = clean[srcIdx];
    const s1 = clean[Math.min(srcIdx + 1, lastSrc)];
    const dt = s1.timestampMs - s0.timestampMs;
    const alpha = dt > 0 ? Math.min(1, Math.max(0, (targetMs - s0.timestampMs) / dt)) : 0;

    rOut[k] = s0.r + alpha * (s1.r - s0.r);
    gOut[k] = s0.g + alpha * (s1.g - s0.g);
    bOut[k] = s0.b + alpha * (s1.b - s0.b);
  }

  return {
    sampleRateHz,
    r: rOut,
    g: gOut,
    b: bOut,
    durationSec,
    startTimestampMs: startMs,
  };
}

/**
 * Removes the best-fit linear trend (least-squares line) from a signal.
 */
export function linearDetrend(signal: Float64Array): Float64Array {
  const n = signal.length;
  const out = new Float64Array(n);
  if (n <= 1) return out;

  const yMean = mean(signal);
  const xMean = (n - 1) / 2;

  let covXY = 0;
  let varX = 0;
  for (let i = 0; i < n; i++) {
    const dx = i - xMean;
    covXY += dx * (signal[i] - yMean);
    varX += dx * dx;
  }

  const slope = varX > 0 ? covXY / varX : 0;
  for (let i = 0; i < n; i++) {
    out[i] = signal[i] - (yMean + slope * (i - xMean));
  }
  return out;
}

/**
 * Applies a periodic/symmetric Hann window to reduce spectral leakage.
 */
export function applyHannWindow(signal: Float64Array): Float64Array {
  const n = signal.length;
  const out = new Float64Array(n);
  if (n === 0) return out;
  if (n === 1) {
    out[0] = signal[0];
    return out;
  }

  const denom = n - 1;
  const twoPi = 2 * Math.PI;
  for (let i = 0; i < n; i++) {
    const w = 0.5 * (1 - Math.cos((twoPi * i) / denom));
    out[i] = signal[i] * w;
  }
  return out;
}

/**
 * Designs a 2nd-order Butterworth low-pass biquad via the bilinear transform.
 */
export function designButterworthLowpass2(
  cutoffHz: number,
  sampleRateHz: number,
): BiquadCoefficients {
  const nyquist = sampleRateHz / 2;
  const fc = Math.min(Math.max(cutoffHz, 1e-4), nyquist * 0.99);
  const k = Math.tan((Math.PI * fc) / sampleRateHz);
  const k2 = k * k;
  const sqrt2K = Math.SQRT2 * k;
  const norm = 1 / (1 + sqrt2K + k2);

  return {
    b0: k2 * norm,
    b1: 2 * k2 * norm,
    b2: k2 * norm,
    a1: 2 * (k2 - 1) * norm,
    a2: (1 - sqrt2K + k2) * norm,
  };
}

/**
 * Designs a 2nd-order Butterworth high-pass biquad via the bilinear transform.
 */
export function designButterworthHighpass2(
  cutoffHz: number,
  sampleRateHz: number,
): BiquadCoefficients {
  const nyquist = sampleRateHz / 2;
  const fc = Math.min(Math.max(cutoffHz, 1e-4), nyquist * 0.99);
  const k = Math.tan((Math.PI * fc) / sampleRateHz);
  const k2 = k * k;
  const sqrt2K = Math.SQRT2 * k;
  const norm = 1 / (1 + sqrt2K + k2);

  return {
    b0: norm,
    b1: -2 * norm,
    b2: norm,
    a1: 2 * (k2 - 1) * norm,
    a2: (1 - sqrt2K + k2) * norm,
  };
}

/**
 * Computes the maximum pole radius of a biquad section.
 * A stable digital filter strictly satisfies radius < 1.
 */
export function getBiquadPoleRadius(coeff: BiquadCoefficients): number {
  const { a1, a2 } = coeff;
  const disc = a1 * a1 - 4 * a2;
  if (disc < 0) {
    // Complex conjugate poles: |p| = sqrt(a2)
    return Math.sqrt(Math.max(0, a2));
  }
  const sqrtDisc = Math.sqrt(disc);
  const p1 = Math.abs((-a1 + sqrtDisc) / 2);
  const p2 = Math.abs((-a1 - sqrtDisc) / 2);
  return Math.max(p1, p2);
}

/**
 * Evaluates |H(e^(j*2*pi*f/fs))| for a biquad section at `freqHz`.
 */
export function biquadFrequencyResponseMagnitude(
  coeff: BiquadCoefficients,
  freqHz: number,
  sampleRateHz: number,
): number {
  const w = (2 * Math.PI * freqHz) / sampleRateHz;
  const cos1 = Math.cos(w);
  const sin1 = Math.sin(w);
  const cos2 = Math.cos(2 * w);
  const sin2 = Math.sin(2 * w);

  const numRe = coeff.b0 + coeff.b1 * cos1 + coeff.b2 * cos2;
  const numIm = -(coeff.b1 * sin1 + coeff.b2 * sin2);
  const denRe = 1 + coeff.a1 * cos1 + coeff.a2 * cos2;
  const denIm = -(coeff.a1 * sin1 + coeff.a2 * sin2);

  const numMag = Math.hypot(numRe, numIm);
  const denMag = Math.hypot(denRe, denIm);
  return denMag > 0 ? numMag / denMag : 0;
}

/**
 * Applies a Direct Form II Transposed biquad filter with steady-state initial
 * conditions (`zi`) scaled by `signal[0]` to eliminate startup step transients.
 */
function applyBiquadForward(signal: Float64Array, coeff: BiquadCoefficients): Float64Array {
  const n = signal.length;
  const out = new Float64Array(n);
  if (n === 0) return out;

  const { b0, b1, b2, a1, a2 } = coeff;
  const denom = 1 + a1 + a2;
  const dcGain = Math.abs(denom) > 1e-12 ? (b0 + b1 + b2) / denom : 0;

  // Steady-state DF-II transposed state for unit step input.
  const zi2 = b2 - a2 * dcGain;
  const zi1 = b1 - a1 * dcGain + zi2;

  const x0 = signal[0];
  let z1 = zi1 * x0;
  let z2 = zi2 * x0;

  for (let i = 0; i < n; i++) {
    const x = signal[i];
    const y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    out[i] = y;
  }
  return out;
}

/**
 * Zero-phase (forward-backward) biquad filtering with odd-symmetric endpoint
 * reflection padding (equivalent to scipy.signal.filtfilt).
 */
export function filtfiltBiquad(
  signal: Float64Array,
  coeff: BiquadCoefficients,
): Float64Array {
  const n = signal.length;
  if (n < 4) {
    return Float64Array.from(signal);
  }

  const padLen = Math.min(24, n - 1);
  const paddedLen = n + 2 * padLen;
  const padded = new Float64Array(paddedLen);

  // Odd reflection at left boundary: 2*x[0] - x[padLen .. 1]
  const first = signal[0];
  for (let i = 0; i < padLen; i++) {
    padded[i] = 2 * first - signal[padLen - i];
  }
  // Center copy
  padded.set(signal, padLen);
  // Odd reflection at right boundary: 2*x[n-1] - x[n-2 .. n-1-padLen]
  const last = signal[n - 1];
  for (let i = 0; i < padLen; i++) {
    padded[padLen + n + i] = 2 * last - signal[n - 2 - i];
  }

  // Forward pass
  const forward = applyBiquadForward(padded, coeff);

  // Reverse
  const reversed = new Float64Array(paddedLen);
  for (let i = 0; i < paddedLen; i++) {
    reversed[i] = forward[paddedLen - 1 - i];
  }

  // Backward pass
  const backward = applyBiquadForward(reversed, coeff);

  // Reverse and strip padding
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = backward[paddedLen - 1 - (padLen + i)];
  }
  return out;
}

/**
 * Zero-phase 2nd-order Butterworth bandpass filter (cascaded HP + LP biquads,
 * applied forward-backward) for 0.70 Hz – 3.50 Hz (42 – 210 BPM).
 */
export function butterworthBandpassZeroPhase(
  signal: Float64Array,
  sampleRateHz = DEFAULT_SAMPLE_RATE_HZ,
  lowCutHz = DEFAULT_MIN_FREQ_HZ,
  highCutHz = DEFAULT_MAX_FREQ_HZ,
): Float64Array {
  if (signal.length < 4) {
    return Float64Array.from(signal);
  }
  const hp = designButterworthHighpass2(lowCutHz, sampleRateHz);
  const lp = designButterworthLowpass2(highCutHz, sampleRateHz);
  const highPassed = filtfiltBiquad(signal, hp);
  return filtfiltBiquad(highPassed, lp);
}

/**
 * Returns the smallest power of two >= `value`.
 */
export function nextPowerOfTwo(value: number): number {
  let n = 1;
  while (n < value) {
    n <<= 1;
  }
  return n;
}

/**
 * In-place iterative Cooley-Tukey radix-2 FFT.
 * `real` and `imag` must share the same power-of-two length.
 */
export function fftRadix2(real: Float64Array, imag: Float64Array): void {
  const n = real.length;
  if (n !== imag.length || n === 0 || (n & (n - 1)) !== 0) {
    throw new Error(`fftRadix2 requires equal power-of-two arrays, got ${n} and ${imag.length}`);
  }

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      const tmpR = real[i];
      real[i] = real[j];
      real[j] = tmpR;
      const tmpI = imag[i];
      imag[i] = imag[j];
      imag[j] = tmpI;
    }
  }

  // Butterfly stages
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angleStep = (-2 * Math.PI) / len;
    const wStepR = Math.cos(angleStep);
    const wStepI = Math.sin(angleStep);

    for (let i = 0; i < n; i += len) {
      let wR = 1;
      let wI = 0;
      for (let j = 0; j < halfLen; j++) {
        const uIdx = i + j;
        const vIdx = uIdx + halfLen;
        const uR = real[uIdx];
        const uI = imag[uIdx];
        const vR = real[vIdx] * wR - imag[vIdx] * wI;
        const vI = real[vIdx] * wI + imag[vIdx] * wR;

        real[uIdx] = uR + vR;
        imag[uIdx] = uI + vI;
        real[vIdx] = uR - vR;
        imag[vIdx] = uI - vI;

        const nextWR = wR * wStepR - wI * wStepI;
        wI = wR * wStepI + wI * wStepR;
        wR = nextWR;
      }
    }
  }
}

/**
 * Computes the one-sided zero-padded FFT magnitude and power spectrum.
 */
export function computeSpectrum(
  signal: Float64Array,
  sampleRateHz = DEFAULT_SAMPLE_RATE_HZ,
  minFftSize = DEFAULT_FFT_SIZE,
): SpectrumResult {
  const fftSize = nextPowerOfTwo(Math.max(minFftSize, signal.length, 2));
  const real = new Float64Array(fftSize);
  const imag = new Float64Array(fftSize);
  real.set(signal);

  fftRadix2(real, imag);

  const halfBins = (fftSize >> 1) + 1;
  const frequencies = new Float64Array(halfBins);
  const magnitudes = new Float64Array(halfBins);
  const power = new Float64Array(halfBins);
  const binResolutionHz = sampleRateHz / fftSize;

  for (let k = 0; k < halfBins; k++) {
    const re = real[k];
    const im = imag[k];
    const p = re * re + im * im;
    frequencies[k] = k * binResolutionHz;
    power[k] = p;
    magnitudes[k] = Math.sqrt(p);
  }

  return {
    frequencies,
    magnitudes,
    power,
    fftSize,
    sampleRateHz,
    binResolutionHz,
  };
}

/**
 * Quadratic (parabolic) interpolation around a discrete spectral peak bin
 * for sub-bin frequency estimation (< 0.2 BPM precision at 4096-pt FFT / 30 Hz).
 */
export function parabolicInterpolatePeak(
  values: Float64Array,
  peakIndex: number,
): ParabolicPeakResult {
  const n = values.length;
  if (n === 0 || peakIndex < 0 || peakIndex >= n) {
    return { index: peakIndex, value: 0, delta: 0 };
  }
  if (peakIndex === 0 || peakIndex === n - 1) {
    return { index: peakIndex, value: values[peakIndex], delta: 0 };
  }

  const alpha = values[peakIndex - 1];
  const beta = values[peakIndex];
  const gamma = values[peakIndex + 1];
  const denom = alpha - 2 * beta + gamma;

  if (Math.abs(denom) < 1e-18) {
    return { index: peakIndex, value: beta, delta: 0 };
  }

  const rawDelta = (0.5 * (alpha - gamma)) / denom;
  const delta = Math.max(-0.5, Math.min(0.5, rawDelta));
  const value = beta - 0.25 * (alpha - gamma) * delta;

  return {
    index: peakIndex + delta,
    value,
    delta,
  };
}
