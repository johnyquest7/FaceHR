/**
 * rPPG Pulse Extraction Algorithms:
 * - POS: Plane-Orthogonal-to-Skin (Wang et al., IEEE TBME 2017)
 * - CHROM: Chrominance-based rPPG (De Haan & Jeanne, IEEE TBME 2013)
 */

import {
  DEFAULT_MAX_FREQ_HZ,
  DEFAULT_MIN_FREQ_HZ,
  UniformRgbSeries,
  butterworthBandpassZeroPhase,
  linearDetrend,
  mean,
  std,
} from './dsp';

export interface PosOptions {
  /** Sliding window length in seconds (default 1.6 s per Wang et al. 2017) */
  windowSec?: number;
  /** Whether to detrend and bandpass-filter the overlap-added pulse signal */
  bandpass?: boolean;
  minFreqHz?: number;
  maxFreqHz?: number;
}

export interface ChromOptions {
  minFreqHz?: number;
  maxFreqHz?: number;
}

/**
 * Extracts a blood-volume pulse (BVP) signal using the POS algorithm
 * (Wang et al., "Algorithmic Principles of Remote PPG", IEEE TBME 2017).
 *
 * For each sliding window (~1.6 s):
 *   1. Temporal mean normalization: C_n = C / mean(C)
 *   2. Orthogonal projections:
 *      S1 = G_n - B_n
 *      S2 = -2 * R_n + G_n + B_n
 *   3. Adaptive alpha tuning:
 *      alpha = std(S1) / max(std(S2), 1e-8)
 *      h = S1 + alpha * S2
 *   4. Zero-mean overlap-add into output pulse vector H.
 */
export function extractPosPulse(
  series: UniformRgbSeries,
  options: PosOptions = {},
): Float64Array {
  const { r, g, b, sampleRateHz } = series;
  const n = r.length;
  if (n < 4) {
    return new Float64Array(n);
  }

  const windowSec = options.windowSec ?? 1.6;
  const applyBandpass = options.bandpass ?? true;
  const minFreqHz = options.minFreqHz ?? DEFAULT_MIN_FREQ_HZ;
  const maxFreqHz = options.maxFreqHz ?? DEFAULT_MAX_FREQ_HZ;

  const winLen = Math.max(4, Math.min(n, Math.round(windowSec * sampleRateHz)));
  const out = new Float64Array(n);
  const s1 = new Float64Array(winLen);
  const s2 = new Float64Array(winLen);
  const h = new Float64Array(winLen);

  for (let start = 0; start <= n - winLen; start++) {
    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    for (let j = 0; j < winLen; j++) {
      const idx = start + j;
      sumR += r[idx];
      sumG += g[idx];
      sumB += b[idx];
    }

    const meanR = sumR / winLen;
    const meanG = sumG / winLen;
    const meanB = sumB / winLen;
    if (meanR <= 1e-6 || meanG <= 1e-6 || meanB <= 1e-6) {
      continue;
    }

    const invR = 1 / meanR;
    const invG = 1 / meanG;
    const invB = 1 / meanB;

    let sumS1 = 0;
    let sumS2 = 0;
    for (let j = 0; j < winLen; j++) {
      const idx = start + j;
      const rn = r[idx] * invR;
      const gn = g[idx] * invG;
      const bn = b[idx] * invB;

      const v1 = gn - bn;
      const v2 = -2 * rn + gn + bn;
      s1[j] = v1;
      s2[j] = v2;
      sumS1 += v1;
      sumS2 += v2;
    }

    const meanS1 = sumS1 / winLen;
    const meanS2 = sumS2 / winLen;
    const stdS1 = std(s1, meanS1);
    const stdS2 = std(s2, meanS2);
    const alpha = stdS2 > 1e-8 ? Math.min(stdS1 / Math.max(stdS2, 1e-8), 50) : 0;

    let sumH = 0;
    for (let j = 0; j < winLen; j++) {
      const val = s1[j] + alpha * s2[j];
      h[j] = val;
      sumH += val;
    }
    const meanH = sumH / winLen;

    for (let j = 0; j < winLen; j++) {
      out[start + j] += h[j] - meanH;
    }
  }

  const detrended = linearDetrend(out);
  if (!applyBandpass) {
    return detrended;
  }
  return butterworthBandpassZeroPhase(detrended, sampleRateHz, minFreqHz, maxFreqHz);
}

/**
 * Extracts a blood-volume pulse (BVP) signal using the CHROM algorithm
 * (De Haan & Jeanne, "Robust Pulse Rate From Chrominance-Based rPPG", IEEE TBME 2013).
 *
 *   1. Temporal mean normalization: C_n = C / mean(C)
 *   2. Chrominance signals:
 *      X = 3 * R_n - 2 * G_n
 *      Y = 1.5 * R_n + G_n - 1.5 * B_n
 *   3. Zero-phase bandpass filtering: X_f, Y_f
 *   4. Alpha tuning:
 *      alpha = std(X_f) / max(std(Y_f), 1e-8)
 *      S = X_f - alpha * Y_f
 */
export function extractChromPulse(
  series: UniformRgbSeries,
  options: ChromOptions = {},
): Float64Array {
  const { r, g, b, sampleRateHz } = series;
  const n = r.length;
  if (n < 4) {
    return new Float64Array(n);
  }

  const minFreqHz = options.minFreqHz ?? DEFAULT_MIN_FREQ_HZ;
  const maxFreqHz = options.maxFreqHz ?? DEFAULT_MAX_FREQ_HZ;

  const meanR = mean(r);
  const meanG = mean(g);
  const meanB = mean(b);
  if (meanR <= 1e-6 || meanG <= 1e-6 || meanB <= 1e-6) {
    return new Float64Array(n);
  }

  const invR = 1 / meanR;
  const invG = 1 / meanG;
  const invB = 1 / meanB;

  const x = new Float64Array(n);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const rn = r[i] * invR;
    const gn = g[i] * invG;
    const bn = b[i] * invB;
    x[i] = 3 * rn - 2 * gn;
    y[i] = 1.5 * rn + gn - 1.5 * bn;
  }

  const xf = butterworthBandpassZeroPhase(linearDetrend(x), sampleRateHz, minFreqHz, maxFreqHz);
  const yf = butterworthBandpassZeroPhase(linearDetrend(y), sampleRateHz, minFreqHz, maxFreqHz);

  const stdX = std(xf);
  const stdY = std(yf);
  const alpha = stdY > 1e-8 ? Math.min(stdX / Math.max(stdY, 1e-8), 50) : 0;

  const s = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    s[i] = xf[i] - alpha * yf[i];
  }

  return linearDetrend(s);
}
