import type { CSSProperties } from 'react';
import { Activity, Heart } from 'lucide-react';
import type { HeartRateEstimate, MeasurementStatus } from '../types/rppg';

export interface BpmDisplayProps {
  estimate: HeartRateEstimate;
  isRunning: boolean;
}

function getStatusPill(
  status: MeasurementStatus,
  isRunning: boolean
): { label: string; tone: 'idle' | 'warming' | 'measuring' | 'warning' } {
  if (!isRunning || status === 'idle') {
    return { label: 'Ready', tone: 'idle' };
  }
  switch (status) {
    case 'initializing':
      return { label: 'Initializing', tone: 'warming' };
    case 'no-face':
      return { label: 'Center Face', tone: 'warning' };
    case 'warming-up':
      return { label: 'Calibrating', tone: 'warming' };
    case 'measuring':
      return { label: 'Live Lock', tone: 'measuring' };
    case 'low-signal':
      return { label: 'Low Signal', tone: 'warning' };
    case 'error':
      return { label: 'Camera Error', tone: 'warning' };
  }
}

function buildSparklinePaths(samples: number[]): { line: string; area: string; lastX: number; lastY: number } | null {
  if (samples.length < 4) return null;

  const width = 280;
  const height = 44;
  const padY = 5;
  const usableH = height - padY * 2;
  const stepX = width / (samples.length - 1);

  const coords: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < samples.length; i++) {
    const raw = samples[i];
    const clamped = Number.isFinite(raw) ? Math.max(-1, Math.min(1, raw)) : 0;
    const x = i * stepX;
    const y = padY + ((1 - clamped) / 2) * usableH;
    coords.push({ x, y });
  }

  const line = coords
    .map((pt, i) => `${i === 0 ? 'M' : 'L'}${pt.x.toFixed(1)},${pt.y.toFixed(1)}`)
    .join(' ');

  const last = coords[coords.length - 1];
  const area = `${line} L${width.toFixed(1)},${height} L0,${height} Z`;

  return { line, area, lastX: last.x, lastY: last.y };
}

export function BpmDisplay({ estimate, isRunning }: BpmDisplayProps) {
  const { bpm, status, diagnostics, waveform } = estimate;
  const isWarmingUp = isRunning && status === 'warming-up';
  const hasLockedBpm =
    isRunning &&
    bpm !== null &&
    Number.isFinite(bpm) &&
    status !== 'idle' &&
    status !== 'no-face' &&
    status !== 'error';

  const displayBpm = hasLockedBpm ? Math.round(bpm) : null;
  const clampedBpm = displayBpm !== null ? Math.max(40, Math.min(210, displayBpm)) : 72;
  const pulseDurationSec = 60 / clampedBpm;

  const pill = getStatusPill(status, isRunning);

  // Warm-up progress ring geometry
  const ringRadius = 28;
  const ringCircumference = 2 * Math.PI * ringRadius;
  const progress = Math.max(0, Math.min(1, diagnostics.bufferProgress));
  const ringOffset = ringCircumference * (1 - progress);
  const warmupPercent = Math.round(progress * 100);

  const sparkline = isRunning ? buildSparklinePaths(waveform) : null;

  const heartStyle = {
    '--pulse-duration': `${pulseDurationSec.toFixed(3)}s`,
  } as CSSProperties;

  return (
    <section className="glass-card bpm-card" aria-label="Live heart rate readout">
      <div className="bpm-header">
        <span className="card-eyebrow">Heart Rate</span>
        <span className={`status-pill ${pill.tone}`}>{pill.label}</span>
      </div>

      <div className="bpm-main">
        <div className="bpm-value-wrap" aria-live="polite" aria-atomic="true">
          <span className={`bpm-number tabular-nums${displayBpm === null ? ' muted' : ''}`}>
            {displayBpm !== null ? displayBpm : '--'}
          </span>
          <span className="bpm-unit">BPM</span>
        </div>

        <div className="bpm-heart-badge" style={heartStyle}>
          {isWarmingUp && (
            <svg className="warmup-ring-svg" viewBox="0 0 64 64" aria-hidden="true">
              <circle className="warmup-ring-track" cx="32" cy="32" r={ringRadius} />
              <circle
                className="warmup-ring-progress"
                cx="32"
                cy="32"
                r={ringRadius}
                strokeDasharray={ringCircumference}
                strokeDashoffset={ringOffset}
              />
            </svg>
          )}

          <div className={`heart-icon-wrap${displayBpm !== null ? ' beating' : ''}`}>
            <Heart
              size={24}
              className="heart-svg-icon"
              fill={displayBpm !== null ? 'currentColor' : 'none'}
              strokeWidth={2.2}
            />
          </div>

          {isWarmingUp && (
            <span className="warmup-pct tabular-nums" aria-label={`Calibrating ${warmupPercent}%`}>
              {warmupPercent}%
            </span>
          )}
        </div>
      </div>

      {/* Minimal live filtered pulse sparkline */}
      <div className="sparkline-wrap" aria-label="Live pulse waveform">
        {sparkline ? (
          <>
            <svg
              className="sparkline-svg"
              viewBox="0 0 280 44"
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              <defs>
                <linearGradient id="pulseAreaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#fb7185" stopOpacity="0.32" />
                  <stop offset="100%" stopColor="#fb7185" stopOpacity="0" />
                </linearGradient>
              </defs>
              <path d={sparkline.area} fill="url(#pulseAreaGrad)" />
              <path
                d={sparkline.line}
                fill="none"
                stroke="#fb7185"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                vectorEffect="non-scaling-stroke"
              />
              <circle cx={sparkline.lastX} cy={sparkline.lastY} r="2.8" fill="#fda4af" />
            </svg>
            <span className="sparkline-label">rPPG Wave</span>
          </>
        ) : (
          <div className="sparkline-placeholder">
            <Activity size={14} />
            <span>
              {isRunning
                ? 'Extracting optical pulse signal…'
                : 'Pulse wave appears once scanning starts'}
            </span>
          </div>
        )}
      </div>
    </section>
  );
}
