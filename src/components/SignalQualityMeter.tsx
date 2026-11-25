import {
  Activity,
  CheckCircle2,
  Gauge,
  Info,
  ScanFace,
  Sun,
} from 'lucide-react';
import type { HeartRateEstimate, SignalQualityLevel } from '../types/rppg';

export interface SignalQualityMeterProps {
  estimate: HeartRateEstimate;
  faceDetected: boolean;
  isRunning: boolean;
}

const QUALITY_LABEL: Record<SignalQualityLevel, string> = {
  poor: 'Poor',
  fair: 'Fair',
  good: 'Good',
  excellent: 'Excellent',
};

function scoreTone(score: number): 'ok' | 'warn' | 'bad' {
  if (score >= 70) return 'ok';
  if (score >= 45) return 'warn';
  return 'bad';
}

export function SignalQualityMeter({
  estimate,
  faceDetected,
  isRunning,
}: SignalQualityMeterProps) {
  const { confidence, quality, guidance, diagnostics, status } = estimate;
  const clampedConfidence = isRunning ? Math.max(0, Math.min(100, Math.round(confidence))) : 0;
  const activeQualityClass = isRunning && status !== 'idle' ? quality : 'idle';
  const qualityText = isRunning && status !== 'idle' ? QUALITY_LABEL[quality] : 'Standby';

  const stillnessPct = Math.max(0, Math.min(100, Math.round(diagnostics.motionStability)));
  const lightingPct = Math.max(0, Math.min(100, Math.round(diagnostics.lightingScore)));
  const fpsVal = diagnostics.fps > 0 ? Math.round(diagnostics.fps) : 0;

  const agreementBpm = diagnostics.algorithmAgreementBpm;
  const hasAgreement = isRunning && agreementBpm !== null && Number.isFinite(agreementBpm);
  const fpsTooltip = hasAgreement
    ? `POS / CHROM agreement: ±${agreementBpm.toFixed(1)} BPM (SNR ${diagnostics.snrDb.toFixed(1)} dB)`
    : 'Effective frame processing rate (Hz)';

  const guidanceIconTone =
    !isRunning || status === 'idle'
      ? ''
      : quality === 'excellent' || quality === 'good'
      ? ' good'
      : quality === 'fair'
      ? ' fair'
      : '';

  return (
    <section className="glass-card quality-card" aria-label="Signal quality and coaching">
      <div className="quality-header">
        <span className="card-eyebrow">Signal Quality</span>
        <div className="quality-score-group">
          <span className="quality-pct tabular-nums">
            {isRunning ? `${clampedConfidence}%` : '--'}
          </span>
          <span className={`quality-badge ${activeQualityClass}`}>{qualityText}</span>
        </div>
      </div>

      {/* Confidence bar */}
      <div
        className="confidence-track"
        role="progressbar"
        aria-label="Signal confidence"
        aria-valuenow={clampedConfidence}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className={`confidence-fill ${activeQualityClass}`}
          style={{ width: `${clampedConfidence}%` }}
        />
      </div>

      {/* Real-time coaching text */}
      <div className="guidance-row" aria-live="polite">
        {quality === 'good' || quality === 'excellent' ? (
          <CheckCircle2 size={15} className={`guidance-icon${guidanceIconTone}`} />
        ) : (
          <Info size={15} className={`guidance-icon${guidanceIconTone}`} />
        )}
        <span>{guidance}</span>
      </div>

      {/* Compact 4-chip diagnostics row */}
      <div className="diagnostics-grid">
        <div className="diag-chip">
          <span className="diag-label">
            <ScanFace size={12} />
            Face
          </span>
          <span
            className={`diag-value${
              !isRunning ? '' : faceDetected ? ' ok' : ' warn'
            }`}
          >
            {!isRunning ? '--' : faceDetected ? 'Locked' : 'Searching'}
          </span>
        </div>

        <div className="diag-chip">
          <span className="diag-label">
            <Activity size={12} />
            Stillness
          </span>
          <span
            className={`diag-value tabular-nums${
              !isRunning ? '' : ` ${scoreTone(stillnessPct)}`
            }`}
          >
            {isRunning ? `${stillnessPct}%` : '--'}
          </span>
        </div>

        <div className="diag-chip">
          <span className="diag-label">
            <Sun size={12} />
            Lighting
          </span>
          <span
            className={`diag-value tabular-nums${
              !isRunning ? '' : ` ${scoreTone(lightingPct)}`
            }`}
          >
            {isRunning ? `${lightingPct}%` : '--'}
          </span>
        </div>

        <div className="diag-chip" title={fpsTooltip}>
          <span className="diag-label">
            <Gauge size={12} />
            FPS
          </span>
          <span
            className={`diag-value tabular-nums${
              !isRunning ? '' : fpsVal >= 20 ? ' ok' : fpsVal >= 14 ? ' warn' : ' bad'
            }`}
          >
            {isRunning && fpsVal > 0 ? `${fpsVal} Hz` : '--'}
          </span>
        </div>
      </div>

      {/* Subtle POS/CHROM cross-check footer when locked */}
      {hasAgreement && (
        <div className="diag-subtext tabular-nums" title={fpsTooltip}>
          <span>POS / CHROM Δ {agreementBpm.toFixed(1)} BPM</span>
          <span>SNR {diagnostics.snrDb.toFixed(1)} dB</span>
        </div>
      )}
    </section>
  );
}
