import type { RefObject } from 'react';
import {
  AlertTriangle,
  Camera,
  Loader2,
  Play,
  RefreshCw,
  ShieldAlert,
  Sparkles,
  SwitchCamera,
} from 'lucide-react';
import type {
  CameraErrorInfo,
  FacingMode,
  MeasurementStatus,
  RoiBox,
  SignalQualityLevel,
  TrackingMode,
} from '../types/rppg';

export interface CameraViewfinderProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  simCanvasRef: RefObject<HTMLCanvasElement | null>;
  isRunning: boolean;
  isStarting: boolean;
  isSimulationMode: boolean;
  facingMode: FacingMode;
  hasMultipleCameras: boolean;
  faceDetected: boolean;
  roiBoxes: RoiBox[];
  trackingMode: TrackingMode;
  status: MeasurementStatus;
  quality: SignalQualityLevel;
  guidance: string;
  cameraError: CameraErrorInfo | null;
  onStart: () => void;
  onStartSimulation: () => void;
  onToggleFacingMode: () => void;
}

function getOvalTone(
  isRunning: boolean,
  status: MeasurementStatus,
  quality: SignalQualityLevel,
  faceDetected: boolean
): 'idle' | 'searching' | 'warming' | 'locked' | 'low-signal' {
  if (!isRunning || status === 'idle' || status === 'initializing') {
    return 'idle';
  }
  if (!faceDetected || status === 'no-face') {
    return 'searching';
  }
  if (status === 'warming-up') {
    return 'warming';
  }
  if (status === 'low-signal' || quality === 'poor') {
    return 'low-signal';
  }
  if (status === 'measuring' && (quality === 'good' || quality === 'excellent')) {
    return 'locked';
  }
  return 'warming';
}

function getTrackingBadge(
  isRunning: boolean,
  isStarting: boolean,
  isSimulationMode: boolean,
  trackingMode: TrackingMode,
  faceDetected: boolean
): { label: string; dotClass: string } {
  if (isStarting) {
    return { label: 'Starting Camera…', dotClass: 'warming' };
  }
  if (!isRunning) {
    return { label: 'Camera Standby', dotClass: '' };
  }
  if (isSimulationMode || trackingMode === 'simulation') {
    return { label: 'Demo Pulse', dotClass: 'sim' };
  }
  if (!faceDetected) {
    return { label: 'Searching Face', dotClass: 'warming' };
  }
  if (trackingMode === 'mediapipe') {
    return { label: 'Face Tracked', dotClass: 'active' };
  }
  return { label: 'Skin Mode', dotClass: 'active' };
}

export function CameraViewfinder({
  videoRef,
  simCanvasRef,
  isRunning,
  isStarting,
  isSimulationMode,
  facingMode,
  hasMultipleCameras,
  faceDetected,
  roiBoxes,
  trackingMode,
  status,
  quality,
  guidance,
  cameraError,
  onStart,
  onStartSimulation,
  onToggleFacingMode,
}: CameraViewfinderProps) {
  const isMirrored = facingMode === 'user' && !isSimulationMode;
  const ovalTone = getOvalTone(isRunning, status, quality, faceDetected);
  const badge = getTrackingBadge(
    isRunning,
    isStarting,
    isSimulationMode,
    trackingMode,
    faceDetected
  );

  // Prefer specific forehead/cheek ROIs when available; fall back to face ROI otherwise
  const specificRois = roiBoxes.filter((box) => box.label !== 'face');
  const displayRois = specificRois.length > 0 ? specificRois : roiBoxes;
  const showRois = isRunning && faceDetected && displayRois.length > 0;

  return (
    <div className="viewfinder-card" aria-label="Camera heart rate viewfinder">
      {/* Live hardware camera feed */}
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        className={`viewfinder-media${isMirrored ? ' mirrored' : ''}${
          isSimulationMode || !isRunning ? ' hidden' : ''
        }`}
      />

      {/* Synthetic simulation canvas for camera-less demo testing */}
      <canvas
        ref={simCanvasRef}
        className={`viewfinder-media${isSimulationMode && isRunning ? '' : ' hidden'}`}
        aria-label="Simulated facial pulse feed"
      />

      {/* Soft cinematic vignette */}
      <div className="viewfinder-vignette" aria-hidden="true" />

      {/* Subtle forehead / cheek ROI indicators in normalized frame coordinates */}
      {showRois && (
        <svg
          className="viewfinder-svg"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          {displayRois.map((roi, idx) => {
            const boxClass = `roi-box${isSimulationMode ? ' simulation' : ''}`;
            if (roi.polygon && roi.polygon.length >= 3) {
              const points = roi.polygon
                .map((pt) => {
                  const x = (isMirrored ? 1 - pt.x : pt.x) * 100;
                  const y = pt.y * 100;
                  return `${x.toFixed(2)},${y.toFixed(2)}`;
                })
                .join(' ');
              return <polygon key={`${roi.label}-${idx}`} points={points} className={boxClass} />;
            }

            const x = (isMirrored ? 1 - roi.x - roi.width : roi.x) * 100;
            const y = roi.y * 100;
            const w = roi.width * 100;
            const h = roi.height * 100;

            return (
              <rect
                key={`${roi.label}-${idx}`}
                x={x}
                y={y}
                width={w}
                height={h}
                rx={1.5}
                ry={1.5}
                className={boxClass}
              />
            );
          })}
        </svg>
      )}

      {/* Face alignment oval overlay (aspect-preserved) */}
      <svg
        className="viewfinder-svg"
        viewBox="0 0 400 300"
        preserveAspectRatio="xMidYMid meet"
        aria-hidden="true"
      >
        <ellipse
          cx="200"
          cy="148"
          rx="86"
          ry="112"
          fill="none"
          strokeWidth="2.2"
          strokeDasharray={ovalTone === 'locked' ? undefined : '8 7'}
          strokeLinecap="round"
          className={`oval-guide ${ovalTone}`}
        />
        {/* Subtle top/bottom alignment ticks */}
        <line
          x1="200"
          y1="26"
          x2="200"
          y2="34"
          strokeWidth="2"
          strokeLinecap="round"
          className={`oval-guide ${ovalTone}`}
          opacity="0.7"
        />
        <line
          x1="200"
          y1="262"
          x2="200"
          y2="270"
          strokeWidth="2"
          strokeLinecap="round"
          className={`oval-guide ${ovalTone}`}
          opacity="0.7"
        />
      </svg>

      {/* Top floating status & camera switch controls */}
      <div className="viewfinder-top-bar">
        <div className="tracking-pill" role="status" aria-live="polite">
          <span className={`tracking-dot ${badge.dotClass}`} aria-hidden="true" />
          <span>{badge.label}</span>
        </div>

        {hasMultipleCameras && !isSimulationMode && (
          <button
            type="button"
            className="camera-flip-btn"
            onClick={onToggleFacingMode}
            disabled={isStarting}
            aria-label={
              facingMode === 'user' ? 'Switch to back camera' : 'Switch to front camera'
            }
            title={facingMode === 'user' ? 'Switch to back camera' : 'Switch to front camera'}
          >
            <SwitchCamera size={18} />
          </button>
        )}
      </div>

      {/* Live coaching pill inside viewfinder while running */}
      {isRunning && !cameraError && guidance && (
        <div className="viewfinder-coaching-pill" aria-live="polite">
          {guidance}
        </div>
      )}

      {/* Idle Overlay */}
      {!isRunning && !isStarting && !cameraError && (
        <div className="viewfinder-overlay">
          <div className="overlay-icon-ring" aria-hidden="true">
            <Camera size={22} />
          </div>
          <h2 className="overlay-title">Position your face in the oval</h2>
          <p className="overlay-desc">
            Stay still in soft, even lighting. Video frames are analyzed locally and never leave
            your device.
          </p>
          <div className="overlay-actions">
            <button type="button" className="btn-primary compact" onClick={onStart}>
              <Play size={16} fill="currentColor" />
              <span>Start Heart Rate Scan</span>
            </button>
          </div>
          <button type="button" className="link-subtle" onClick={onStartSimulation}>
            <Sparkles size={14} />
            <span>Test with Demo Pulse</span>
          </button>
        </div>
      )}

      {/* Initializing Overlay */}
      {isStarting && !cameraError && (
        <div className="viewfinder-overlay">
          <div className="overlay-icon-ring" aria-hidden="true">
            <Loader2 size={22} className="spin" />
          </div>
          <h2 className="overlay-title">Opening camera…</h2>
          <p className="overlay-desc">Allow camera access when prompted to start scanning.</p>
        </div>
      )}

      {/* Camera Error Overlay */}
      {cameraError && (
        <div className="viewfinder-overlay error-state" role="alert">
          <div className="overlay-icon-ring error" aria-hidden="true">
            {cameraError.code === 'insecure-context' ? (
              <ShieldAlert size={22} />
            ) : (
              <AlertTriangle size={22} />
            )}
          </div>
          <h2 className="overlay-title">{cameraError.title}</h2>
          <p className="overlay-desc">{cameraError.message}</p>
          {cameraError.recoveryHint && (
            <p className="overlay-hint">{cameraError.recoveryHint}</p>
          )}
          <div className="overlay-actions">
            <button type="button" className="btn-primary compact" onClick={onStart}>
              <RefreshCw size={15} />
              <span>Retry Camera</span>
            </button>
            <button type="button" className="btn-secondary compact" onClick={onStartSimulation}>
              <Sparkles size={15} />
              <span>Demo Pulse</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
