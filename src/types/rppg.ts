/**
 * Shared contracts for PulseCam (on-device facial rPPG heart rate monitor).
 * All modules (DSP, camera/vision, and UI) adhere strictly to these types.
 */

export type SignalQualityLevel = 'poor' | 'fair' | 'good' | 'excellent';

export type MeasurementStatus =
  | 'idle'
  | 'initializing'
  | 'no-face'
  | 'warming-up'
  | 'measuring'
  | 'low-signal'
  | 'error';

export type TrackingMode = 'mediapipe' | 'skin-fallback' | 'simulation';

export type FacingMode = 'user' | 'environment';

export interface RgbSample {
  /** Mean red channel intensity (0..255) */
  r: number;
  /** Mean green channel intensity (0..255) */
  g: number;
  /** Mean blue channel intensity (0..255) */
  b: number;
  /** Number of valid skin pixels averaged */
  pixelCount: number;
}

export interface NormalizedPoint {
  /** Normalized X coordinate (0..1) in video space */
  x: number;
  /** Normalized Y coordinate (0..1) in video space */
  y: number;
}

export interface RoiBox {
  /** Normalized coordinates (0..1) relative to video frame */
  x: number;
  y: number;
  width: number;
  height: number;
  /** Optional polygon outline in normalized coordinates */
  polygon?: NormalizedPoint[];
  /** Human-readable region label */
  label: 'forehead' | 'left-cheek' | 'right-cheek' | 'face';
}

export interface FrameRoiExtraction {
  /** Monotonic timestamp in milliseconds (e.g. performance.now()) */
  timestampMs: number;
  /** Whether a usable face/skin ROI was detected in this frame */
  faceDetected: boolean;
  /** Weighted fused RGB sample across active ROIs */
  fusedRgb: RgbSample | null;
  /** Per-region RGB samples when available */
  regions: {
    forehead?: RgbSample;
    leftCheek?: RgbSample;
    rightCheek?: RgbSample;
  };
  /** Normalized face bounding box (0..1) for overlay alignment */
  faceBox: RoiBox | null;
  /** Individual ROI overlays for optional subtle guide display */
  roiBoxes: RoiBox[];
  /** Relative frame-to-frame face movement (0 = completely still, >= 1 = high motion) */
  motionScore: number;
  /** Mean facial luminance (0..255); ideal range ~65..210 */
  brightness: number;
  /** Fraction of ROI pixels clipped near black or white (0..1) */
  saturationFraction: number;
  /** Active tracker backend */
  trackingMode: TrackingMode;
}

export interface QualityDiagnostics {
  /** Spectral signal-to-noise ratio in dB */
  snrDb: number;
  /** POS vs CHROM BPM difference in BPM (null if unavailable) */
  algorithmAgreementBpm: number | null;
  /** Motion stability score 0..100 (100 = rock steady) */
  motionStability: number;
  /** Lighting quality score 0..100 (100 = ideal even exposure) */
  lightingScore: number;
  /** Effective measured frame rate (Hz) */
  fps: number;
  /** Buffer fill progress 0..1 (reaches 1.0 at full analysis window) */
  bufferProgress: number;
}

export interface HeartRateEstimate {
  /** Smoothed BPM estimate (null while warming up or when signal is unusable) */
  bpm: number | null;
  /** Raw instantaneous POS BPM estimate before temporal smoothing */
  rawPosBpm: number | null;
  /** Raw instantaneous CHROM BPM estimate before temporal smoothing */
  rawChromBpm: number | null;
  /** Overall confidence score 0..100 */
  confidence: number;
  /** Discrete quality bucket */
  quality: SignalQualityLevel;
  /** Current measurement lifecycle state */
  status: MeasurementStatus;
  /** Concise real-time coaching message for the user */
  guidance: string;
  /** Detailed quality breakdown */
  diagnostics: QualityDiagnostics;
  /** Recent normalized filtered pulse waveform samples (latest ~120 points, -1..1) */
  waveform: number[];
  /** Timestamp of this estimate (ms) */
  updatedAtMs: number;
}

export interface CameraErrorInfo {
  code:
    | 'insecure-context'
    | 'permission-denied'
    | 'no-camera'
    | 'camera-in-use'
    | 'unsupported'
    | 'unknown';
  title: string;
  message: string;
  recoveryHint: string;
}

export interface HeartRateMonitorState {
  /** Current live heart rate & signal quality estimate */
  estimate: HeartRateEstimate;
  /** Latest face/ROI geometry for the camera overlay */
  faceBox: RoiBox | null;
  roiBoxes: RoiBox[];
  faceDetected: boolean;
  trackingMode: TrackingMode;
  /** Camera status */
  isRunning: boolean;
  isStarting: boolean;
  facingMode: FacingMode;
  hasMultipleCameras: boolean;
  cameraError: CameraErrorInfo | null;
  /** Simulation mode for camera-less testing */
  isSimulationMode: boolean;
  simulationBpm: number;
  /** Actions */
  start: () => Promise<void>;
  stop: () => void;
  reset: () => void;
  toggleFacingMode: () => Promise<void>;
  startSimulation: (targetBpm?: number) => void;
  setSimulationBpm: (targetBpm: number) => void;
}
