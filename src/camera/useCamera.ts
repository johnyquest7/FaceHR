import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import type { CameraErrorInfo, FacingMode } from '../types/rppg';

interface WakeLockSentinelLike {
  released: boolean;
  release: () => Promise<void>;
}

interface WakeLockApiLike {
  request?: (type: 'screen') => Promise<WakeLockSentinelLike>;
}

export interface UseCameraResult {
  isActive: boolean;
  isStarting: boolean;
  facingMode: FacingMode;
  hasMultipleCameras: boolean;
  isExposureLocked: boolean;
  error: CameraErrorInfo | null;
  start: (overrideFacingMode?: FacingMode) => Promise<boolean>;
  stop: () => void;
  toggleFacingMode: () => Promise<boolean>;
  clearError: () => void;
}

const AE_AWB_SETTLE_DELAY_MS = 1500;

interface ExtendedTrackCapabilities extends MediaTrackCapabilities {
  exposureMode?: string[];
  exposureTime?: { min?: number; max?: number };
  iso?: { min?: number; max?: number };
  whiteBalanceMode?: string[];
  colorTemperature?: { min?: number; max?: number };
  focusMode?: string[];
  focusDistance?: { min?: number; max?: number };
}

interface ExtendedTrackSettings extends MediaTrackSettings {
  exposureMode?: string;
  exposureTime?: number;
  iso?: number;
  whiteBalanceMode?: string;
  colorTemperature?: number;
  focusMode?: string;
  focusDistance?: number;
}

function clampToCapability(
  value: number | undefined,
  range: { min?: number; max?: number } | undefined
): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
  let v = value;
  if (range?.min !== undefined && v < range.min) v = range.min;
  if (range?.max !== undefined && v > range.max) v = range.max;
  return v;
}

/**
 * Attempts to lock camera Auto-Exposure (AE), Auto-White-Balance (AWB), and Auto-Focus
 * to their settled values via WebRTC MediaTrackConstraints.
 * Prevents smartphones from dynamically shifting exposure/gain mid-scan, which otherwise
 * injects 0.5–1.5 Hz artifacts into the facial rPPG band.
 * Gracefully no-ops on browsers (such as iOS Safari) that do not expose manual track locks.
 */
async function tryLockCameraAutoControls(stream: MediaStream): Promise<boolean> {
  const videoTrack = stream.getVideoTracks()[0];
  if (!videoTrack || videoTrack.readyState !== 'live') return false;
  if (
    typeof videoTrack.getCapabilities !== 'function' ||
    typeof videoTrack.applyConstraints !== 'function'
  ) {
    return false;
  }

  let caps: ExtendedTrackCapabilities;
  let settings: ExtendedTrackSettings;
  try {
    caps = videoTrack.getCapabilities() as ExtendedTrackCapabilities;
    settings =
      typeof videoTrack.getSettings === 'function'
        ? (videoTrack.getSettings() as ExtendedTrackSettings)
        : {};
  } catch {
    return false;
  }

  const aeConstraint: Record<string, unknown> = {};
  if (Array.isArray(caps.exposureMode) && caps.exposureMode.includes('manual')) {
    aeConstraint.exposureMode = 'manual';
    const expTime = clampToCapability(settings.exposureTime, caps.exposureTime);
    if (expTime !== undefined) aeConstraint.exposureTime = expTime;
    const iso = clampToCapability(settings.iso, caps.iso);
    if (iso !== undefined) aeConstraint.iso = iso;
  }

  const awbConstraint: Record<string, unknown> = {};
  if (Array.isArray(caps.whiteBalanceMode) && caps.whiteBalanceMode.includes('manual')) {
    awbConstraint.whiteBalanceMode = 'manual';
    const temp = clampToCapability(settings.colorTemperature, caps.colorTemperature);
    if (temp !== undefined) awbConstraint.colorTemperature = temp;
  }

  const focusConstraint: Record<string, unknown> = {};
  if (Array.isArray(caps.focusMode) && caps.focusMode.includes('manual')) {
    focusConstraint.focusMode = 'manual';
    const dist = clampToCapability(settings.focusDistance, caps.focusDistance);
    if (dist !== undefined) focusConstraint.focusDistance = dist;
  }

  const combined = { ...aeConstraint, ...awbConstraint, ...focusConstraint };
  if (Object.keys(combined).length === 0) {
    return false;
  }

  try {
    await videoTrack.applyConstraints({
      advanced: [combined as MediaTrackConstraintSet],
    });
    return Object.keys(aeConstraint).length > 0 || Object.keys(awbConstraint).length > 0;
  } catch {
    // Some Android drivers reject combined AE+AWB+AF locks; fall back to locking individually
    let lockedAny = false;
    for (const part of [aeConstraint, awbConstraint]) {
      if (Object.keys(part).length === 0) continue;
      try {
        await videoTrack.applyConstraints({
          advanced: [part as MediaTrackConstraintSet],
        });
        lockedAny = true;
      } catch {
        // Ignore unsupported per-driver constraint
      }
    }
    return lockedAny;
  }
}

/**
 * Verifies that the current browser context supports WebRTC camera capture.
 * Returns a CameraErrorInfo if blocked (e.g. non-HTTPS mobile context), or null when supported.
 */
export function checkCameraSupport(): CameraErrorInfo | null {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') {
    return {
      code: 'unsupported',
      title: 'Camera Unavailable',
      message: 'Camera capture requires a browser environment.',
      recoveryHint: 'Open PulseCam in a modern mobile or desktop browser, or use Demo Pulse mode.',
    };
  }

  if (!window.isSecureContext) {
    return {
      code: 'insecure-context',
      title: 'HTTPS Connection Required',
      message:
        'Mobile and desktop browsers block camera access on insecure (HTTP) pages except localhost.',
      recoveryHint:
        'Open PulseCam over HTTPS (use npm run dev:https, a Cloudflare HTTPS tunnel, or the deployed HTTPS URL), or try Demo Pulse mode.',
    };
  }

  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
    return {
      code: 'unsupported',
      title: 'Camera API Not Supported',
      message: 'This browser does not expose camera capture (navigator.mediaDevices.getUserMedia).',
      recoveryHint:
        'Use Safari on iOS, Chrome on Android, or an up-to-date desktop browser. On iOS, in-app browsers may restrict camera access—open in Safari.',
    };
  }

  return null;
}

/**
 * Maps browser getUserMedia exceptions to clear, user-actionable CameraErrorInfo objects.
 */
export function classifyCameraError(err: unknown): CameraErrorInfo {
  const errorName =
    err && typeof err === 'object' && 'name' in err ? String((err as { name: unknown }).name) : '';
  const errorMessage =
    err && typeof err === 'object' && 'message' in err
      ? String((err as { message: unknown }).message)
      : '';

  if (
    errorName === 'NotAllowedError' ||
    errorName === 'PermissionDeniedError' ||
    errorName === 'SecurityError'
  ) {
    return {
      code: 'permission-denied',
      title: 'Camera Permission Denied',
      message: 'PulseCam needs camera access to measure subtle facial color changes on-device.',
      recoveryHint:
        'Allow camera access in your browser address bar or phone Settings → Privacy → Camera, then tap Retry.',
    };
  }

  if (errorName === 'NotFoundError' || errorName === 'DevicesNotFoundError') {
    return {
      code: 'no-camera',
      title: 'No Camera Found',
      message: 'No usable video camera was detected on this device.',
      recoveryHint:
        'Connect a webcam or switch to a device with a camera. You can also test the engine with Demo Pulse mode.',
    };
  }

  if (
    errorName === 'NotReadableError' ||
    errorName === 'TrackStartError' ||
    errorName === 'AbortError'
  ) {
    return {
      code: 'camera-in-use',
      title: 'Camera Busy or Unavailable',
      message: 'Your camera appears to be in use by another app or could not be started.',
      recoveryHint:
        'Close other video/camera apps (Zoom, FaceTime, Camera, other browser tabs) and tap Retry.',
    };
  }

  return {
    code: 'unknown',
    title: 'Unable to Start Camera',
    message: errorMessage || 'An unexpected error occurred while starting the camera.',
    recoveryHint:
      'Refresh the page, confirm camera permissions over HTTPS, or try Demo Pulse mode.',
  };
}

function stopMediaStream(stream: MediaStream | null): void {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    try {
      track.stop();
    } catch {
      // Ignore already-ended track cleanup errors
    }
  }
}

async function requestStreamWithFallback(facingMode: FacingMode): Promise<MediaStream> {
  const mediaDevices = navigator.mediaDevices;

  // Primary mobile-safe 640x480 @ 30fps constraints (prefer >= 20fps so phones avoid 15fps binning)
  const primaryConstraints: MediaStreamConstraints = {
    video: {
      facingMode: { ideal: facingMode },
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30, min: 20 },
    },
    audio: false,
  };

  try {
    return await mediaDevices.getUserMedia(primaryConstraints);
  } catch (err) {
    const name =
      err && typeof err === 'object' && 'name' in err ? String((err as { name: unknown }).name) : '';
    if (name !== 'OverconstrainedError' && name !== 'ConstraintNotSatisfiedError') {
      throw err;
    }

    // Relax frameRate/resolution constraints on older cameras or virtual webcams
    try {
      return await mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30, min: 15 },
        },
        audio: false,
      });
    } catch {
      try {
        return await mediaDevices.getUserMedia({
          video: { facingMode },
          audio: false,
        });
      } catch {
        return await mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      }
    }
  }
}

export function useCamera(videoRef: RefObject<HTMLVideoElement | null>): UseCameraResult {
  const [isActive, setIsActive] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [facingMode, setFacingMode] = useState<FacingMode>('user');
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  const [isExposureLocked, setIsExposureLocked] = useState(false);
  const [error, setError] = useState<CameraErrorInfo | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const wakeLockRef = useRef<WakeLockSentinelLike | null>(null);
  const aeLockTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const facingModeRef = useRef<FacingMode>('user');
  const isActiveRef = useRef(false);
  const startRequestIdRef = useRef(0);
  const unmountedRef = useRef(false);

  const refreshVideoDevices = useCallback(async () => {
    if (
      typeof navigator === 'undefined' ||
      !navigator.mediaDevices ||
      typeof navigator.mediaDevices.enumerateDevices !== 'function'
    ) {
      return;
    }
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      if (unmountedRef.current) return;
      const videoInputs = devices.filter((d) => d.kind === 'videoinput');
      setHasMultipleCameras(videoInputs.length > 1);
    } catch {
      // Ignore device enumeration errors on restricted browsers
    }
  }, []);

  const releaseWakeLock = useCallback(async () => {
    const sentinel = wakeLockRef.current;
    wakeLockRef.current = null;
    if (sentinel && !sentinel.released) {
      try {
        await sentinel.release();
      } catch {
        // Ignore wake lock release errors
      }
    }
  }, []);

  const acquireWakeLock = useCallback(async () => {
    if (typeof navigator === 'undefined' || typeof document === 'undefined') return;
    if (document.visibilityState !== 'visible') return;
    const wakeLock = (navigator as unknown as { wakeLock?: WakeLockApiLike }).wakeLock;
    if (!wakeLock || typeof wakeLock.request !== 'function') return;

    try {
      if (wakeLockRef.current && !wakeLockRef.current.released) {
        return;
      }
      const sentinel = await wakeLock.request('screen');
      if (unmountedRef.current || !isActiveRef.current) {
        await sentinel.release().catch(() => undefined);
        return;
      }
      wakeLockRef.current = sentinel;
    } catch {
      // Screen Wake Lock may fail on low battery or unsupported contexts; non-fatal
    }
  }, []);

  const clearAeLockTimer = useCallback(() => {
    if (aeLockTimerRef.current !== null) {
      clearTimeout(aeLockTimerRef.current);
      aeLockTimerRef.current = null;
    }
  }, []);

  const teardownStream = useCallback(() => {
    clearAeLockTimer();
    stopMediaStream(streamRef.current);
    streamRef.current = null;

    const video = videoRef.current;
    if (video) {
      try {
        video.pause();
      } catch {
        // Ignore pause errors
      }
      video.srcObject = null;
    }
  }, [clearAeLockTimer, videoRef]);

  const stop = useCallback(() => {
    startRequestIdRef.current += 1;
    isActiveRef.current = false;
    teardownStream();
    void releaseWakeLock();
    if (!unmountedRef.current) {
      setIsActive(false);
      setIsStarting(false);
      setIsExposureLocked(false);
    }
  }, [releaseWakeLock, teardownStream]);

  const start = useCallback(
    async (overrideFacingMode?: FacingMode): Promise<boolean> => {
      const targetFacing = overrideFacingMode ?? facingModeRef.current;
      facingModeRef.current = targetFacing;
      if (!unmountedRef.current) {
        setFacingMode(targetFacing);
      }

      const supportError = checkCameraSupport();
      if (supportError) {
        stop();
        if (!unmountedRef.current) {
          setError(supportError);
        }
        return false;
      }

      const requestId = startRequestIdRef.current + 1;
      startRequestIdRef.current = requestId;

      teardownStream();
      if (!unmountedRef.current) {
        setError(null);
        setIsStarting(true);
        setIsActive(false);
      }
      isActiveRef.current = false;

      let stream: MediaStream | null = null;
      try {
        stream = await requestStreamWithFallback(targetFacing);

        // Guard against stale async completion (stop() or newer start() or unmount)
        if (unmountedRef.current || startRequestIdRef.current !== requestId) {
          stopMediaStream(stream);
          return false;
        }

        const video = videoRef.current;
        if (video) {
          video.playsInline = true;
          video.muted = true;
          video.autoplay = true;
          video.setAttribute('playsinline', 'true');
          video.setAttribute('muted', 'true');
          video.srcObject = stream;

          if (video.readyState < 1) {
            await new Promise<void>((resolve) => {
              let settled = false;
              const finish = () => {
                if (settled) return;
                settled = true;
                video.removeEventListener('loadedmetadata', finish);
                clearTimeout(timeoutId);
                resolve();
              };
              const timeoutId = setTimeout(finish, 2500);
              video.addEventListener('loadedmetadata', finish, { once: true });
            });
          }

          if (unmountedRef.current || startRequestIdRef.current !== requestId) {
            stopMediaStream(stream);
            video.srcObject = null;
            return false;
          }

          try {
            await video.play();
          } catch {
            // Some mobile browsers require a brief retry once metadata is settled
          }
        }

        streamRef.current = stream;
        isActiveRef.current = true;

        if (!unmountedRef.current) {
          setIsActive(true);
          setIsStarting(false);
          setIsExposureLocked(false);
          setError(null);
        }

        // Allow mobile AE/AWB ~1.5s to converge on the user's face, then lock manual exposure/WB
        // when supported so sensor auto-gain stops corrupting the 0.7–3.5 Hz pulse band.
        const lockedStream = stream;
        clearAeLockTimer();
        aeLockTimerRef.current = setTimeout(() => {
          aeLockTimerRef.current = null;
          if (
            unmountedRef.current ||
            startRequestIdRef.current !== requestId ||
            streamRef.current !== lockedStream
          ) {
            return;
          }
          void tryLockCameraAutoControls(lockedStream).then((locked) => {
            if (
              !unmountedRef.current &&
              startRequestIdRef.current === requestId &&
              streamRef.current === lockedStream
            ) {
              setIsExposureLocked(locked);
            }
          });
        }, AE_AWB_SETTLE_DELAY_MS);

        void acquireWakeLock();
        void refreshVideoDevices();
        return true;
      } catch (err) {
        stopMediaStream(stream);
        if (unmountedRef.current || startRequestIdRef.current !== requestId) {
          return false;
        }
        isActiveRef.current = false;
        setIsActive(false);
        setIsStarting(false);
        setError(classifyCameraError(err));
        return false;
      }
    },
    [acquireWakeLock, clearAeLockTimer, refreshVideoDevices, stop, teardownStream, videoRef]
  );

  const toggleFacingMode = useCallback(async (): Promise<boolean> => {
    const nextFacing: FacingMode = facingModeRef.current === 'user' ? 'environment' : 'user';
    facingModeRef.current = nextFacing;
    setFacingMode(nextFacing);

    if (isActiveRef.current) {
      return start(nextFacing);
    }
    return true;
  }, [start]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Re-acquire Screen Wake Lock when returning to foreground while active
  useEffect(() => {
    if (typeof document === 'undefined') return;
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible' && isActiveRef.current) {
        void acquireWakeLock();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [acquireWakeLock]);

  // Initial device count + live updates when cameras are plugged/unplugged
  useEffect(() => {
    void refreshVideoDevices();

    if (
      typeof navigator === 'undefined' ||
      !navigator.mediaDevices ||
      typeof navigator.mediaDevices.addEventListener !== 'function'
    ) {
      return;
    }

    const handleDeviceChange = () => {
      void refreshVideoDevices();
    };

    navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
    };
  }, [refreshVideoDevices]);

  // Ensure complete stream + wake lock teardown on unmount
  useEffect(() => {
    unmountedRef.current = false;
    return () => {
      unmountedRef.current = true;
      startRequestIdRef.current += 1;
      isActiveRef.current = false;
      teardownStream();
      void releaseWakeLock();
    };
  }, [releaseWakeLock, teardownStream]);

  return {
    isActive,
    isStarting,
    facingMode,
    hasMultipleCameras,
    isExposureLocked,
    error,
    start,
    stop,
    toggleFacingMode,
    clearError,
  };
}
