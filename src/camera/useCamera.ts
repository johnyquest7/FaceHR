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
  error: CameraErrorInfo | null;
  start: (overrideFacingMode?: FacingMode) => Promise<boolean>;
  stop: () => void;
  toggleFacingMode: () => Promise<boolean>;
  clearError: () => void;
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

  // Primary mobile-safe 640x480 @ ~30fps constraints
  const primaryConstraints: MediaStreamConstraints = {
    video: {
      facingMode: { ideal: facingMode },
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30, min: 15 },
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

export function useCamera(videoRef: RefObject<HTMLVideoElement | null>): UseCameraResult {
  const [isActive, setIsActive] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [facingMode, setFacingMode] = useState<FacingMode>('user');
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  const [error, setError] = useState<CameraErrorInfo | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const wakeLockRef = useRef<WakeLockSentinelLike | null>(null);
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

  const teardownStream = useCallback(() => {
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
  }, [videoRef]);

  const stop = useCallback(() => {
    startRequestIdRef.current += 1;
    isActiveRef.current = false;
    teardownStream();
    void releaseWakeLock();
    if (!unmountedRef.current) {
      setIsActive(false);
      setIsStarting(false);
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
          setError(null);
        }

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
    [acquireWakeLock, refreshVideoDevices, stop, teardownStream, videoRef]
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
    error,
    start,
    stop,
    toggleFacingMode,
    clearError,
  };
}
