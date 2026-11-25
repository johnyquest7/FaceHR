import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import { RppgProcessor, createIdleEstimate } from '../rppg';
import type {
  FrameRoiExtraction,
  HeartRateEstimate,
  HeartRateMonitorState,
  MeasurementStatus,
  RoiBox,
  TrackingMode,
} from '../types/rppg';
import { FaceRoiTracker } from './faceRoiTracker';
import { SyntheticPulseGenerator } from './syntheticFeed';
import { useCamera } from './useCamera';

const UI_UPDATE_INTERVAL_MS = 125; // ~8 Hz UI refresh
const SIM_FRAME_INTERVAL_MS = 33.33; // ~30 Hz synthetic feed
const RAF_CAMERA_MIN_INTERVAL_MS = 31; // ~30 Hz cap for rAF fallback
const RVFC_CAMERA_MIN_INTERVAL_MS = 24; // cap high-Hz cameras (~30-40 Hz max)
const DEFAULT_SIM_BPM = 72;

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

function nowMs(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

export function useHeartRateMonitor(
  videoRef: RefObject<HTMLVideoElement | null>,
  simCanvasRef: RefObject<HTMLCanvasElement | null>
): HeartRateMonitorState {
  const camera = useCamera(videoRef);

  const [estimate, setEstimate] = useState<HeartRateEstimate>(() => createIdleEstimate());
  const [faceBox, setFaceBox] = useState<RoiBox | null>(null);
  const [roiBoxes, setRoiBoxes] = useState<RoiBox[]>([]);
  const [faceDetected, setFaceDetected] = useState(false);
  const [trackingMode, setTrackingMode] = useState<TrackingMode>('skin-fallback');
  const [isSimulationMode, setIsSimulationMode] = useState(false);
  const [simulationBpm, setSimulationBpmState] = useState(DEFAULT_SIM_BPM);

  const processorRef = useRef<RppgProcessor | null>(null);
  const trackerRef = useRef<FaceRoiTracker | null>(null);
  const syntheticGenRef = useRef<SyntheticPulseGenerator | null>(null);

  const loopSessionIdRef = useRef(0);
  const rafIdRef = useRef<number | null>(null);
  const rvfcIdRef = useRef<number | null>(null);
  const rvfcVideoRef = useRef<HTMLVideoElement | null>(null);

  const lastProcessedMsRef = useRef(0);
  const lastUiPushMsRef = useRef(0);
  const lastReportedStatusRef = useRef<MeasurementStatus>('idle');
  const lastReportedFaceRef = useRef(false);
  const lastReportedModeRef = useRef<TrackingMode>('skin-fallback');

  const isSimulationRef = useRef(false);
  const isRunningCameraRef = useRef(false);
  const simulationBpmRef = useRef(DEFAULT_SIM_BPM);
  const unmountedRef = useRef(false);

  const getProcessor = useCallback((): RppgProcessor => {
    if (!processorRef.current) {
      processorRef.current = new RppgProcessor();
    }
    return processorRef.current;
  }, []);

  const getTracker = useCallback((): FaceRoiTracker => {
    if (!trackerRef.current) {
      trackerRef.current = new FaceRoiTracker();
    }
    return trackerRef.current;
  }, []);

  const getSyntheticGen = useCallback((): SyntheticPulseGenerator => {
    if (!syntheticGenRef.current) {
      syntheticGenRef.current = new SyntheticPulseGenerator({
        targetBpm: simulationBpmRef.current,
      });
    }
    return syntheticGenRef.current;
  }, []);

  const cancelFrameLoop = useCallback(() => {
    loopSessionIdRef.current += 1;

    if (rafIdRef.current !== null && typeof cancelAnimationFrame === 'function') {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }

    if (
      rvfcIdRef.current !== null &&
      rvfcVideoRef.current &&
      typeof rvfcVideoRef.current.cancelVideoFrameCallback === 'function'
    ) {
      try {
        rvfcVideoRef.current.cancelVideoFrameCallback(rvfcIdRef.current);
      } catch {
        // Ignore cancel errors if video element detached
      }
      rvfcIdRef.current = null;
      rvfcVideoRef.current = null;
    }
  }, []);

  const pushExtractionToPipeline = useCallback(
    (extraction: FrameRoiExtraction, timestampMs: number) => {
      const nextEstimate = getProcessor().pushSample(extraction);

      const statusChanged = nextEstimate.status !== lastReportedStatusRef.current;
      const faceChanged = extraction.faceDetected !== lastReportedFaceRef.current;
      const modeChanged = extraction.trackingMode !== lastReportedModeRef.current;
      const dueForUiUpdate = timestampMs - lastUiPushMsRef.current >= UI_UPDATE_INTERVAL_MS;

      if (!unmountedRef.current && (dueForUiUpdate || statusChanged || faceChanged || modeChanged)) {
        lastUiPushMsRef.current = timestampMs;
        lastReportedStatusRef.current = nextEstimate.status;
        lastReportedFaceRef.current = extraction.faceDetected;
        lastReportedModeRef.current = extraction.trackingMode;

        setEstimate(nextEstimate);
        setFaceBox(extraction.faceBox);
        setRoiBoxes(extraction.roiBoxes);
        setFaceDetected(extraction.faceDetected);
        setTrackingMode(extraction.trackingMode);
      }
    },
    [getProcessor]
  );

  const startCameraFrameLoop = useCallback(() => {
    cancelFrameLoop();
    const sessionId = loopSessionIdRef.current;
    lastProcessedMsRef.current = 0;
    lastUiPushMsRef.current = 0;

    const video = videoRef.current;
    const tracker = getTracker();

    const processCurrentVideoFrame = (ts: number) => {
      const currentVideo = videoRef.current;
      if (!currentVideo || currentVideo.readyState < 2 || currentVideo.videoWidth <= 0) {
        return;
      }
      const extraction = tracker.extractFrame(currentVideo, ts);
      pushExtractionToPipeline(extraction, ts);
    };

    if (video && typeof video.requestVideoFrameCallback === 'function') {
      const onVideoFrame = (now: number) => {
        if (unmountedRef.current || loopSessionIdRef.current !== sessionId) return;

        const ts = Number.isFinite(now) && now > 0 ? now : nowMs();
        if (ts - lastProcessedMsRef.current >= RVFC_CAMERA_MIN_INTERVAL_MS) {
          lastProcessedMsRef.current = ts;
          processCurrentVideoFrame(ts);
        }

        const currentVideo = videoRef.current;
        if (
          !unmountedRef.current &&
          loopSessionIdRef.current === sessionId &&
          currentVideo &&
          typeof currentVideo.requestVideoFrameCallback === 'function'
        ) {
          rvfcVideoRef.current = currentVideo;
          rvfcIdRef.current = currentVideo.requestVideoFrameCallback(onVideoFrame);
        }
      };

      rvfcVideoRef.current = video;
      rvfcIdRef.current = video.requestVideoFrameCallback(onVideoFrame);
      return;
    }

    // Fallback requestAnimationFrame loop (~30 Hz cap) for browsers without rVFC
    let lastVideoTime = -1;
    const onAnimationFrame = (now: number) => {
      if (unmountedRef.current || loopSessionIdRef.current !== sessionId) return;

      const currentVideo = videoRef.current;
      if (
        currentVideo &&
        currentVideo.readyState >= 2 &&
        now - lastProcessedMsRef.current >= RAF_CAMERA_MIN_INTERVAL_MS
      ) {
        if (currentVideo.currentTime !== lastVideoTime || now - lastProcessedMsRef.current >= 45) {
          lastVideoTime = currentVideo.currentTime;
          lastProcessedMsRef.current = now;
          processCurrentVideoFrame(now);
        }
      }

      if (!unmountedRef.current && loopSessionIdRef.current === sessionId) {
        rafIdRef.current = requestAnimationFrame(onAnimationFrame);
      }
    };

    rafIdRef.current = requestAnimationFrame(onAnimationFrame);
  }, [cancelFrameLoop, getTracker, pushExtractionToPipeline, videoRef]);

  const startSimulationFrameLoop = useCallback(() => {
    cancelFrameLoop();
    const sessionId = loopSessionIdRef.current;
    lastProcessedMsRef.current = 0;
    lastUiPushMsRef.current = 0;

    const generator = getSyntheticGen();

    const onSimFrame = (now: number) => {
      if (unmountedRef.current || loopSessionIdRef.current !== sessionId) return;

      if (lastProcessedMsRef.current === 0 || now - lastProcessedMsRef.current >= SIM_FRAME_INTERVAL_MS - 2) {
        lastProcessedMsRef.current = now;
        const canvas = simCanvasRef.current;
        const extraction = canvas
          ? generator.renderToCanvas(canvas, now)
          : generator.generateFrame(now);
        pushExtractionToPipeline(extraction, now);
      }

      if (!unmountedRef.current && loopSessionIdRef.current === sessionId) {
        rafIdRef.current = requestAnimationFrame(onSimFrame);
      }
    };

    rafIdRef.current = requestAnimationFrame(onSimFrame);
  }, [cancelFrameLoop, getSyntheticGen, pushExtractionToPipeline, simCanvasRef]);

  const stop = useCallback(() => {
    cancelFrameLoop();
    camera.stop();
    isSimulationRef.current = false;
    isRunningCameraRef.current = false;

    processorRef.current?.reset();
    trackerRef.current?.reset();
    syntheticGenRef.current?.reset();

    const idle = createIdleEstimate();
    lastReportedStatusRef.current = idle.status;
    lastReportedFaceRef.current = false;

    if (!unmountedRef.current) {
      setIsSimulationMode(false);
      setEstimate(idle);
      setFaceBox(null);
      setRoiBoxes([]);
      setFaceDetected(false);
    }
  }, [camera, cancelFrameLoop]);

  const start = useCallback(async () => {
    cancelFrameLoop();
    isSimulationRef.current = false;
    isRunningCameraRef.current = false;
    if (!unmountedRef.current) {
      setIsSimulationMode(false);
    }

    getProcessor().reset();
    getTracker().reset();

    const initEstimate: HeartRateEstimate = {
      ...createIdleEstimate(),
      status: 'initializing',
      guidance: 'Starting camera and face tracker…',
      updatedAtMs: nowMs(),
    };
    lastReportedStatusRef.current = 'initializing';
    lastReportedFaceRef.current = false;
    if (!unmountedRef.current) {
      setEstimate(initEstimate);
      setFaceBox(null);
      setRoiBoxes([]);
      setFaceDetected(false);
    }

    // Begin lazy MediaPipe WASM/model load concurrently while requesting camera permission
    const tracker = getTracker();
    void tracker.init().then((resolvedMode) => {
      if (!unmountedRef.current && isRunningCameraRef.current) {
        lastReportedModeRef.current = resolvedMode;
        setTrackingMode(resolvedMode);
      }
    });

    const started = await camera.start();
    if (unmountedRef.current) return;

    if (!started) {
      isRunningCameraRef.current = false;
      const errEstimate: HeartRateEstimate = {
        ...createIdleEstimate(),
        status: 'error',
        guidance: 'Camera unavailable. Check permissions or try Demo Pulse mode.',
        updatedAtMs: nowMs(),
      };
      lastReportedStatusRef.current = 'error';
      setEstimate(errEstimate);
      return;
    }

    isRunningCameraRef.current = true;
    const currentMode = tracker.getTrackingMode();
    lastReportedModeRef.current = currentMode;
    setTrackingMode(currentMode);

    startCameraFrameLoop();
  }, [camera, cancelFrameLoop, getProcessor, getTracker, startCameraFrameLoop]);

  const reset = useCallback(() => {
    getProcessor().reset();
    trackerRef.current?.reset();
    syntheticGenRef.current?.reset();

    const active = isSimulationRef.current || isRunningCameraRef.current;
    const nextEstimate: HeartRateEstimate = active
      ? {
          ...createIdleEstimate(),
          status: 'warming-up',
          guidance: 'Measurement reset — hold still and face the light.',
          updatedAtMs: nowMs(),
        }
      : createIdleEstimate();

    lastReportedStatusRef.current = nextEstimate.status;
    lastUiPushMsRef.current = 0;

    if (!unmountedRef.current) {
      setEstimate(nextEstimate);
      if (!active) {
        setFaceBox(null);
        setRoiBoxes([]);
        setFaceDetected(false);
      }
    }
  }, [getProcessor]);

  const toggleFacingMode = useCallback(async () => {
    const wasRunningCamera = isRunningCameraRef.current;
    if (wasRunningCamera) {
      cancelFrameLoop();
    }

    const ok = await camera.toggleFacingMode();
    if (unmountedRef.current) return;

    if (wasRunningCamera && ok) {
      getProcessor().reset();
      getTracker().reset();
      startCameraFrameLoop();
    } else if (wasRunningCamera && !ok) {
      isRunningCameraRef.current = false;
    }
  }, [camera, cancelFrameLoop, getProcessor, getTracker, startCameraFrameLoop]);

  const startSimulation = useCallback(
    (targetBpm?: number) => {
      cancelFrameLoop();
      camera.stop();
      camera.clearError();
      isRunningCameraRef.current = false;

      const clampedBpm = clamp(
        targetBpm !== undefined && Number.isFinite(targetBpm)
          ? targetBpm
          : simulationBpmRef.current,
        40,
        200
      );
      simulationBpmRef.current = clampedBpm;

      getProcessor().reset();
      const generator = getSyntheticGen();
      generator.reset();
      generator.setTargetBpm(clampedBpm);

      isSimulationRef.current = true;
      lastReportedModeRef.current = 'simulation';
      lastReportedStatusRef.current = 'warming-up';
      lastReportedFaceRef.current = true;

      if (!unmountedRef.current) {
        setSimulationBpmState(clampedBpm);
        setIsSimulationMode(true);
        setTrackingMode('simulation');
        setEstimate({
          ...createIdleEstimate(),
          status: 'warming-up',
          guidance: `Demo Pulse active (${Math.round(clampedBpm)} BPM) — locking onto synthetic wave…`,
          updatedAtMs: nowMs(),
        });
      }

      startSimulationFrameLoop();
    },
    [camera, cancelFrameLoop, getProcessor, getSyntheticGen, startSimulationFrameLoop]
  );

  const setSimulationBpm = useCallback(
    (targetBpm: number) => {
      if (!Number.isFinite(targetBpm)) return;
      const clampedBpm = clamp(targetBpm, 40, 200);
      simulationBpmRef.current = clampedBpm;
      if (!unmountedRef.current) {
        setSimulationBpmState(clampedBpm);
      }
      syntheticGenRef.current?.setTargetBpm(clampedBpm);
    },
    []
  );

  // Clean up frame loop and MediaPipe resources on unmount
  useEffect(() => {
    unmountedRef.current = false;
    return () => {
      unmountedRef.current = true;
      isSimulationRef.current = false;
      isRunningCameraRef.current = false;
      cancelFrameLoop();
      trackerRef.current?.dispose();
      trackerRef.current = null;
    };
  }, [cancelFrameLoop]);

  return {
    estimate,
    faceBox,
    roiBoxes,
    faceDetected,
    trackingMode,
    isRunning: isSimulationMode || camera.isActive,
    isStarting: !isSimulationMode && camera.isStarting,
    facingMode: camera.facingMode,
    hasMultipleCameras: camera.hasMultipleCameras,
    cameraError: isSimulationMode ? null : camera.error,
    isSimulationMode,
    simulationBpm,
    start,
    stop,
    reset,
    toggleFacingMode,
    startSimulation,
    setSimulationBpm,
  };
}
