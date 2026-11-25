import { useRef, useState } from 'react';
import {
  HeartPulse,
  Play,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Square,
} from 'lucide-react';
import { useHeartRateMonitor } from './camera';
import { BpmDisplay } from './components/BpmDisplay';
import { CameraViewfinder } from './components/CameraViewfinder';
import { PwaInstallPrompt } from './components/PwaInstallPrompt';
import { SignalQualityMeter } from './components/SignalQualityMeter';

const DEMO_BPM_PRESETS = [60, 72, 95, 120] as const;

export function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const simCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [iosBannerOpen, setIosBannerOpen] = useState(false);

  const monitor = useHeartRateMonitor(videoRef, simCanvasRef);

  const handleToggleDemo = () => {
    if (monitor.isSimulationMode && monitor.isRunning) {
      monitor.stop();
    } else {
      monitor.startSimulation(monitor.simulationBpm || 72);
    }
  };

  return (
    <div className="app-shell">
      {/* Compact Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <HeartPulse size={20} strokeWidth={2.2} />
          </div>
          <div className="brand-text">
            <h1 className="brand-title">PulseCam</h1>
            <span className="brand-sub">Optical Heart Rate</span>
          </div>
        </div>

        <div className="header-actions">
          <div className="privacy-badge" title="All video and pulse processing runs locally in your browser">
            <ShieldCheck size={13} aria-hidden="true" />
            <span>100% On-Device</span>
          </div>
          <PwaInstallPrompt
            variant="button"
            iosBannerOpen={iosBannerOpen}
            onIosBannerChange={setIosBannerOpen}
          />
        </div>
      </header>

      {/* Dismissible iOS Safari Home Screen installation guide */}
      <PwaInstallPrompt
        variant="ios-banner"
        iosBannerOpen={iosBannerOpen}
        onDismissIosBanner={() => setIosBannerOpen(false)}
      />

      {/* Quick accuracy guidance bar */}
      <div className="outline-tip-bar" role="note">
        <span className="outline-tip-dot" aria-hidden="true" />
        <p className="outline-tip-text">
          Keep the face outline <strong className="outline-tip-green">green</strong> for the best
          results <span className="outline-tip-legend">(amber = warming up/adjusting, red = low signal)</span>.
        </p>
      </div>

      {/* Responsive Dashboard: single-screen vertical on mobile, 2-column on desktop */}
      <main className="dashboard-grid">
        <div className="viewfinder-slot">
          <CameraViewfinder
            videoRef={videoRef}
            simCanvasRef={simCanvasRef}
            isRunning={monitor.isRunning}
            isStarting={monitor.isStarting}
            isSimulationMode={monitor.isSimulationMode}
            facingMode={monitor.facingMode}
            hasMultipleCameras={monitor.hasMultipleCameras}
            faceDetected={monitor.faceDetected}
            roiBoxes={monitor.roiBoxes}
            trackingMode={monitor.trackingMode}
            status={monitor.estimate.status}
            quality={monitor.estimate.quality}
            guidance={monitor.estimate.guidance}
            cameraError={monitor.cameraError}
            onStart={() => void monitor.start()}
            onStartSimulation={() => monitor.startSimulation(monitor.simulationBpm || 72)}
            onToggleFacingMode={() => void monitor.toggleFacingMode()}
          />
        </div>

        <div className="metrics-column">
          <BpmDisplay estimate={monitor.estimate} isRunning={monitor.isRunning} />
          <SignalQualityMeter
            estimate={monitor.estimate}
            faceDetected={monitor.faceDetected}
            isRunning={monitor.isRunning}
          />
        </div>

        <div className="controls-slot">
          <div className="controls-bar">
            {monitor.isRunning ? (
              <button
                type="button"
                className="btn-stop"
                onClick={monitor.stop}
              >
                <Square size={15} fill="currentColor" />
                <span>Stop Scan</span>
              </button>
            ) : (
              <button
                type="button"
                className="btn-primary"
                onClick={() => void monitor.start()}
                disabled={monitor.isStarting}
              >
                <Play size={16} fill="currentColor" />
                <span>{monitor.isStarting ? 'Starting…' : 'Start Scan'}</span>
              </button>
            )}

            <button
              type="button"
              className="btn-secondary"
              onClick={monitor.reset}
              disabled={!monitor.isRunning && monitor.estimate.bpm === null}
              aria-label="Reset reading"
              title="Reset current measurement buffer"
            >
              <RotateCcw size={16} />
              <span>Reset</span>
            </button>

            <button
              type="button"
              className={`btn-secondary${
                monitor.isSimulationMode && monitor.isRunning ? ' active-sim' : ''
              }`}
              onClick={handleToggleDemo}
              aria-pressed={monitor.isSimulationMode && monitor.isRunning}
              title="Test the live rPPG engine with a synthetic optical pulse feed"
            >
              <Sparkles size={15} />
              <span>Demo</span>
            </button>
          </div>

          {/* Demo Pulse target BPM selector when simulation mode is active */}
          {monitor.isSimulationMode && monitor.isRunning && (
            <div className="demo-bar" role="region" aria-label="Demo Pulse target heart rate">
              <span className="demo-bar-label">
                <Sparkles size={13} aria-hidden="true" />
                Demo Pulse Target
              </span>
              <div className="demo-bpm-group" role="group" aria-label="Select simulated BPM">
                {DEMO_BPM_PRESETS.map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    className={`demo-bpm-chip tabular-nums${
                      monitor.simulationBpm === preset ? ' selected' : ''
                    }`}
                    onClick={() => monitor.setSimulationBpm(preset)}
                    aria-pressed={monitor.simulationBpm === preset}
                  >
                    {preset} BPM
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Non-medical wellness disclaimer */}
      <footer className="app-footer">
        <span>For wellness and experimental use only — not a medical device.</span>
      </footer>
    </div>
  );
}

export default App;
