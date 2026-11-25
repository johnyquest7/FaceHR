import { useEffect, useState } from 'react';
import { Download, Share, X } from 'lucide-react';

interface BeforeInstallPromptEvent extends Event {
  readonly platforms: string[];
  readonly userChoice: Promise<{ outcome: 'accepted' | 'dismissed'; platform: string }>;
  prompt: () => Promise<void>;
}

interface NavigatorWithStandalone extends Navigator {
  standalone?: boolean;
}

function isStandaloneMode(): boolean {
  if (typeof window === 'undefined') return false;
  const nav = window.navigator as NavigatorWithStandalone;
  return (
    window.matchMedia('(display-mode: standalone)').matches ||
    nav.standalone === true
  );
}

function detectIosSafari(): boolean {
  if (typeof window === 'undefined') return false;
  const ua = window.navigator.userAgent;
  const isIos =
    /iPad|iPhone|iPod/.test(ua) ||
    (window.navigator.platform === 'MacIntel' && window.navigator.maxTouchPoints > 1);
  const isSafari = /Safari/.test(ua) && !/CriOS|FxiOS|EdgiOS|OPiOS/.test(ua);
  return isIos && isSafari;
}

export interface PwaInstallPromptProps {
  /** Optional callback to notify parent when iOS guide banner should render */
  onIosBannerChange?: (visible: boolean) => void;
  /** Render only the iOS Add-to-Home-Screen banner (for placement below the header) */
  variant?: 'button' | 'ios-banner';
  iosBannerOpen?: boolean;
  onDismissIosBanner?: () => void;
}

export function usePwaInstall() {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [isInstalled, setIsInstalled] = useState<boolean>(() => isStandaloneMode());
  const [isIosSafari] = useState<boolean>(() => !isStandaloneMode() && detectIosSafari());
  const [iosTipOpen, setIosTipOpen] = useState<boolean>(false);

  useEffect(() => {
    const handleBeforeInstallPrompt = (event: Event) => {
      event.preventDefault();
      setDeferredPrompt(event as BeforeInstallPromptEvent);
    };

    const handleAppInstalled = () => {
      setIsInstalled(true);
      setDeferredPrompt(null);
      setIosTipOpen(false);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, []);

  const triggerInstall = async () => {
    if (!deferredPrompt) return;
    await deferredPrompt.prompt();
    const choice = await deferredPrompt.userChoice;
    if (choice.outcome === 'accepted') {
      setDeferredPrompt(null);
    }
  };

  return {
    canPromptInstall: !isInstalled && deferredPrompt !== null,
    isIosInstallable: !isInstalled && isIosSafari,
    iosTipOpen,
    setIosTipOpen,
    triggerInstall,
  };
}

export function PwaInstallPrompt({
  variant = 'button',
  iosBannerOpen,
  onDismissIosBanner,
  onIosBannerChange,
}: PwaInstallPromptProps) {
  const {
    canPromptInstall,
    isIosInstallable,
    iosTipOpen,
    setIosTipOpen,
    triggerInstall,
  } = usePwaInstall();

  const effectiveBannerOpen = iosBannerOpen ?? iosTipOpen;

  if (variant === 'ios-banner') {
    if (!effectiveBannerOpen) return null;
    return (
      <div className="ios-install-banner" role="region" aria-label="Install PulseCam on iOS">
        <div className="ios-install-content">
          <Share size={16} className="ios-install-icon" aria-hidden="true" />
          <span>
            Install PulseCam: tap <span className="ios-install-strong">Share</span> in Safari, then
            choose <span className="ios-install-strong">Add to Home Screen</span>.
          </span>
        </div>
        <button
          type="button"
          className="ios-install-dismiss"
          onClick={() => {
            setIosTipOpen(false);
            onDismissIosBanner?.();
          }}
          aria-label="Dismiss install tip"
        >
          <X size={16} />
        </button>
      </div>
    );
  }

  if (canPromptInstall) {
    return (
      <button
        type="button"
        className="pwa-install-btn"
        onClick={() => void triggerInstall()}
        aria-label="Install PulseCam app"
      >
        <Download size={14} />
        <span>Install App</span>
      </button>
    );
  }

  if (isIosInstallable) {
    const toggleIosTip = () => {
      const next = !effectiveBannerOpen;
      setIosTipOpen(next);
      onIosBannerChange?.(next);
    };

    return (
      <button
        type="button"
        className="pwa-install-btn"
        onClick={toggleIosTip}
        aria-expanded={effectiveBannerOpen}
        aria-label="How to add PulseCam to Home Screen"
      >
        <Share size={14} />
        <span>Add App</span>
      </button>
    );
  }

  return null;
}
