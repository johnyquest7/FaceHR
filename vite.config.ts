import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import basicSsl from '@vitejs/plugin-basic-ssl';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig(({ mode }) => {
  const enableHttps = mode === 'https' || process.env.VITE_HTTPS === 'true';

  return {
    base: process.env.VITE_BASE_PATH || './',
    plugins: [
      react(),
      ...(enableHttps ? [basicSsl()] : []),
      VitePWA({
        registerType: 'autoUpdate',
        includeAssets: ['favicon.svg', 'icons/icon-192.png', 'icons/icon-512.png'],
        manifest: {
          name: 'PulseCam — Camera Heart Rate Monitor',
          short_name: 'PulseCam',
          description: '100% on-device camera heart rate estimator using remote photoplethysmography (rPPG).',
          theme_color: '#090d16',
          background_color: '#090d16',
          display: 'standalone',
          orientation: 'portrait-primary',
          start_url: './',
          scope: './',
          icons: [
            {
              src: 'icons/icon-192.png',
              sizes: '192x192',
              type: 'image/png',
              purpose: 'any maskable',
            },
            {
              src: 'icons/icon-512.png',
              sizes: '512x512',
              type: 'image/png',
              purpose: 'any maskable',
            },
          ],
        },
        workbox: {
          globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
          globIgnores: ['**/mediapipe/wasm/**'],
          runtimeCaching: [
            {
              urlPattern: /\/mediapipe\/wasm\/.*/i,
              handler: 'CacheFirst',
              options: {
                cacheName: 'mediapipe-wasm',
                expiration: {
                  maxEntries: 10,
                  maxAgeSeconds: 60 * 60 * 24 * 30,
                },
              },
            },
            {
              urlPattern: /^https:\/\/storage\.googleapis\.com\/mediapipe-models\/.*/i,
              handler: 'CacheFirst',
              options: {
                cacheName: 'mediapipe-models',
                expiration: {
                  maxEntries: 5,
                  maxAgeSeconds: 60 * 60 * 24 * 30,
                },
              },
            },
          ],
        },
      }),
    ],
    server: {
      host: enableHttps ? true : 'localhost',
      port: 5173,
    },
    preview: {
      host: true,
      port: 4173,
    },
  };
});
