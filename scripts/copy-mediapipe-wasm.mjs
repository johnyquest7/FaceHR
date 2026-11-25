import { cpSync, existsSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const srcDir = resolve('node_modules/@mediapipe/tasks-vision/wasm');
const destDir = resolve('public/mediapipe/wasm');

if (!existsSync(srcDir)) {
  console.warn('[copy-mediapipe-wasm] Source directory not found, skipping:', srcDir);
  process.exit(0);
}

mkdirSync(destDir, { recursive: true });
cpSync(srcDir, destDir, { recursive: true, force: true });
console.log('[copy-mediapipe-wasm] Copied MediaPipe WASM runtime files to public/mediapipe/wasm');
