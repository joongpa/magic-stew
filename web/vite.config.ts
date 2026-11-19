import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import {localDiagnostics} from './scripts/local-diagnostics';

const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  plugins: [svelte(),localDiagnostics()],
  resolve: {
    conditions: [
      "onnxruntime-web-use-extern-wasm",
      "module",
      "browser",
      "development|production",
    ],
  },
  worker: { format: "es" },
  // cubing.js dynamically imports its worker entry. Keep Vite's preload helper
  // separate from the Svelte entry, so workers never import/execute the app.
  build: { target: "es2022", modulePreload: false, rollupOptions: { output: {
    manualChunks(id) { if (id.includes('vite/preload-helper')) return 'preload-helper'; },
  } } },
});
