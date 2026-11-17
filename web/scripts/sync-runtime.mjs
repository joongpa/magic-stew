import { mkdir, copyFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

// Serve the WASM bootstrap from the same origin, with exactly the JS package's version.
const require = createRequire(import.meta.url);
const dist = dirname(require.resolve("onnxruntime-web"));
const target = new URL("../public/runtime/", import.meta.url);
await mkdir(target, { recursive: true });
for (const file of [
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
]) {
  await copyFile(join(dist, file), new URL(file, target));
}
