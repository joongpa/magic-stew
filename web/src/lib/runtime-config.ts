import type { InferenceBackend } from "./detector";

/** WebGPU keeps one WASM orchestration thread; CPU gets an isolated SIMD pool. */
export function wasmThreadCount(
  backend: InferenceBackend,
  isolated: boolean,
  cores: number,
): number {
  if (backend !== "wasm" || !isolated) return 1;
  return Math.max(
    1,
    Math.min(4, Math.floor(Number.isFinite(cores) ? cores : 1)),
  );
}
