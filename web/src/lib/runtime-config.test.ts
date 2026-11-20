import assert from "node:assert/strict";
import test from "node:test";
import { hasWebGpuAdapter, wasmThreadCount } from "./runtime-config";

test("WebGPU requires an available adapter, not just an exposed API", async () => {
  assert.equal(await hasWebGpuAdapter(undefined), false);
  assert.equal(await hasWebGpuAdapter({requestAdapter: async () => null}), false);
  assert.equal(await hasWebGpuAdapter({requestAdapter: async () => ({} as GPUAdapter)}), true);
  assert.equal(await hasWebGpuAdapter({requestAdapter: async () => { throw new Error('GPU blocked'); }}), false);
});

test("CPU uses up to four threads only when isolated", () => {
  assert.equal(wasmThreadCount("wasm", true, 8), 4);
  assert.equal(wasmThreadCount("wasm", true, 2), 2);
  assert.equal(wasmThreadCount("wasm", false, 8), 1);
  assert.equal(wasmThreadCount("webgpu", true, 8), 1);
  assert.equal(wasmThreadCount("wasm", true, NaN), 1);
});
