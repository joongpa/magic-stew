import assert from "node:assert/strict";
import test from "node:test";
import { wasmThreadCount } from "./runtime-config";

test("CPU uses up to four threads only when isolated", () => {
  assert.equal(wasmThreadCount("wasm", true, 8), 4);
  assert.equal(wasmThreadCount("wasm", true, 2), 2);
  assert.equal(wasmThreadCount("wasm", false, 8), 1);
  assert.equal(wasmThreadCount("webgpu", true, 8), 1);
  assert.equal(wasmThreadCount("wasm", true, NaN), 1);
});
