import assert from "node:assert/strict";
import test from "node:test";
import * as current from "./preprocess";
import * as reference from "../../tests/resize-reference";

function pixels(width: number, height: number) {
  const data = new Uint8ClampedArray(width * height * 4);
  let seed = 17;
  for (let i = 0; i < data.length; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    data[i] = seed >>> 24;
  }
  return data;
}

// Frozen pre-optimization kernels catch changes in sampling, channel order,
// float32 accumulation and 8-bit rounding across non-integer resize ratios.
for (const [w, h, tw, th] of [
  [608, 608, 256, 256],
  [591, 591, 256, 256],
  [721, 719, 256, 256],
  [1280, 720, 640, 360],
  [256, 256, 160, 160],
  [256, 256, 288, 288],
  [512, 512, 256, 256],
  [1, 7, 13, 5],
  [17, 1, 3, 8],
  [7, 9, 7, 9],
]) {
  for (const kind of ["resizeArea", "resizeLinear"] as const) {
    test(`${kind} preserves pixels at ${w}x${h} -> ${tw}x${th}`, () => {
      const input = pixels(w, h);
      assert.deepEqual(
        current[kind](input, w, h, tw, th),
        reference[kind](input, w, h, tw, th),
      );
    });
  }
}
test("two-step pose resize preserves pixels and returned buffer ownership across cache eviction", () => {
  const input = pixels(608, 608);
  const result = current.posePixels(input, 608, 608, 256, 160);
  const expected = reference.posePixels(input, 608, 608, 256, 160);
  assert.deepEqual(result, expected);
  for (let side = 3; side < 38; side++) {
    current.resizeArea(pixels(side, side), side, side, 2, 2);
    current.resizeLinear(pixels(side, side), side, side, 2, 2);
  }
  assert.deepEqual(result, expected);
  assert.deepEqual(current.posePixels(input, 608, 608, 256, 160), expected);
});
