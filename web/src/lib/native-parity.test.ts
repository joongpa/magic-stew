import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  decode,
  decodeCube,
  letterbox,
  roiTransform,
  RoiTracker,
} from "./detector";
import type { ModelManifest, Settings } from "./detector";
import { resizeArea, resizeLinear } from "./preprocess";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/native-parity.json", import.meta.url),
    "utf8",
  ),
);
const manifest = JSON.parse(
  readFileSync(
    new URL("../../public/models/detector.json", import.meta.url),
    "utf8",
  ),
) as ModelManifest;
const settings: Settings = {
  thresholds: manifest.classThresholds,
  presenceThreshold: 0.05,
  bboxThreshold: 0.25,
  roiPadding: 0.4,
};
function raw(name: string) {
  const bytes = readFileSync(new URL(`./fixtures/${name}`, import.meta.url));
  return new Float32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
}

for (const c of fixture.resize) {
  test(`OpenCV ${c.kind} pixel parity: ${c.width}x${c.height} -> ${c.targetWidth}x${c.targetHeight}`, () => {
    const input = new Uint8ClampedArray(c.width * c.height * 4);
    for (let i = 0; i < c.width * c.height; i++) {
      input.set(c.source.slice(i * 3, i * 3 + 3), i * 4);
      input[i * 4 + 3] = 255;
    }
    const actual = (c.kind === "area" ? resizeArea : resizeLinear)(
      input,
      c.width,
      c.height,
      c.targetWidth,
      c.targetHeight,
    );
    let max = 0;
    for (let i = 0; i < c.targetWidth * c.targetHeight; i++)
      for (let k = 0; k < 3; k++)
        max = Math.max(
          max,
          Math.abs(actual[i * 4 + k] - c.expected[i * 3 + k]),
        );
    assert.ok(
      max <= 1,
      `Max pixel difference ${max}; expected at most one 8-bit level`,
    );
  });
}
for (const c of fixture.pose) {
  test(`native renderer detection and tracking parity on real ${c.size}px tensor`, () => {
    const t = roiTransform(c.sourceWidth, c.sourceHeight, c.size, c.roi);
    const actual = decode(raw(c.raw), c.shape, t, manifest, settings);
    assert.equal(actual.length, c.expected.length);
    for (let i = 0; i < actual.length; i++) {
      assert.equal(actual[i].classId, c.expected[i].classId);
      for (const key of [
        "score",
        "presence",
        "x",
        "y",
        "localX",
        "localY",
      ] as const)
        assert.ok(
          Math.abs(actual[i][key]! - c.expected[i][key]) < 0.0002,
          `${i} ${key}: ${actual[i][key]} vs ${c.expected[i][key]}`,
        );
    }
    const tracker = new RoiTracker();
    tracker.initialize(
      { box: c.roi, score: 0.9 },
      c.sourceWidth,
      c.sourceHeight,
      0,
    );
    tracker.observe(
      actual,
      c.sourceWidth,
      c.sourceHeight,
      manifest.tracking.gain,
      manifest.tracking.padding,
    );
    assert.deepEqual(tracker.roi, c.nextRoi);
  });
}
test("native rectangular cube acquisition matches the real detector tensor", () => {
  const c = fixture.bbox;
  const t = letterbox(c.sourceWidth, c.sourceHeight, 640, false, 32);
  assert.deepEqual([1, 3, t.inputHeight, t.inputWidth], c.inputShape);
  const actual = decodeCube(raw(c.raw), c.shape, t, 0.25);
  assert.deepEqual(actual?.box, c.expected.box);
  assert.ok(actual && Math.abs(actual.score - c.expected.score) < 1e-6);
});
