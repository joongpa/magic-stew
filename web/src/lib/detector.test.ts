import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  decode,
  decodeCube,
  letterbox,
  rgbaToChw,
  RoiTracker,
  roiTransform,
} from "./detector";
import type { ModelManifest, Settings } from "./detector";

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
const t = letterbox(160, 160, 160, false);
function tensor(rows: number[][]): Float32Array {
  return new Float32Array(
    Array.from({ length: 16 }, (_, channel) =>
      rows.map((row) => row[channel]),
    ).flat(),
  );
}
function row(
  classId: number,
  score: number,
  presence = 0.99,
  x = 80,
  y = 80,
): number[] {
  const r = [x, y, 10, 10, ...Array(9).fill(0.01), x, y, presence];
  r[4 + classId] = score;
  return r;
}

test("RGB normalization is planar NCHW and ignores alpha", () => {
  assert.deepEqual(
    [...rgbaToChw(new Uint8ClampedArray([255, 0, 128, 0, 0, 255, 0, 255]))],
    [1, 0, 0, 1, Math.fround(128 / 255), 0],
  );
});

test("letterbox uses integer padding and restores source coordinates", () => {
  const transform = letterbox(640, 360, 160, false);
  assert.equal(transform.padY, 35);
  const [d] = decode(
    tensor([row(0, 0.8, 0.99, 80, 80)]),
    [1, 16, 1],
    transform,
    manifest,
    settings,
  );
  assert.equal(d.x, 320);
  assert.equal(d.y, 180);
  const crop = letterbox(640, 360, 160, true);
  assert.equal(crop.cropX, 140);
  assert.equal(crop.padY, 0);
  const [cropped] = decode(
    tensor([row(0, 0.8, 0.99, 0, 0)]),
    [1, 16, 1],
    crop,
    manifest,
    settings,
  );
  assert.equal(cropped.x, 140);
  assert.equal(cropped.y, 0);
});

test("object scores determine per-class filtering, separately from presence", () => {
  const data = tensor([
    row(3, 0.89, 1),
    row(3, 0.95, 0.2),
    row(0, 0.3, 0.06),
    row(1, 0.39, 1),
    row(2, 0.8, 0.01),
  ]);
  const detections = decode(data, [1, 16, 5], t, manifest, settings);
  assert.equal(detections.length, 1);
  assert.equal(detections[0].classId, 0);
  assert.ok(detections[0].score < 0.31 && detections[0].presence < 0.07);
});

test("class-aware NMS suppresses duplicates but retains coincident different classes", () => {
  assert.deepEqual(
    decode(
      tensor([row(0, 0.8), row(0, 0.6), row(1, 0.9)]),
      [1, 16, 3],
      t,
      manifest,
      settings,
    ).map((d) => d.classId),
    [1, 0],
  );
});

test("NMS precedes presence filtering, matching the native prediction order", () => {
  assert.equal(
    decode(
      tensor([row(0, 0.9, 0.01), row(0, 0.8, 1)]),
      [1, 16, 2],
      t,
      manifest,
      settings,
    ).length,
    0,
  );
});

test("rejects incompatible tensor layouts and clips keypoints like the native predictor", () => {
  assert.throws(
    () => decode(new Float32Array(9), [1, 9], t, manifest, settings),
    /Unexpected/,
  );
  assert.equal(
    decode(
      tensor([row(0, 0.9, 0.99, 80, 10)]),
      [1, 16, 1],
      letterbox(640, 360, 160, false),
      manifest,
      settings,
    )[0].y,
    0,
  );
});

test("cube acquisition uses its own score, the best cube, and full-frame letterbox mapping", () => {
  // [1,5,3]: cx, cy, width, height, cube score (there is no keypoint channel).
  const data = new Float32Array([
    80, 100, 120, 80, 80, 80, 20, 20, 20, 20, 20, 20, 0.1, 0.8, 0.7,
  ]);
  const cube = decodeCube(
    data,
    [1, 5, 3],
    letterbox(640, 360, 160, false),
    0.25,
  );
  assert.deepEqual(cube?.box, [360, 140, 440, 220]);
  assert.ok(cube && Math.abs(cube.score - 0.8) < 1e-6);
  assert.equal(decodeCube(data, [1, 5, 3], t, 0.9), null);
  assert.throws(() => decodeCube(data, [1, 6, 3], t, 0.25), /Unexpected cube/);
});

test("a pose inside an offset ROI maps into the original full frame", () => {
  const transform = roiTransform(1280, 720, 160, [100, 120, 300, 320]);
  const [point] = decode(
    tensor([row(0, 0.8, 0.99, 80, 80)]),
    [1, 16, 1],
    transform,
    manifest,
    settings,
  );
  assert.equal(point.x, 200);
  assert.equal(point.y, 220);
});

function point(x: number, y: number) {
  return {
    classId: 0,
    score: 0.9,
    presence: 0.9,
    x,
    y,
    box: [x - 1, y - 1, x + 1, y + 1] as [number, number, number, number],
  };
}

test("ROI tracker matches the native acquisition, movement, miss and reacquisition example", () => {
  const tracker = new RoiTracker();
  const cube = {
    box: [10, 10, 30, 30] as [number, number, number, number],
    score: 0.9,
  };
  assert.deepEqual(tracker.initialize(cube, 60, 60, 0), [10, 10, 30, 30]);
  tracker.observe([point(20, 15), point(29.9, 25)], 60, 60, 1, 0);
  assert.deepEqual(tracker.roi, [15, 10, 35, 30]);
  assert.equal(tracker.cube, cube); // acquisition score remains the last actual bbox score
  tracker.observe([], 60, 60, 1, 0);
  assert.equal(tracker.roi, null);
  assert.equal(tracker.cube, null);
  assert.deepEqual(tracker.initialize(cube, 60, 60, 0), [10, 10, 30, 30]);
});

test("ROI padding, base-side floor and clipping match the native renderer", () => {
  const tracker = new RoiTracker();
  assert.deepEqual(
    tracker.initialize({ box: [10, 10, 30, 30], score: 0.9 }, 100, 100, 0.8),
    [0, 0, 52, 52],
  );
  tracker.observe([point(26, 26)], 100, 100, 1.4, 0.18);
  assert.deepEqual(tracker.roi, [0, 0, 52, 52]);
  tracker.reset();
  assert.deepEqual(
    tracker.initialize({ box: [20, 20, 50, 50], score: 0.9 }, 100, 100, 0.8),
    [0, 0, 78, 78],
  );
});

test("global prefilter and max-det precede per-class pose thresholds", () => {
  const rows = Array.from({ length: 129 }, (_, i) =>
    row(
      3,
      0.8 - i * 0.0001,
      0.99,
      (i % 16) * 12 + 5,
      Math.floor(i / 16) * 12 + 5,
    ),
  );
  rows.push(row(0, 0.3, 0.99, 155, 155));
  // The native 128 slots are occupied by sticker objects that later fail 0.90.
  assert.equal(
    decode(tensor(rows), [1, 16, rows.length], t, manifest, settings).length,
    0,
  );
});
