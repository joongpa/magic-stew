import assert from "node:assert/strict";
import test from "node:test";
import { pipelineFrame, poseObservations } from "./pose-fitter";
import type { FrameResult } from "./detector";

test("pose fitting receives source coordinates and already-filtered object classes", () => {
  const frame = {
    transform: { sourceWidth: 1080, sourceHeight: 1920 },
    detections: [
      { classId: 0, x: 430, y: 910, localX: 2, localY: 3, score: 0.5, presence: 0.9 },
      { classId: 2, x: 450, y: 920, localX: 4, localY: 5, score: 0.7, presence: 0.95 },
      { classId: 7, x: 440, y: 915, localX: 3, localY: 4, score: 0.92, presence: 0.99 },
    ],
  } as unknown as FrameResult;
  assert.deepEqual(poseObservations(frame), {
    width: 1080, height: 1920,
    observations: { points: [[430, 910], [450, 920]], classes: [0, 2], stickers: [[440, 915]], uncertainties: [1] },
  });
  assert.equal(frame.detections[0].score, 0.5);
  assert.equal(frame.detections[0].presence, 0.9);
});

test("CPU pipeline gets ROI pixel scale, source frame number, colors and presence scores", () => {
  const frame = { transform: { sourceWidth: 1920, sourceHeight: 1080, cropWidth: 240, cropHeight: 200 },
    detections: [{ classId: 1, x: 20, y: 30, score: 0.45, presence: 0.98 },
      { classId: 5, x: 40, y: 50, score: 0.91, presence: 0.99 }] } as FrameResult;
  const input = pipelineFrame(frame, 392);
  assert.equal(input.frame, 392);
  assert.equal(input.pixel_scale, 0.6);
  assert.deepEqual(input.landmark_presence, [0.98]);
  assert.deepEqual(input.colors, ["orange"]);
  assert.deepEqual(input.observations.stickers, [[40, 50]]);
});
