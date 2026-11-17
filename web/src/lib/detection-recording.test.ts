import assert from "node:assert/strict";
import test from "node:test";
import { DetectionRecording } from "./detection-recording";
import type { RecordingContext } from "./detection-recording";
import type { FrameResult } from "./detector";
import { letterbox } from "./detector";

function frame(score: number): FrameResult {
  return {
    detections: [
      {
        classId: 0,
        score,
        presence: 0.99,
        x: 12,
        y: 34,
        box: [10, 30, 20, 40],
      },
    ],
    transform: letterbox(640, 480, 160, false),
    cube: { box: [0, 0, 100, 100], score: 0.88 },
    roi: [0, 0, 100, 100],
    roiSource: "tracker",
    trackingLost: false,
    inferenceMs: 25,
    bboxMs: null,
    poseMs: 25,
    totalMs: 26,
    profile: {
      bbox: null,
      pose: null,
      bboxDecodeMs: 0,
      poseDecodeMs: 0,
      trackingDecodeMs: 0,
      roiUpdateMs: 0,
      nonInferenceMs: 1,
      otherMs: 1,
    },
  };
}
const context: RecordingContext = {
  sourceId: 1,
  sourceName: "first.mp4",
  sourceKind: "video",
  sourceFps: 60,
  segment: 1,
  timestampSeconds: 0,
  backend: "webgpu",
  settings: {
    thresholds: [0.25, 0.4, 0.25],
    presenceThreshold: 0.05,
    bboxThreshold: 0.25,
    roiPadding: 0.4,
  },
};
test("download data retains every result, source, seek segment and separate confidence channels", () => {
  const recording = new DetectionRecording();
  recording.append(frame(0.42), context);
  recording.append(frame(0.75), { ...context, timestampSeconds: 1 / 30 });
  recording.append(frame(0.91), {
    ...context,
    sourceId: 2,
    sourceName: "next.mp4",
    segment: 7,
    backend: "wasm",
  });
  const saved = JSON.parse(JSON.stringify(recording.toJSON())) as ReturnType<
    DetectionRecording["toJSON"]
  >;
  assert.equal(recording.length, 3);
  assert.equal(saved.frames.length, 3);
  assert.deepEqual(
    saved.frames.map((f) => f.result.detections[0].score),
    [0.42, 0.75, 0.91],
  );
  assert.deepEqual(
    saved.frames.map((f) => f.result.detections[0].presence),
    [0.99, 0.99, 0.99],
  );
  assert.equal(saved.frames[0].result.cube?.score, 0.88);
  assert.equal(saved.frames[1].timestampSeconds, 1 / 30);
  assert.equal(saved.frames[2].sourceId, 2);
  assert.equal(saved.frames[2].segment, 7);
  assert.equal(saved.frames[2].backend, "wasm");
  assert.deepEqual(saved.frames[0].settings, context.settings);
  assert.equal(saved.sources.length, 2);
  assert.equal(saved.sources[0].name, "first.mp4");
  assert.match(saved.confidenceFields["result.detections[].score"], /object/);
  assert.match(
    saved.confidenceFields["result.detections[].presence"],
    /presence/,
  );
});
