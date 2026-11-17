import type { FrameResult, InferenceBackend, Settings } from "./detector";

export interface RecordingContext {
  sourceId: number;
  sourceName: string;
  sourceKind: "image" | "video" | "camera";
  sourceFps: number;
  segment: number;
  timestampSeconds: number;
  backend: InferenceBackend;
  settings: Settings;
}

/** Retains results without per-frame serialization or copying the detections. */
export class DetectionRecording {
  private readonly startedAt = new Date().toISOString();
  private readonly sources = new Map<
    number,
    { id: number; name: string; kind: string; fps: number }
  >();
  private readonly frames: {
    sequence: number;
    sourceId: number;
    segment: number;
    timestampSeconds: number;
    backend: InferenceBackend;
    settings: Settings;
    result: FrameResult;
  }[] = [];

  get length() {
    return this.frames.length;
  }

  append(result: FrameResult, context: RecordingContext) {
    if (!this.sources.has(context.sourceId))
      this.sources.set(context.sourceId, {
        id: context.sourceId,
        name: context.sourceName,
        kind: context.sourceKind,
        fps: context.sourceFps,
      });
    this.frames.push({
      sequence: this.frames.length,
      sourceId: context.sourceId,
      segment: context.segment,
      timestampSeconds: context.timestampSeconds,
      backend: context.backend,
      settings: context.settings,
      result,
    });
  }

  toJSON() {
    return {
      version: 1,
      startedAt: this.startedAt,
      confidenceFields: {
        "result.detections[].score": "Pose per-object detection/class score",
        "result.detections[].presence": "Keypoint-presence score",
        "result.cube.score":
          "Separate cube bbox detector score from the last acquisition",
      },
      fittingFields: {
        "result.fittedPose.stage": "cpu_pipeline means startup, initial, neighbors, orientation and recovery are complete",
        "result.fittedPose.pipelineVersion": "Content-addressed Rust/WASM asset loaded by the fitter",
        "result.fittedPose.fittingConfig": "Frame sequence configuration and buffered fitting segment for standalone replay",
        "result.fittedPose.bank[].score": "Balanced geometry cost; lower is better, not a detector confidence",
        "result.fittedPose.bank[].nearest": "Projected sticker-face ID for each observed sticker keypoint",
        "result.fittedPose.bank[].valid": "Whether that sticker keypoint has an eligible projected-face assignment",
        "result.fittedPose.bank[].flat_fallback": "Reserved zero-angle option outside the five assignment-distinct slots",
      },
      sources: [...this.sources.values()],
      frames: this.frames,
    };
  }
}
