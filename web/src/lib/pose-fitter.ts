import type { FrameResult } from "./detector";
import type { FittedPose, PoseObservations } from "./pose-types";
import { BufferedPosePipeline, type PipelineFrame } from "../../../pose-fitter/runtime/buffered-pipeline.mjs";
import { PoseWorkerPool } from "./pose-worker-pool";

export function poseObservations(frame: FrameResult): PoseObservations {
  const landmarks = frame.detections.filter(point => point.classId < 3);
  const stickers = frame.detections.filter(point => point.classId >= 3);
  return {
    width: frame.transform.sourceWidth, height: frame.transform.sourceHeight,
    observations: { points: landmarks.map(p => [p.x, p.y]), classes: landmarks.map(p => p.classId),
      stickers: stickers.map(p => [p.x, p.y]), uncertainties: stickers.map(() => 1) },
  };
}
const COLORS = ["corner", "corner-edge", "corner-edge-center", "white", "yellow", "orange", "red", "blue", "green"];
const CENTERS = [
  { axis: 2, side: 1, color: "white" }, { axis: 2, side: -1, color: "yellow" },
  { axis: 1, side: -1, color: "green" }, { axis: 1, side: 1, color: "blue" },
  { axis: 0, side: 1, color: "red" }, { axis: 0, side: -1, color: "orange" },
];
export function pipelineFrame(frame: FrameResult, frameIndex: number): PipelineFrame {
  return { ...poseObservations(frame), frame: frameIndex,
    pixel_scale: Math.max(Math.max(frame.transform.cropWidth, frame.transform.cropHeight) / 400, 0.3),
    landmark_presence: frame.detections.filter(p => p.classId < 3).map(p => p.presence),
    colors: frame.detections.filter(p => p.classId >= 3).map(p => COLORS[p.classId]),
  };
}

/** Shared buffered scheduler; all numerical stages execute in the CPU pool. */
export class PoseFitter {
  private pool: PoseWorkerPool;
  private stream?: BufferedPosePipeline;
  private segments = new Set<BufferedPosePipeline>();
  private dimensions = "";
  private fps = 0;
  private segment = 0;
  constructor(baseUrl: string) { this.pool = new PoseWorkerPool(baseUrl); }
  fit(frame: FrameResult, reset: boolean, timestamp = 0, sourceFps = 30): Promise<FittedPose> {
    const fps = sourceFps > 0 ? sourceFps : 30;
    const dimensions = frame.transform.sourceWidth + "x" + frame.transform.sourceHeight;
    if (reset || !this.stream || this.stream.ended || dimensions !== this.dimensions || Math.abs(fps - this.fps) > 0.001) {
      if (this.stream && !this.stream.ended) void this.stream.flush().catch(() => {});
      this.stream = new BufferedPosePipeline(job => this.pool.execute(job), { fps, turningStartFrame: null, centers: CENTERS });
      this.segments.add(this.stream); this.dimensions = dimensions; this.fps = fps;
      this.segment++;
    }
    const sourceFrame = Math.round(timestamp * fps), segment = this.segment;
    return this.stream.push(pipelineFrame(frame, sourceFrame)).then(value => ({ ...value, sourceFrame,
      fittingConfig: { segment, fps, turningStartFrame: null, centers: CENTERS } }));
  }
  async flush() {
    const segments = [...this.segments];
    await Promise.all(segments.map(segment => segment.flush()));
    for (const segment of segments) this.segments.delete(segment);
  }
  dispose() {
    for (const segment of this.segments) segment.cancel(); this.segments.clear(); this.pool.dispose();
  }
}
