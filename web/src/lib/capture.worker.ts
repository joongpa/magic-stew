import { FrameSelector } from "./frame-stream";

interface Processor {
  readable: ReadableStream<VideoFrame>;
  discardedFrames?: number;
}
type ProcessorConstructor = new (options: {
  track: MediaStreamTrack;
  maxBufferSize: number;
}) => Processor;
interface CaptureScope {
  MediaStreamTrackProcessor?: ProcessorConstructor;
  onmessage: (event: MessageEvent) => void;
  postMessage: (message: unknown, transfer?: Transferable[]) => void;
}
const scope = self as unknown as CaptureScope;
let paused = false,
  pending = 0,
  stopped = false;
let track: MediaStreamTrack | undefined;
let reader: ReadableStreamDefaultReader<VideoFrame> | undefined;
const MAX_PENDING = 60;
let canvas: OffscreenCanvas | undefined;

scope.onmessage = async ({ data }) => {
  if (data.type === "ack") {
    pending = Math.max(0, pending - 1);
    return;
  }
  if (data.type === "pause") {
    paused = true;
    return;
  }
  if (data.type === "resume") {
    paused = false;
    return;
  }
  if (data.type !== "start") return;
  track = data.track;
  try {
    let processor: Processor | undefined;
    if (data.readable)
      reader = (data.readable as ReadableStream<VideoFrame>).getReader();
    else {
      const Constructor = scope.MediaStreamTrackProcessor;
      if (!Constructor || !track)
        throw new Error(
          "This browser does not support frame-preserving camera capture. Use a supported Chromium browser or open a video file.",
        );
      processor = new Constructor({ track, maxBufferSize: 8 });
      reader = processor.readable.getReader();
    }
    const selector = new FrameSelector(data.sourceFps || 30);
    let firstTimestamp: number | undefined;
    scope.postMessage({ type: "started" });
    while (!stopped) {
      const { value: frame, done } = await reader.read();
      if (done) break;
      try {
        if (processor?.discardedFrames)
          throw new Error(
            "The browser discarded camera frames. Capture stopped; buffered frames will finish processing.",
          );
        if (paused || !selector.accept(frame.timestamp / 1e6)) continue;
        firstTimestamp ??= frame.timestamp;
        if (
          !canvas ||
          canvas.width !== frame.displayWidth ||
          canvas.height !== frame.displayHeight
        ) {
          canvas = new OffscreenCanvas(frame.displayWidth, frame.displayHeight);
        }
        const context = canvas.getContext("2d", { alpha: false });
        if (!context) throw new Error("Camera frame copying is unavailable.");
        // Copy out of the camera's native buffer pool before retaining a frame.
        context.drawImage(frame, 0, 0);
        const bitmap = canvas.transferToImageBitmap();
        pending++;
        scope.postMessage(
          {
            type: "frame",
            bitmap,
            timestamp: (frame.timestamp - firstTimestamp) / 1e6,
          },
          [bitmap],
        );
        if (pending >= MAX_PENDING) {
          scope.postMessage({
            type: "stopped",
            message:
              "Camera processing fell behind the 60-frame buffer. Capture stopped; every buffered frame will finish processing. Resume to start a new capture.",
          });
          stopped = true;
        }
      } finally {
        frame.close();
      }
    }
  } catch (error) {
    scope.postMessage({
      type: "stopped",
      message: error instanceof Error ? error.message : String(error),
    });
  } finally {
    stopped = true;
    track?.stop();
    await reader?.cancel().catch(() => {});
    scope.postMessage({ type: "ended" });
  }
};
