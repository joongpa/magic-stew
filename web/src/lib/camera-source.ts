import type { SourceFrame } from "./frame-stream";

interface Processor {
  readable: ReadableStream<VideoFrame>;
  discardedFrames?: number;
}
type ProcessorConstructor = new (options: {
  track: MediaStreamTrack;
  maxBufferSize: number;
}) => Processor;

export async function startCameraCapture(
  original: MediaStreamTrack,
  onFrame: (frame: SourceFrame) => void,
  onStopped: (message: string) => void,
) {
  const worker = new Worker(new URL("./capture.worker.ts", import.meta.url), {
    type: "module",
  });
  const track = original.clone();
  let monitor: ReturnType<typeof setInterval> | undefined;
  let ended = false;
  const stop = () => {
    ended = true;
    clearInterval(monitor);
    worker.terminate();
    track.stop();
  };
  const ready = new Promise<void>((resolve, reject) => {
    let started = false;
    worker.onmessage = ({ data }) => {
      if (ended) {
        data.bitmap?.close();
        return;
      }
      if (data.type === "started") {
        started = true;
        resolve();
      }
      if (data.type === "frame")
        onFrame({ bitmap: data.bitmap, timestamp: data.timestamp });
      if (data.type === "stopped") {
        if (!started) reject(new Error(data.message));
        else onStopped(data.message);
      }
      if (data.type === "ended") {
        clearInterval(monitor);
        onStopped("");
      }
    };
    worker.onerror = () => {
      if (!started) reject(new Error("Camera frame capture could not start."));
      else
        onStopped(
          "Camera frame capture failed. Buffered frames will finish processing.",
        );
    };
  });
  try {
    // Browsers expose TrackProcessor in different realms. Support both without
    // falling back to presentation callbacks, which may skip frames under load.
    const Constructor = (
      window as unknown as { MediaStreamTrackProcessor?: ProcessorConstructor }
    ).MediaStreamTrackProcessor;
    const sourceFps = original.getSettings().frameRate || 30;
    if (Constructor) {
      const processor = new Constructor({ track, maxBufferSize: 8 });
      const readable = processor.readable;
      worker.postMessage({ type: "start", readable, sourceFps }, [readable]);
      monitor = setInterval(() => {
        if (processor.discardedFrames) {
          stop();
          onStopped(
            "The browser discarded camera frames. Capture stopped; buffered frames will finish processing.",
          );
        }
      }, 50);
    } else
      worker.postMessage({ type: "start", track, sourceFps }, [
        track as unknown as Transferable,
      ]);
    await ready;
    return {
      pause() {
        worker.postMessage({ type: "pause" });
      },
      resume() {
        worker.postMessage({ type: "resume" });
      },
      acknowledge() {
        if (!ended) worker.postMessage({ type: "ack" });
      },
      stop,
    };
  } catch (error) {
    stop();
    throw error;
  }
}
export type CameraCapture = Awaited<ReturnType<typeof startCameraCapture>>;
