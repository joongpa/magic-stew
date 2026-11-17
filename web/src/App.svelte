<script lang="ts">
  import { onMount } from "svelte";
  import { FramePresenter } from "./lib/frame-presenter";
  import { DetectionRecording } from "./lib/detection-recording";
  import type { RecordingContext } from "./lib/detection-recording";
  import { COLORS, LABELS } from "./lib/detector";
  import type {
    FrameResult,
    ModelManifest,
    Settings,
    WorkerRequest,
    WorkerResponse,
  } from "./lib/detector";
  import { FrameQueue, openVideoFile } from "./lib/frame-stream";
  import type { SourceFrame, VideoFileSource } from "./lib/frame-stream";
  import { startCameraCapture } from "./lib/camera-source";
  import type { CameraCapture } from "./lib/camera-source";

  let manifest = $state<ModelManifest>();
  let size = $state(256);
  let backend = $state<"webgpu" | "wasm">("webgpu");
  let runtimeThreads = $state(1);
  let runtimeIsolated = $state(false);
  let status = $state<"loading" | "ready" | "error">("loading");
  let loadingText = $state("Checking WebGPU…");
  let error = $state("");
  let mediaError = $state("");
  let sourceName = $state("No source selected");
  let sourceKind = $state<"none" | "image" | "video" | "camera">("none");
  let running = $state(false);
  let processedFrames = $state(0);
  let queuedFrames = $state(0);
  let sourceFps = $state(0);
  let busy = false,
    refreshPending = false,
    resetTracking = true;
  let bboxThreshold = $state(0.25),
    roiPadding = $state(0.4);
  let showRoi = $state(false);
  const recording = new DetectionRecording();
  let savedFrames = $state(0);
  let thresholds = $state([0.25, 0.4, 0.25, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]);
  let presenceThreshold = $state(0.05);
  let result = $state<FrameResult>();
  let frameTiming = $state<{
    acquireMs: number;
    drawMs: number;
    roundTripMs: number;
    overlayMs: number;
    intervalMs: number;
    pacingMs: number;
  }>();
  let acquireMs = 0,
    pacingMs = 0,
    scheduledAt = 0;
  let previousFrameStart = 0;

  let fps = $state(0),
    duration = $state(0),
    position = $state(0),
    dragging = $state(false);
  let stage: HTMLCanvasElement, fileInput: HTMLInputElement;
  let still: HTMLImageElement | undefined;
  let worker: Worker | undefined;
  let stream: MediaStream | undefined;
  let fileSource: VideoFileSource | undefined;
  let capture: CameraCapture | undefined;
  let seekVersion = 0;
  let seekResume = false,
    seeking = false;
  let captureStopped = false,
    fileEnded = false;
  const cameraQueue = new FrameQueue<SourceFrame & { close(): void }>();
  const presenter = new FramePresenter();
  let timer = 0,
    generation = 0,
    lastResultTime = 0,
    lastStarted = -Infinity;
  let disposed = false;
  let modelGeneration = 0;
  let currentFrame:
    | {
        timestamp: number;
        counted: boolean;
        camera: boolean;
        sentAt: number;
        drawMs: number;
        acquireMs: number;
        pacingMs: number;
        intervalMs: number;
        recording: RecordingContext;
      }
    | undefined;
  const idleWaiters: (() => void)[] = [];
  const baseUrl = new URL(import.meta.env.BASE_URL, location.origin).href;
  const counts = $derived(
    manifest?.classes.map(
      (_, id) => result?.detections.filter((d) => d.classId === id).length ?? 0,
    ) ?? [],
  );
  const settings = (): Settings => ({
    thresholds: thresholds.map((value, id) =>
      Number.isFinite(value)
        ? Math.max(0, Math.min(1, value))
        : (manifest?.classThresholds[id] ?? 1),
    ),
    presenceThreshold: Number.isFinite(presenceThreshold)
      ? Math.max(0, Math.min(1, presenceThreshold))
      : 0.05,
    bboxThreshold: Number.isFinite(bboxThreshold)
      ? Math.max(0, Math.min(1, bboxThreshold))
      : 0.25,
    roiPadding: Number.isFinite(roiPadding)
      ? Math.max(0, Math.min(2, roiPadding))
      : 0.4,
  });
  function send(message: WorkerRequest, transfer: Transferable[] = []) {
    worker?.postMessage(message, transfer);
  }
  function markIdle() {
    busy = false;
    for (const resolve of idleWaiters.splice(0)) resolve();
  }
  function cancelScheduled() {
    clearTimeout(timer);
    timer = 0;
    scheduledAt = 0;
  }

  async function loadModel() {
    const request = ++modelGeneration;
    status = "loading";
    cancelScheduled();
    // Finish an admitted frame before restarting the model; keep the file cursor
    // and camera FIFO intact while the new sessions initialize.
    if (busy) await new Promise<void>((resolve) => idleWaiters.push(resolve));
    if (disposed || request !== modelGeneration) return;
    error = "";
    result = undefined;
    refreshPending = false;
    fps = 0;
    resetTracking = true;
    worker?.terminate();
    worker = undefined;
    if (backend === "webgpu" && !window.isSecureContext) {
      error =
        "Open this app over HTTPS or localhost to use WebGPU and the camera.";
      status = "error";
      return;
    }
    if (backend === "webgpu" && !("gpu" in navigator)) {
      error =
        "WebGPU is unavailable in this browser. Select WASM (CPU), or open this app in a WebGPU-enabled browser.";
      status = "error";
      return;
    }
    loadingText = "Reading detector configuration…";
    try {
      if (!manifest) {
        const response = await fetch(new URL("models/detector.json", baseUrl));
        if (
          !response.ok ||
          !response.headers.get("content-type")?.includes("json")
        )
          throw new Error(
            "Detector files are missing. Run the ONNX export command in web/README.md.",
          );
        manifest = await response.json();
        if (
          !manifest ||
          manifest.version !== 3 ||
          !manifest.bbox ||
          !manifest.tracking ||
          manifest.layout !== "NCHW-xywh-classes-keypoint" ||
          manifest.classes.length !== 9 ||
          manifest.keypointShape.join(",") !== "1,3"
        )
          throw new Error(
            "Incompatible detector manifest. Re-export the checkpoint.",
          );
        size = manifest.defaultSize;
        thresholds = [...manifest.classThresholds];
        presenceThreshold = manifest.presenceThreshold;
        bboxThreshold = manifest.bbox.threshold;
        roiPadding = manifest.tracking.roiPadding;
      }
      if (disposed || request !== modelGeneration) return;
      loadingText = `Loading ${size} × ${size} detector…`;
      const currentWorker = new Worker(
        new URL("./lib/inference.worker.ts", import.meta.url),
        { type: "module" },
      );
      worker = currentWorker;
      currentWorker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
        if (currentWorker !== worker) return;
        if (data.type === "progress")
          loadingText =
            data.fraction < 1
              ? `Downloading ${data.stage.toLowerCase()} · ${Math.round(data.fraction * 100)}%`
              : `${data.stage} · preparing ${backend === "webgpu" ? "GPU" : "CPU"}…`;
        if (data.type === "ready") {
          runtimeThreads = data.threads;
          runtimeIsolated = data.isolated;
          status = "ready";
          lastResultTime = 0;
          if (sourceKind === "image") void inferFrame();
          else if (running) scheduleNext();
          else if (presenter.snapshot) void inferFrame();
        }
        if (data.type === "error") {
          status = "error";
          error = data.message;
          capture?.pause();
          currentFrame = undefined;
          markIdle();
        }
        if (data.type === "result") {
          result = data.result;
          if (
            currentFrame &&
            (currentFrame.counted ||
              currentFrame.recording.sourceKind === "image")
          ) {
            recording.append(data.result, currentFrame.recording);
            savedFrames = recording.length;
          }
          if (currentFrame?.counted) {
            processedFrames++;
            position = currentFrame.timestamp;
            const now = performance.now();
            if (lastResultTime && running)
              fps = 0.8 * fps + 0.2 * (1000 / (now - lastResultTime));
            lastResultTime = now;
            if (currentFrame.camera) capture?.acknowledge();
          }
          const completed = currentFrame;
          const receivedAt = performance.now();
          currentFrame = undefined;
          markIdle();
          const overlayStart = performance.now();
          presenter.present(stage, showRoi ? drawOverlay : undefined);
          if (completed?.counted)
            frameTiming = {
              acquireMs: completed.acquireMs,
              drawMs: completed.drawMs,
              roundTripMs: receivedAt - completed.sentAt,
              overlayMs: performance.now() - overlayStart,
              intervalMs: completed.intervalMs,
              pacingMs: completed.pacingMs,
            };
          if (refreshPending) {
            refreshPending = false;
            void inferFrame();
          } else scheduleNext();
        }
      };
      currentWorker.onerror = () => {
        status = "error";
        error = "The detector worker could not start. Reload the detector.";
        capture?.pause();
        currentFrame = undefined;
        markIdle();
      };
      send({
        type: "init",
        baseUrl,
        manifest: $state.snapshot(manifest),
        size,
        backend,
      });
    } catch (cause) {
      status = "error";
      error = cause instanceof Error ? cause.message : String(cause);
    }
  }

  function stopSource() {
    generation++;
    seekVersion++;
    seeking = false;
    running = false;
    cancelScheduled();
    resetTracking = true;
    capture?.stop();
    capture = undefined;
    stream?.getTracks().forEach((track) => track.stop());
    stream = undefined;
    fileSource?.close();
    fileSource = undefined;
    cameraQueue.clear();
    queuedFrames = 0;
    captureStopped = false;
    fileEnded = false;
    still = undefined;
    presenter.clear(stage);
    sourceKind = "none";
    sourceName = "No source selected";
    result = undefined;
    processedFrames = 0;
    sourceFps = 0;
    position = 0;
    duration = 0;
    lastResultTime = 0;
    lastStarted = -Infinity;
    previousFrameStart = 0;
    frameTiming = undefined;
    fps = 0;
    if (busy) {
      worker?.terminate();
      worker = undefined;
      currentFrame = undefined;
      markIdle();
      void loadModel();
    }
  }

  async function camera() {
    stopSource();
    mediaError = "";
    const token = generation;
    try {
      if (!navigator.mediaDevices?.getUserMedia)
        throw new Error("Camera access requires HTTPS or localhost.");
      const acquired = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30, max: 30 },
        },
        audio: false,
      });
      if (token !== generation || disposed) {
        acquired.getTracks().forEach((track) => track.stop());
        return;
      }
      stream = acquired;
      sourceFps = acquired.getVideoTracks()[0].getSettings().frameRate || 30;
      const activeCapture = await startCameraCapture(
        acquired.getVideoTracks()[0],
        (frame) => {
          if (token !== generation || disposed) {
            frame.bitmap.close();
            return;
          }
          cameraQueue.push({ ...frame, close: () => frame.bitmap.close() });
          queuedFrames = cameraQueue.length;
          scheduleNext();
        },
        (message) => {
          if (token !== generation || disposed) return;
          captureStopped = true;
          stream?.getTracks().forEach((track) => track.stop());
          if (message) mediaError = message;
          if (!cameraQueue.length && !busy) running = false;
        },
      );
      if (token !== generation || disposed) {
        activeCapture.stop();
        return;
      }
      capture = activeCapture;
      sourceKind = "camera";
      sourceName = "Live camera";
      running = true;
      scheduleNext();
    } catch (cause) {
      if (token !== generation) return;
      capture?.stop();
      stream?.getTracks().forEach((track) => track.stop());
      stream = undefined;
      mediaError =
        cause instanceof Error ? cause.message : "Unable to open the camera.";
    }
  }

  async function openFile(file?: File) {
    if (!file) return;
    stopSource();
    mediaError = "";
    const token = generation;
    try {
      if (file.type.startsWith("image/")) {
        const url = URL.createObjectURL(file);
        const image = new Image();
        try {
          image.src = url;
          await image.decode();
        } finally {
          URL.revokeObjectURL(url);
        }
        if (token !== generation || disposed) return;
        still = image;
        sourceKind = "image";
        sourceName = file.name;
        await inferFrame();
      } else if (file.type.startsWith("video/")) {
        const decoded = await openVideoFile(file);
        if (token !== generation || disposed) {
          decoded.close();
          return;
        }
        fileSource = decoded;
        duration = decoded.duration;
        sourceFps = decoded.sourceFps;
        sourceKind = "video";
        sourceName = file.name;
        running = true;
        scheduleNext();
      } else throw new Error("Choose an image or a video file.");
    } catch (cause) {
      if (token === generation)
        mediaError =
          cause instanceof Error ? cause.message : "Unable to read this file.";
    }
  }

  async function sample() {
    stopSource();
    mediaError = "";
    const token = generation;
    try {
      const image = new Image();
      image.src = new URL("sample.jpg", baseUrl).href;
      await image.decode();
      if (token !== generation || disposed) return;
      still = image;
      sourceKind = "image";
      sourceName = "Sample cube";
      await inferFrame();
    } catch {
      mediaError = "The sample image could not be loaded.";
    }
  }

  function dispatch(bitmap: ImageBitmap, timestamp: number, counted: boolean) {
    const drawStart = performance.now();
    presenter.capture(bitmap);
    const sentAt = performance.now();
    const appliedSettings = settings();
    currentFrame = {
      timestamp,
      counted,
      camera: sourceKind === "camera" && counted,
      recording: {
        sourceId: generation,
        sourceName,
        sourceKind: sourceKind as "image" | "video" | "camera",
        sourceFps,
        segment: seekVersion,
        timestampSeconds: timestamp,
        backend,
        settings: appliedSettings,
      },
      sentAt,
      drawMs: sentAt - drawStart,
      acquireMs: counted ? acquireMs : 0,
      pacingMs: counted ? pacingMs : 0,
      intervalMs:
        counted && previousFrameStart ? lastStarted - previousFrameStart : 0,
    };
    send(
      {
        type: "frame",
        bitmap,
        settings: appliedSettings,
        resetTracking: resetTracking || sourceKind === "image",
        preserveAlpha: sourceKind === "image",
      },
      [bitmap],
    );
    resetTracking = false;
    if (counted) previousFrameStart = lastStarted;
  }

  async function inferFrame() {
    if (busy || disposed || sourceKind === "none") return;
    const source = sourceKind === "image" ? still : presenter.snapshot;
    if (!source) {
      scheduleNext();
      return;
    }
    if (status !== "ready") return;
    busy = true;
    const token = generation;
    const activeWorker = worker;
    try {
      const bitmap = await createImageBitmap(source);
      if (token !== generation || activeWorker !== worker || disposed) {
        bitmap.close();
        return;
      }
      dispatch(bitmap, position, false);
    } catch (cause) {
      if (token !== generation) return;
      markIdle();
      mediaError = cause instanceof Error ? cause.message : String(cause);
    }
  }

  async function processNext() {
    timer = 0;
    if (busy || !running || status !== "ready" || disposed) return;
    busy = true;
    // Pace from the start of the whole frame, including source acquisition and
    // drawing. Pacing from post-draw dispatch adds those costs to every interval.
    lastStarted = performance.now();
    pacingMs = scheduledAt ? lastStarted - scheduledAt : 0;
    scheduledAt = 0;
    const acquireStart = lastStarted;
    const token = generation;
    const activeWorker = worker;
    try {
      const frame =
        sourceKind === "video" ? await fileSource?.next() : cameraQueue.shift();
      acquireMs = performance.now() - acquireStart;
      queuedFrames = cameraQueue.length;
      if (token !== generation || activeWorker !== worker || disposed) {
        frame?.bitmap.close();
        return;
      }
      if (!frame) {
        markIdle();
        if (sourceKind === "video") {
          fileEnded = true;
          running = false;
        } else if (captureStopped) running = false;
        return;
      }
      dispatch(frame.bitmap, frame.timestamp, true);
    } catch (cause) {
      if (token !== generation) return;
      markIdle();
      running = false;
      capture?.pause();
      mediaError = cause instanceof Error ? cause.message : String(cause);
    }
  }

  function scheduleNext() {
    if (
      timer ||
      busy ||
      !running ||
      status !== "ready" ||
      disposed ||
      sourceKind === "none" ||
      sourceKind === "image"
    )
      return;
    if (sourceKind === "camera" && !cameraQueue.length) {
      if (captureStopped) running = false;
      return;
    }
    const interval = 1000 / Math.min(30, sourceFps || 30);
    scheduledAt = performance.now();
    const remaining = interval - (scheduledAt - lastStarted);
    if (remaining <= 0) {
      // A completed worker message already gives the browser an asynchronous
      // boundary. An overdue frame needs no additional zero-delay timer task.
      void processNext();
      return;
    }
    timer = window.setTimeout(() => void processNext(), Math.ceil(remaining));
  }

  function drawOverlay(
    ctx: OffscreenCanvasRenderingContext2D,
    width: number,
    height: number,
  ) {
    if (!result) return;
    const sx = width / result.transform.sourceWidth,
      sy = height / result.transform.sourceHeight;
    const radius = Math.max(3.5, width / 230);
    if (showRoi && result.roi) {
      const t = result.transform;
      ctx.strokeStyle = "#c6f69bcc";
      ctx.lineWidth = (1.5 * radius) / 3.5;
      ctx.setLineDash([radius * 2, radius * 2]);
      ctx.strokeRect(
        t.cropX * sx,
        t.cropY * sy,
        t.cropWidth * sx,
        t.cropHeight * sy,
      );
      ctx.setLineDash([]);
    }
  }

  function redrawOverlay() {
    if (result) presenter.redraw(stage, showRoi ? drawOverlay : undefined);
  }
  function downloadDetections() {
    if (!recording.length) return;
    const blob = new Blob(
      [
        JSON.stringify({
          ...recording.toJSON(),
          model: $state.snapshot(manifest),
        }),
      ],
      {
        type: "application/json",
      },
    );
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "cube-detections.json";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function refresh() {
    resetTracking = true;
    if (busy) refreshPending = true;
    else void inferFrame();
  }
  function reacquire() {
    refresh();
  }
  function resetThresholds() {
    if (manifest) {
      thresholds = [...manifest.classThresholds];
      presenceThreshold = manifest.presenceThreshold;
      bboxThreshold = manifest.bbox.threshold;
      roiPadding = manifest.tracking.roiPadding;
    }
    refresh();
  }

  async function seekFile(timestamp: number) {
    if (!fileSource) return;
    const request = ++seekVersion;
    if (!seeking) {
      seekResume = running;
      seeking = true;
    }
    running = false;
    cancelScheduled();
    const token = generation;
    if (busy) await new Promise<void>((resolve) => idleWaiters.push(resolve));
    if (token !== generation || request !== seekVersion || !fileSource) return;
    fileSource.seek(timestamp);
    position = timestamp;
    processedFrames = 0;
    lastResultTime = 0;
    lastStarted = -Infinity;
    previousFrameStart = 0;
    frameTiming = undefined;
    fps = 0;
    fileEnded = false;
    resetTracking = true;
    // Display/process the first selected frame at the seek point even when paused.
    running = true;
    await processNext();
    if (request !== seekVersion) return;
    running = seekResume;
    seeking = false;
  }
  async function togglePlayback() {
    if (running) {
      running = false;
      cancelScheduled();
      capture?.pause();
      return;
    }
    if (sourceKind === "camera" && captureStopped) {
      await camera();
      return;
    }
    if (sourceKind === "video" && fileEnded) {
      await seekFile(0);
    }
    running = true;
    lastResultTime = 0;
    capture?.resume();
    scheduleNext();
  }
  function drop(event: DragEvent) {
    event.preventDefault();
    dragging = false;
    void openFile(event.dataTransfer?.files[0]);
  }
  function cleanup() {
    disposed = true;
    generation++;
    cancelScheduled();
    capture?.stop();
    cameraQueue.clear();
    fileSource?.close();
    worker?.terminate();
    stream?.getTracks().forEach((track) => track.stop());
    markIdle();
  }
  onMount(() => {
    void loadModel();
    return cleanup;
  });
</script>

<svelte:head><title>Live Recognition · Detector</title></svelte:head>
<svelte:window onpagehide={cleanup} />

<div class="shell">
  <header class="topbar">
    <a class="brand" href="./" aria-label="Live Recognition home"
      ><span class="cube-mark" aria-hidden="true"
        ><i></i><i></i><i></i><i></i></span
      >live<span>recognition</span></a
    >
    <span class="build-label">DETECTOR <span class="version">01</span></span>
    <div class="device-badge">
      <span class="status-dot" class:ready={status === "ready"}
      ></span>{status === "ready"
        ? backend === "webgpu"
          ? "WebGPU ready"
          : "WASM CPU ready"
        : status === "loading"
          ? "Loading detector"
          : "Detector unavailable"}
    </div>
  </header>

  <main>
    <div class="page-heading">
      <div>
        <p class="eyebrow">CUBE VISION / LOCAL INFERENCE</p>
        <h1>See every point.</h1>
        <p class="subtitle">
          Landmarks and sticker colors, straight from your camera.
        </p>
      </div>
      <span class="privacy"
        ><span aria-hidden="true">◈</span> Your media stays on this device</span
      >
    </div>
    <div class="workspace">
      <section class="viewer" aria-label="Detection viewer">
        <div class="viewer-toolbar">
          <span class="source-label"
            ><span class="status-dot" class:ready={sourceKind !== "none"}
            ></span>{sourceName}</span
          ><span class="input-badge">POSE {size} × {size}</span>
        </div>
        <div
          class="stage"
          class:dragging
          ondragover={(e) => {
            e.preventDefault();
            dragging = true;
          }}
          ondragleave={() => (dragging = false)}
          ondrop={drop}
          role="region"
          aria-label="Drop an image or video"
        >
          <canvas
            bind:this={stage}
            class:empty={sourceKind === "none"}
            aria-label="Processed source frame"
          ></canvas>
          {#if sourceKind === "none"}
            <div class="empty-state">
              <div class="viewfinder" aria-hidden="true"><span>＋</span></div>
              <p class="eyebrow">READY WHEN YOU ARE</p>
              <h2>Bring your cube into view.</h2>
              <p>Start your camera or drop an image or video here.</p>
              <div class="source-actions">
                <button
                  class="primary"
                  onclick={camera}
                  disabled={status !== "ready"}
                  >Start camera <span aria-hidden="true">↗</span></button
                ><button onclick={() => fileInput.click()}>Open file</button>
              </div>
              <button class="text-button" onclick={sample}
                >Try a sample image <span aria-hidden="true">→</span></button
              >
            </div>
          {/if}
          {#if status === "loading"}<div class="loading-badge" role="status">
              <span class="spinner"></span>{loadingText}
            </div>{/if}
        </div>
        <div class="transport">
          <div class="transport-actions">
            {#if sourceKind !== "none"}<button
                class="small"
                onclick={() => fileInput.click()}>Open file</button
              ><button
                class="small"
                onclick={camera}
                disabled={status !== "ready"}>Camera</button
              >{#if sourceKind === "video" || sourceKind === "camera"}<button
                  class="small"
                  onclick={togglePlayback}
                  >{running ? "Pause" : "Resume"}</button
                >{/if}<button class="small subtle" onclick={stopSource}
                >Clear</button
              >{:else}<span class="muted">Images, videos, or a live camera</span
              >{/if}
          </div>
          <span class="local-label"
            >ON DEVICE <span aria-hidden="true">↗</span></span
          >
        </div>
        {#if sourceKind === "video" && duration > 0}<div class="scrubber">
            <span>{position.toFixed(1)}s</span><input
              aria-label="Video position"
              type="range"
              min="0"
              max={duration}
              step="0.01"
              value={position}
              oninput={(e) => {
                void seekFile(Number(e.currentTarget.value));
              }}
            /><span>{duration.toFixed(1)}s</span>
          </div>{/if}
        {#if sourceKind === "video" || sourceKind === "camera"}
          <div class="frame-policy" aria-label="Frame processing status">
            <span
              >{sourceFps.toFixed(2)} fps source → {Math.min(
                30,
                sourceFps || 30,
              ).toFixed(2)} fps target</span
            ><span
              >{processedFrames} frames processed{sourceKind === "camera"
                ? ` · ${queuedFrames} queued`
                : ""}</span
            >
          </div>
        {/if}
        <div class="pipeline-status" role="status">
          <span class="status-dot" class:ready={!!result?.roi}></span><span
            >{!result
              ? "Cube acquisition → pose tracking"
              : result.roiSource === "none"
                ? "Searching for cube · pose skipped"
                : result.trackingLost
                  ? "Tracking lost · reacquiring next frame"
                  : result.roiSource === "detector"
                    ? "Cube acquired → pose tracking"
                    : "Tracking cube from pose points"}</span
          >{#if result?.cube}<span class="acquisition-score"
              >Last cube acquisition: {result.cube.score.toFixed(2)}</span
            >{/if}
        </div>
        <div class="metrics">
          <div>
            <span class="metric-label"
              >{result?.profile.pose?.preprocessing === "gpu"
                ? "POSE + GPU PREPROCESS"
                : "POSE INFERENCE"}</span
            ><strong
              >{result?.poseMs != null ? result.poseMs.toFixed(1) : "—"}<small
                >ms</small
              ></strong
            >
          </div>
          <div>
            <span class="metric-label">DETECTOR WORKER</span><strong
              >{result ? result.totalMs.toFixed(1) : "—"}<small>ms</small
              ></strong
            >
          </div>
          <div>
            <span class="metric-label">PROCESSED FPS</span><strong
              >{fps ? fps.toFixed(1) : "—"}<small>fps</small></strong
            >
          </div>
          <div>
            <span class="metric-label">POINTS</span><strong
              >{result?.detections.length ?? "—"}<small>detected</small></strong
            >
          </div>
        </div>
        {#if frameTiming}
          <details
            class="frame-policy frame-timing"
            aria-label="Frame timing details"
          >
            <summary>Frame timing</summary>
            <div>
              Frame interval: {frameTiming.intervalMs.toFixed(1)}ms · Source
              wait: {frameTiming.acquireMs.toFixed(1)}ms · Source drawing: {frameTiming.drawMs.toFixed(
                1,
              )}ms · Worker round trip: {frameTiming.roundTripMs.toFixed(1)}ms ·
              Presentation: {frameTiming.overlayMs.toFixed(1)}ms · Pacing wait: {frameTiming.pacingMs.toFixed(
                1,
              )}ms
            </div>
            {#if result?.profile.pose?.preprocessing === "gpu" || result?.profile.bbox?.preprocessing === "gpu"}
              <div>
                GPU preprocessing submission: {(
                  (result?.profile.pose?.gpuSubmitMs ?? 0) +
                  (result?.profile.bbox?.gpuSubmitMs ?? 0)
                ).toFixed(1)}ms. GPU execution is included in the model session
                times.
              </div>
            {:else}
              <div>
                Worker resize: {(
                  (result?.profile.pose?.resizeMs ?? 0) +
                  (result?.profile.bbox?.resizeMs ?? 0)
                ).toFixed(1)}ms · Pixel read: {(
                  (result?.profile.pose?.pixelReadMs ?? 0) +
                  (result?.profile.bbox?.pixelReadMs ?? 0)
                ).toFixed(1)}ms{result?.bboxMs != null
                  ? ` · Cube acquisition: ${result.bboxMs.toFixed(1)}ms`
                  : ""}
              </div>
            {/if}
          </details>
        {/if}
        {#if mediaError}<div class="error" role="alert">{mediaError}</div>{/if}
        {#if status === "error"}<div class="error" role="alert">
            <p>{error}</p>
            <button class="small" onclick={loadModel}>Reload detector</button>
          </div>{/if}
      </section>

      <aside class="controls">
        <section class="control-section">
          <div class="section-heading">
            <h2>Detector</h2>
            <span class="tag">YOLO26m</span>
          </div>
          <p class="hint flow-description">
            {manifest?.bbox.size ?? 640}px cube acquisition → {size}px pose
            tracking
          </p>
          <label class="field-label" for="backend">Inference backend</label>
          <select
            id="backend"
            bind:value={backend}
            disabled={status === "loading"}
            onchange={loadModel}
          >
            <option value="webgpu">WebGPU (GPU)</option><option value="wasm"
              >WASM (CPU)</option
            >
          </select>
          {#if backend === "wasm" && status === "ready"}<p
              class="hint"
              aria-label="CPU runtime configuration"
            >
              SIMD · {runtimeThreads} CPU {runtimeThreads === 1
                ? "thread"
                : "threads"}{runtimeIsolated
                ? ""
                : " · threading unavailable on this host"}
            </p>{/if}
          <label class="field-label" for="resolution"
            >Pose input resolution</label
          ><select
            id="resolution"
            bind:value={size}
            disabled={status === "loading" || !manifest}
            onchange={loadModel}
            >{#each manifest?.variants ?? [{ size: 160 }, { size: 256 }, { size: 288 }] as item}<option
                value={item.size}
                >{item.size} × {item.size}{item.size === 160
                  ? " · Fast"
                  : item.size === 256
                    ? " · Search script"
                    : " · More detail"}</option
              >{/each}</select
          >
          <p class="hint">
            Pose points track the crop between frames. The cube detector runs
            again after a tracking miss.
          </p>
          <button
            class="small"
            onclick={reacquire}
            disabled={sourceKind === "none"}>Reacquire cube</button
          >
        </section>
        <section class="control-section">
          <div class="section-heading">
            <h2>Detected points</h2>
            <span class="muted">{result?.detections.length ?? 0} total</span>
          </div>
          <div class="class-list">
            {#each LABELS as label, id}<div class="class-row">
                <span class="class-name"
                  ><span class="color-dot" style={`--dot: ${COLORS[id]}`}
                  ></span>{label}</span
                ><span class="count">{counts[id] ?? 0}</span>
              </div>{/each}
          </div>
        </section>
        <section class="control-section">
          <h2>Detection data</h2>
          <p class="hint">
            {savedFrames} frame records retained in this tab. Coordinates and scores
            are recorded without drawing points.
          </p>
          <button
            class="small"
            onclick={downloadDetections}
            disabled={!savedFrames}>Download detection data</button
          >
          <label class="toggle">
            <input
              type="checkbox"
              bind:checked={showRoi}
              onchange={redrawOverlay}
            /><span>Tracked ROI outline</span>
          </label>
        </section>
        <details class="control-section">
          <summary>Detection thresholds</summary><label class="threshold"
            ><span>Cube bbox score</span><input
              aria-label="Cube bbox score threshold"
              type="number"
              min="0"
              max="1"
              step="0.05"
              bind:value={bboxThreshold}
              onchange={reacquire}
            /></label
          ><label class="threshold"
            ><span>Initial ROI padding</span><input
              type="number"
              min="0"
              max="2"
              step="0.05"
              bind:value={roiPadding}
              onchange={reacquire}
            /></label
          >
          <p class="hint">
            These filter pose object/class scores. Each object has one keypoint.
          </p>
          {#each LABELS as label, id}<label class="threshold"
              ><span>{label}</span><input
                aria-label={`${label} object score threshold`}
                type="number"
                min="0"
                max="1"
                step="0.05"
                bind:value={thresholds[id]}
                onchange={refresh}
              /></label
            >{/each}<label class="threshold presence-field"
            ><span>Keypoint presence</span><input
              type="number"
              min="0"
              max="1"
              step="0.05"
              bind:value={presenceThreshold}
              onchange={refresh}
            /></label
          >
          <p class="hint">
            Presence is separate from the object score. Sticker presence has a
            0.25 minimum.
          </p>
          <button class="small" onclick={resetThresholds}>Reset defaults</button
          >
        </details>
      </aside>
    </div>
    <footer>
      <span>LIVE RECOGNITION</span>
      <p>Two models. One continuous view of the cube.</p>
      <span>DETECTOR / 01</span>
    </footer>
  </main>
</div>
<input
  class="hidden"
  bind:this={fileInput}
  type="file"
  accept="image/*,video/*"
  onchange={(e) => {
    void openFile(e.currentTarget.files?.[0]);
    e.currentTarget.value = "";
  }}
/>
