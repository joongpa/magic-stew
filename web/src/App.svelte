<script lang="ts">
  import { onMount, untrack } from "svelte";
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
  import { PoseFitter } from "./lib/pose-fitter";
  import type { FittedPose } from "./lib/pose-types";
  import { OrderedStage } from "./lib/ordered-stage";
  import { VideoResultCache, videoResultKey } from "./lib/video-result-cache";
  import type { SavedVideoResult } from "./lib/video-result-cache";
  import { MoveSearch } from "./lib/move-search";
  import type { MoveSearchResult,MoveSearchConfig } from "./lib/move-search-types";
  import { trimRange, timecode } from "./lib/solve-range";
  import { projectedWireframe, poseIsFlagged } from "./lib/reconstruction-view";
  import { buildColorTimeline } from "./lib/cube-colors";
  import { buildProjectedCubeTimeline, projectedCubeAtFrame } from "./lib/projected-cube";
  import { reviewIndex, steppedTimestamp, frameArrow } from "./lib/frame-review";
  import ProjectedCube from "./lib/ProjectedCube.svelte";
  import {createSearchConfig,ADMISSION_CAPACITY,poseWorkerCount,POLICY_VERSION} from "../../pose-fitter/runtime/reconstruction-policy.mjs";
  import {runtimeFingerprints} from './lib/capture-provenance';

  let manifest = $state<ModelManifest>();
  let size = $state(256);
  let backend = $state<"webgpu" | "wasm">("webgpu");
  let runtimeThreads = $state(1);
  let runtimeIsolated = $state(false);
  let detectorIdentity:Extract<WorkerResponse,{type:'ready'}>['models'],detectorPreprocessing='gpu';
  let activeRunId:string|undefined;
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
  let showPose = $state(true);
  let searchEnabled = $state(true), initialScramble = $state("");
  let openingHoldSeconds = $state(0);
  let searchDiagnostics = $state(true);
  let searchBidirectional = $state(true);
  let searchCleanup = $state(true);
  let searchCleanupTolerance = $state(0.02);
  let searchResult = $state<MoveSearchResult>(), searchError = $state("");
  let moveSearch: MoveSearch | undefined;
  let poseFitting = $state(false);
  let poseError = $state("");
  let poseFitter: PoseFitter | undefined;
  let resetPose = true;
  let fitBufferedFrames = $state(0);
  const videoResults = new VideoResultCache();
  let cachedVideoFrames = $state(0), viewingSavedResult = $state(false);
  let presentedTimestamp=$state(0);
  let requestedReviewTimestamp:number|undefined,reviewInProgress=false;
  let videoInputComplete = $state(false);
  let trimming = $state(false), solveStart = $state(0), solveEnd = $state(0);
  let thumbnails = $state<string[]>([]);
  let previewVersion = 0, searchOriginTimestamp = 0;
  let projectedAfterPass = $state(false);
  const reconstructionChoices = $derived(new Map(searchResult?.steps.map(step => [step.frame_index, step.geometry_option_index] as const) ?? []));
  const reconstructionFlags = $derived(new Map(searchResult?.steps.map(step => [step.frame_index,step.pose_flagged === true] as const) ?? []));
  const reconstructionComplete = $derived(videoInputComplete && (!searchEnabled || !!searchError ||
    !!searchResult && ["solved", "already_solved", "exhausted", "state_limit"].includes(searchResult.status)));
  const recording = new DetectionRecording();
  let savedFrames = $state(0);
  let thresholds = $state([0.25, 0.4, 0.25, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]);
  let presenceThreshold = $state(0.05);
  let result = $state<FrameResult>();
  const displayedPoseFlagged = $derived(poseIsFlagged(result?.fittedPose,
    reconstructionFlags.get(result?.fittedPose?.sourceFrame ?? Math.round(presentedTimestamp*sourceFps))));
  $effect(() => {
    // Search can flag the displayed frame after its fitted overlay was drawn.
    // Redraw the saved base immediately, including when no next frame is due.
    displayedPoseFlagged;
    untrack(redrawOverlay);
  });
  const colorTimeline = $derived.by(() => {
    if (!searchResult || !moveSearch) return;
    try { return buildColorTimeline(moveSearch.config.scramble,searchResult.steps); }
    catch { return; }
  });
  const cubeSolved = $derived(searchResult?.status === "solved" || searchResult?.status === "already_solved");
  const reviewRows=$derived.by(()=>{
    cachedVideoFrames;
    if(sourceKind!=="video")return [];
    const rows=videoResults.entries(currentCacheKey());
    const saved=rows.length?rows:(!needsPose()?videoResults.entries(currentCacheKey(true)):[]);
    return saved.filter(row=>row.timestamp>=solveStart-0.00001&&row.timestamp<solveEnd);
  });
  const currentReviewIndex=$derived(reviewIndex(reviewRows,presentedTimestamp));
  const projectedTimeline=$derived(colorTimeline?buildProjectedCubeTimeline(reviewRows,colorTimeline,{
    fps:sourceFps,solved:cubeSolved,steps:searchResult?.steps??[],
    turningStartFrame:moveSearch?.config.turning_start_frame,
  }):[]);
  const coloredProjection=$derived(projectedCubeAtFrame(projectedTimeline,
    result?.fittedPose?.sourceFrame??Math.round(presentedTimestamp*sourceFps)));
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
  let presentationTimes: number[] = [];
  let modelGeneration = 0;
  let currentFrame:
    | {
        snapshot: OffscreenCanvas;
        generation: number;
        fitPose: boolean;
        resetPose: boolean;
        timestamp: number;
        counted: boolean;
        camera: boolean;
        sentAt: number;
        drawMs: number;
        acquireMs: number;
        pacingMs: number;
        intervalMs: number;
        recording: RecordingContext;
        cacheKey: string;
      }
    | undefined;
  type FrameContext = NonNullable<typeof currentFrame>;
  type DetectedFrame = { context: FrameContext; result: FrameResult; detectorCompletedAt: number;
    fitting?: Promise<FittedPose>; fitStartedAtMs?: number; fitCompletedAtMs?: number };
  type CachedFrame = { saved: SavedVideoResult; context: Pick<FrameContext, "snapshot" | "generation" | "timestamp"> };
  // Recovery needs 180ms of future source context, plus the adjacent initial fit.
  const completionQueue = new OrderedStage<DetectedFrame | CachedFrame>(ADMISSION_CAPACITY, completeFrame,
    (packet) => presenter.releaseFrame(packet.context.snapshot),
    (size) => { fitBufferedFrames = size; continuePipeline(); },
    (cause) => {
      mediaError = cause instanceof Error ? cause.message : String(cause);
      running = false;
      capture?.pause();
    });
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
  const needsPose = () => showPose || searchEnabled;
  const inferenceAllowed = () => !moveSearch?.solved && !(sourceKind === "video" && videoInputComplete);
  const displayPose = () => showPose && (sourceKind !== "video" || !trimming);
  $effect(() => {
    if (!projectedAfterPass && sourceKind === "video" && reconstructionComplete && !trimming && result?.fittedPose) {
      projectedAfterPass = true; showPose = true; untrack(redrawOverlay);
    }
  });
  const poseFittingOptions = () => ({scramble:moveSearch?.config.scramble ?? initialScramble.trim(),
    turningStartFrame:moveSearch ? moveSearch.config.turning_start_frame : openingHoldSeconds>0 ? Math.floor((searchOriginTimestamp+openingHoldSeconds)*(sourceFps||30)) : null,
    originSeconds:moveSearch?.config.source_origin_seconds??searchOriginTimestamp,
    endSeconds:sourceKind==="video"?(moveSearch?.config.source_end_seconds??solveEnd):undefined});
  const currentCacheKey = (fitPose = needsPose()) => videoResultKey(size, backend, settings(), fitPose, poseFittingOptions());
  const savedFrameCount = () => Math.max(videoResults.count(currentCacheKey()),
    needsPose() ? 0 : videoResults.count(currentCacheKey(true)));
  function findSavedResult(timestamp: number, tolerance = 0.00001) {
    return videoResults.get(currentCacheKey(), timestamp, tolerance) ??
      (!needsPose() ? videoResults.get(currentCacheKey(true), timestamp, tolerance) : undefined);
  }
  function startMoveSearch() {
    moveSearch?.dispose(); moveSearch = undefined; searchResult = undefined; searchError = "";
    if (!searchEnabled || sourceKind === "none") return;
    if (!initialScramble.trim()) { searchError = "Enter the initial scramble to start move search."; return; }
    if(searchCleanup&&(!Number.isFinite(searchCleanupTolerance)||searchCleanupTolerance<0||searchCleanupTolerance>1)){
      searchError="Cleanup color tolerance must be between 0 and 1.";return;
    }
    const token = generation;
    const config = createSearchConfig({scramble:initialScramble,fps:sourceFps||30,
      openingHoldSeconds,diagnostics:searchDiagnostics,bidirectional:searchBidirectional&&sourceKind==="video",originSeconds:sourceKind==="video"?searchOriginTimestamp:0,
      postprocessIdentityLoops:searchCleanup,postprocessColorTolerance:searchCleanup?searchCleanupTolerance:0.02,
      startSeconds:sourceKind==="video"?solveStart:undefined,endSeconds:sourceKind==="video"?solveEnd:undefined});
    const runId=recording.startRun(token,config,{policyVersion:POLICY_VERSION,appOrigin:location.origin,
      poseWorkers:poseWorkerCount(navigator.hardwareConcurrency),admissionCapacity:ADMISSION_CAPACITY,
      detector:{poseSize:size,backend,preprocessing:detectorPreprocessing,settings:settings(),models:detectorIdentity}});
    activeRunId=runId;
    if(config.diagnostics!==false)void recording.setFingerprints(runId,runtimeFingerprints()).catch(()=>{});
    void persistDiagnostics(runId);
    let endCaptured = false;
    moveSearch = new MoveSearch(baseUrl, config, (value, ended) => {
      if (token !== generation || disposed) return;
      const wasSolved=searchResult?.status==="solved"||searchResult?.status==="already_solved";
      searchResult = value; recording.saveSearch(token, config, value,runId,moveSearch?.actions);
      if(!wasSolved&&(value.status==="solved"||value.status==="already_solved")){
        running=false;seekResume=false;refreshPending=false;cancelScheduled();
        if(sourceKind==="video")fileEnded=true;
        if(sourceKind==="camera"){
          captureStopped=true;capture?.stop();capture=undefined;
          stream?.getTracks().forEach(track=>track.stop());stream=undefined;
          cameraQueue.clear();queuedFrames=0;
        }
        void finishMoveSearch();
      }
      if(videoInputComplete&&((ended&&!endCaptured)||['solved','already_solved','state_limit','exhausted'].includes(value.status)))void persistDiagnostics(runId);
      if(ended)endCaptured = true;
    }, message => { if (token === generation) {
      searchError = message;recording.saveError(runId,message,moveSearch?.actions??[]);void persistDiagnostics(runId);
    } }, sourceKind === "video");
  }
  async function finishMoveSearch() {
    const token = generation, activeSearch = moveSearch;
    await waitForPipelineIdle();
    if (token !== generation || disposed || activeSearch !== moveSearch) return;
    if (sourceKind === "video") videoInputComplete = true;
    if(activeRunId)void persistDiagnostics(activeRunId);
    try { await activeSearch?.end(); } catch (cause) { if (token === generation) searchError = cause instanceof Error ? cause.message : String(cause); }
  }
  async function persistDiagnostics(runId:string) {
    if(!import.meta.env.DEV||!recording.diagnosticsEnabled(runId))return;
    try {
      await recording.ready(runId);
      const response=await fetch(new URL('__diagnostics/run',baseUrl),{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({id:runId,...recording.exportRun(runId),model:$state.snapshot(manifest)})});
      if(!response.ok)console.error('Local diagnostic capture failed',await response.text());
    }catch(cause){console.error('Local diagnostic capture failed',cause);}
  }
  function seedCachedSearch() {
    const rows = videoResults.entries(currentCacheKey(true));
    const target = Math.min(30, sourceFps || 30), origin = searchOriginTimestamp;
    let count = 0, last: SavedVideoResult | undefined;
    for (const row of rows) {
      if (row.timestamp < origin - 0.00001 || row.timestamp >= solveEnd) continue;
      const offset = sourceFps > 30.001 ? Math.ceil(count * sourceFps / target - 0.0001) : count;
      const expected = origin + offset / (sourceFps || 30);
      if (row.timestamp < expected - 0.00001) continue;
      if (row.timestamp > expected + 0.00001 || !row.result.fittedPose) break;
      if(activeRunId)recording.linkCachedFrame(activeRunId,row.timestamp,row.result);
      moveSearch?.submit(row.timestamp, row.result, Promise.resolve(row.result.fittedPose)); count++; last = row;
    }
    return { last, complete: count >= Math.ceil((solveEnd - origin) * target - 0.00001) };
  }
  async function changeMoveSearch() {
    if (!searchEnabled) { moveSearch?.dispose(); moveSearch = undefined; if (!showPose) void flushPose(); return; }
    if (trimming) return;
    if (!initialScramble.trim()) { searchEnabled = false; searchError = "Enter the initial scramble to start move search."; return; }
    const wasRunning = running;
    if (sourceKind === "video") videoInputComplete = false;
    running = false; cancelScheduled();
    await waitForPipelineIdle(); startMoveSearch();
    if (!moveSearch) { running = wasRunning; scheduleNext(); return; }
    if (sourceKind === "video" && fileSource) {
      const {last,complete} = seedCachedSearch();
      if (last && complete) {
        await moveSearch.end(); videoInputComplete = true; running = wasRunning&&!moveSearch.solved; scheduleNext(); return;
      }
      // Newly requested fitting starts at the first missing frame, never at an
      // arbitrary mid-solve state carrying the opening scramble.
      const next = last ? last.timestamp + 1 / Math.min(30, sourceFps || 30) : solveStart;
      seeking = false; running = true; await seekFile(next); scheduleNext();
    } else {
      running = wasRunning; resetPose = true;
      if (sourceKind === "image") void inferFrame(); else scheduleNext();
    }
  }
  async function previewTrim(timestamp: number) {
    const request = ++previewVersion, token = generation, source = fileSource;
    if (!source || !trimming) return;
    try {
      const frame = await source.preview(timestamp);
      if (!frame) return;
      try {
        if (request !== previewVersion || token !== generation || !trimming) return;
        presenter.presentFrame(stage, presenter.captureFrame(frame.bitmap));
        position = frame.timestamp; result = undefined;
      } finally { frame.bitmap.close(); }
    } catch (cause) { if (token === generation) mediaError = cause instanceof Error ? cause.message : String(cause); }
  }
  function changeTrim(edge: "start" | "end", value: number) {
    const range = trimRange({ start: solveStart, end: solveEnd }, edge, value, duration, sourceFps);
    solveStart = range.start; solveEnd = range.end;
    void previewTrim(edge === "start" ? solveStart : Math.max(solveStart, solveEnd - 1 / (sourceFps || 30)));
  }
  async function makeThumbnails(token: number) {
    const source = fileSource; if (!source) return;
    for (let index = 0; index < 10 && token === generation; index++) {
      try {
        const frame = await source.preview(duration * (index + 0.5) / 10, true); if (!frame) continue;
        let url = "";
        try {
          const canvas = new OffscreenCanvas(frame.bitmap.width, frame.bitmap.height);
          canvas.getContext("2d")!.drawImage(frame.bitmap, 0, 0);
          url = URL.createObjectURL(await canvas.convertToBlob({ type: "image/jpeg", quality: 0.75 }));
        } finally { frame.bitmap.close(); }
        if (token !== generation) { URL.revokeObjectURL(url); return; }
        thumbnails[index] = url;
      } catch { /* The range remains usable if an optional thumbnail cannot decode. */ }
    }
  }
  async function beginReconstruction() {
    if (!fileSource || !trimming || status !== "ready") return;
    if (searchEnabled && !initialScramble.trim()) { searchError = "Enter the initial scramble."; return; }
    const token = generation; mediaError = ""; previewVersion++; busy = true;
    try {
      fileSource.seek(solveStart); const frame = await fileSource.next();
      if (!frame) throw new Error("There are no video frames in this solve range.");
      if (token !== generation || disposed) { frame.bitmap.close(); return; }
      if (frame.timestamp >= solveEnd) { frame.bitmap.close(); throw new Error("There are no video frames in this solve range."); }
      searchOriginTimestamp = frame.timestamp; startMoveSearch();
      // Existing completed fits can be prepared immediately, even while the
      // viewer has not yet presented those cached frames again.
      const cached = seedCachedSearch();
      trimming = false; videoInputComplete = false; fileEnded = false; running = true;
      projectedAfterPass = false;
      resetTracking = true; resetPose = true; seekVersion++; processedFrames = 0; fps = 0;
      lastResultTime = 0; previousFrameStart = 0; lastStarted = performance.now(); acquireMs = 0; pacingMs = 0;
      const firstSaved = findSavedResult(frame.timestamp);
      if (cached.complete && firstSaved?.result.fittedPose) {
        presentSavedFrame(frame.bitmap, firstSaved); markIdle(); running = false; fileEnded = true;
        void finishMoveSearch(); return;
      }
      admitFrame(frame);
    } catch (cause) { if (token === generation) { markIdle(); mediaError = cause instanceof Error ? cause.message : String(cause); } }
  }
  async function editSolveRange() {
    if (!reconstructionComplete) return;
    running = false; cancelScheduled(); await waitForPipelineIdle();
    moveSearch?.dispose(); moveSearch = undefined; searchResult = undefined; searchError = "";
    trimming = true; videoInputComplete = false; await previewTrim(solveStart);
  }
  function send(message: WorkerRequest, transfer: Transferable[] = []) {
    worker?.postMessage(message, transfer);
  }
  function markIdle() {
    busy = false;
    for (const resolve of idleWaiters.splice(0)) resolve();
  }
  async function waitForPipelineIdle() {
    if (busy) await new Promise<void>((resolve) => idleWaiters.push(resolve));
    await flushPose();
    await completionQueue.whenIdle();
  }
  async function flushPose() {
    try { await poseFitter?.flush(); }
    catch { /* The ordered frame promise reports the fitting error. */ }
  }
  function continuePipeline() {
    if (disposed) return;
    if(!inferenceAllowed())refreshPending=false;
    if (!busy && (refreshPending || !running || status !== "ready")) void flushPose();
    if (refreshPending && !busy && !completionQueue.size && !seeking && status === "ready") {
      refreshPending = false;
      void inferFrame();
    } else scheduleNext();
  }
  async function completeFrame(packet: DetectedFrame | CachedFrame) {
    if ("saved" in packet) {
      if (packet.context.generation === generation && !disposed)
        presentSavedSnapshot(packet.context.snapshot, packet.saved);
      return;
    }
    const completed = packet.context;
    const token = completed.generation;
    if (token !== generation || disposed) return;
    if (packet.fitting) {
      poseFitting = true;
      try {
        packet.result.fittedPose = await packet.fitting;
        if (token === generation) poseError = "";
      } catch (cause) {
        if (token === generation) {
          poseError = cause instanceof Error ? cause.message : String(cause);
          poseFitter?.dispose();
          poseFitter = undefined;
          resetPose = true;
        }
      } finally {
        if (token === generation) poseFitting = false;
      }
    }
    if (token !== generation || disposed) return;
    const overlayStart = performance.now();
    presenter.presentFrame(stage, completed.snapshot, showRoi || displayPose()
      ? (context, width, height) => drawOverlay(context, width, height, packet.result) : undefined);
    const presentedAtMs = performance.now();
    packet.result.pipeline = {
      detectorStartedAtMs: completed.sentAt,
      detectorCompletedAtMs: packet.detectorCompletedAt,
      fitStartedAtMs: packet.fitStartedAtMs, fitCompletedAtMs: packet.fitCompletedAtMs, presentedAtMs,
      presentationMs: presentedAtMs - overlayStart,
    };
    result = packet.result;
    presentedTimestamp=completed.timestamp;
    viewingSavedResult = false;
    if (completed.recording.sourceKind === "video" && (!completed.fitPose || packet.result.fittedPose))
      cachedVideoFrames = videoResults.put(completed.cacheKey, completed.timestamp, packet.result);
    if (completed.counted || completed.recording.sourceKind === "image" || completed.recording.sourceKind === "video") {
      recording.append(packet.result, completed.recording);
      savedFrames = recording.length;
    }
    if (completed.counted) {
      if (!processedFrames || !lastResultTime) presentationTimes = [];
      processedFrames++;
      position = completed.timestamp;
      // Buffered fits may finish together. Measure a window instead of the
      // instantaneous dispatch rate of two adjacent ready frames.
      presentationTimes.push(presentedAtMs);
      while (presentationTimes.length > 2 && presentationTimes[1] < presentedAtMs - 1000) presentationTimes.shift();
      if (presentationTimes.length > 1) fps = (presentationTimes.length - 1) * 1000 / (presentedAtMs - presentationTimes[0]);
      lastResultTime = presentedAtMs;
      if (completed.camera) capture?.acknowledge();
      frameTiming = {
        acquireMs: completed.acquireMs, drawMs: completed.drawMs,
        roundTripMs: packet.detectorCompletedAt - completed.sentAt,
        overlayMs: presentedAtMs - overlayStart, intervalMs: completed.intervalMs,
        pacingMs: completed.pacingMs,
      };
    }
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
    await waitForPipelineIdle();
    if (disposed || request !== modelGeneration) return;
    if (moveSearch) { moveSearch.dispose(); moveSearch = undefined; searchError = "Model changed. Restart move search to use the new results."; }
    error = "";
    result = undefined;
    refreshPending = false;
    fps = 0;
    resetTracking = true;
    resetPose = true;
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
      currentWorker.onmessage = async ({ data }: MessageEvent<WorkerResponse>) => {
        if (currentWorker !== worker) return;
        if (data.type === "progress")
          loadingText =
            data.fraction < 1
              ? `Downloading ${data.stage.toLowerCase()} · ${Math.round(data.fraction * 100)}%`
              : `${data.stage} · preparing ${backend === "webgpu" ? "GPU" : "CPU"}…`;
        if (data.type === "ready") {
          detectorIdentity=data.models;detectorPreprocessing=data.preprocessing;
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
          if (currentFrame) presenter.releaseFrame(currentFrame.snapshot);
          currentFrame = undefined;
          markIdle();
          void flushPose();
        }
        if (data.type === "result") {
          const completed = currentFrame;
          currentFrame = undefined;
          if (completed) {
            const packet: DetectedFrame = { context: completed, result: data.result,
              detectorCompletedAt: performance.now() };
            if (completed.fitPose) {
              poseFitter ??= new PoseFitter(baseUrl);
              packet.fitStartedAtMs = performance.now();
              try {
                packet.fitting = poseFitter.fit(data.result, completed.resetPose,
                  completed.timestamp, completed.recording.sourceFps, poseFittingOptions()).finally(() => {
                    packet.fitCompletedAtMs = performance.now();
                  });
              } catch (cause) { packet.fitting = Promise.reject(cause); }
              // Admission continues while the ordered presenter awaits an earlier fit.
              void packet.fitting.catch(() => {});
              moveSearch?.submit(completed.timestamp, data.result, packet.fitting);
            }
            completionQueue.enqueue(packet);
          }
          // GPU admission is released at detector completion, independently of
          // the ordered fitting/presentation stage.
          markIdle();
          continuePipeline();
        }

      };
      currentWorker.onerror = () => {
        status = "error";
        error = "The detector worker could not start. Reload the detector.";
        capture?.pause();
        if (currentFrame) presenter.releaseFrame(currentFrame.snapshot);
        currentFrame = undefined;
        markIdle();
        void flushPose();
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
    activeRunId=undefined;
    generation++;
    seekVersion++;
    seeking = false;
    running = false;
    refreshPending = false;
    cancelScheduled();
    resetTracking = true;
    resetPose = true;
    poseFitter?.dispose();
    poseFitter = undefined;
    moveSearch?.dispose(); moveSearch = undefined; searchResult = undefined; searchError = "";
    completionQueue.clear();
    poseFitting = false;
    poseError = "";
    capture?.stop();
    capture = undefined;
    stream?.getTracks().forEach((track) => track.stop());
    stream = undefined;
    fileSource?.close();
    fileSource = undefined;
    videoResults.clear(); cachedVideoFrames = 0; viewingSavedResult = false;
    presentedTimestamp=0;requestedReviewTimestamp=undefined;reviewInProgress=false;
    videoInputComplete = false;
    projectedAfterPass = false;
    trimming = false; solveStart = 0; solveEnd = 0; previewVersion++; searchOriginTimestamp = 0;
    thumbnails.forEach(url => URL.revokeObjectURL(url)); thumbnails = [];
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
      startMoveSearch();
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
    recording.setSourceFile(token,file);
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
        startMoveSearch();
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
        solveStart = 0; solveEnd = duration; trimming = true;
        await previewTrim(0); void makeThumbnails(token);
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
    // Recheck after asynchronous decoding/bitmap creation: search can finish
    // while a frame is being acquired, before it reaches the neural worker.
    if(!inferenceAllowed()){
      bitmap.close();markIdle();continuePipeline();return;
    }
    const drawStart = performance.now();
    const snapshot = presenter.captureFrame(bitmap);
    const sentAt = performance.now();
    const appliedSettings = settings();
    currentFrame = {
      snapshot,
      generation,
      fitPose: needsPose(),
      resetPose,
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
        runId:activeRunId,
      },
      cacheKey: videoResultKey(size, backend, appliedSettings, needsPose(), poseFittingOptions()),
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
    if (needsPose()) resetPose = false;
    if (counted) previousFrameStart = lastStarted;
  }

  async function inferFrame() {
    if (trimming || !inferenceAllowed()) { refreshPending=false;return; }
    if (busy || completionQueue.size || disposed || sourceKind === "none") {
      if (!disposed && sourceKind !== "none") refreshPending = true;
      return;
    }
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
    if (busy || !completionQueue.canAdmit || !running || status !== "ready" || disposed) return;
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
      if(!running&&!inferenceAllowed()){
        frame?.bitmap.close();markIdle();return;
      }
      if (frame && sourceKind === "video" && frame.timestamp >= solveEnd) {
        frame.bitmap.close(); markIdle(); fileEnded = true; running = false; void finishMoveSearch(); return;
      }
      if (!frame) {
        markIdle();
        if (sourceKind === "video") {
          fileEnded = true;
          running = false;
        } else if (captureStopped) running = false;
        if (!running) void finishMoveSearch();
        return;
      }
      admitFrame(frame);
    } catch (cause) {
      if (token !== generation) return;
      markIdle();
      running = false;
      capture?.pause();
      mediaError = cause instanceof Error ? cause.message : String(cause);
    }
  }
  function admitFrame(frame: SourceFrame) {
      const saved = sourceKind === "video" ? findSavedResult(frame.timestamp) : undefined;
      if (saved) {
        if (saved.result.fittedPose) moveSearch?.submit(saved.timestamp, saved.result, Promise.resolve(saved.result.fittedPose));
        // A cached island closes the preceding numerical segment. Present it
        // through the same queue so it cannot overtake an earlier pending fit.
        void flushPose(); resetTracking = true; resetPose = true;
        const snapshot = presenter.captureFrame(frame.bitmap);
        frame.bitmap.close();
        completionQueue.enqueue({ saved, context: { snapshot, generation, timestamp: frame.timestamp } });
        markIdle();
        continuePipeline();
      } else if(!inferenceAllowed()){
        // Completed-video playback is decoding only. An uncached frame (for
        // example after an early solution) must never restart neural inference.
        try {
          presenter.presentFrame(stage,presenter.captureFrame(frame.bitmap));
          result=undefined;position=frame.timestamp;presentedTimestamp=frame.timestamp;
          viewingSavedResult=false;frameTiming=undefined;
        } finally {frame.bitmap.close();markIdle();}
        continuePipeline();
      } else dispatch(frame.bitmap, frame.timestamp, true);
  }

  function presentSavedFrame(bitmap: ImageBitmap, saved: SavedVideoResult) {
    try {
      const snapshot = presenter.captureFrame(bitmap);
      presentSavedSnapshot(snapshot, saved);
    } finally { bitmap.close(); }
  }
  function presentSavedSnapshot(snapshot: OffscreenCanvas, saved: SavedVideoResult) {
    presenter.presentFrame(stage, snapshot, showRoi || displayPose()
      ? (ctx, width, height) => drawOverlay(ctx, width, height, saved.result) : undefined);
    result = saved.result; position = saved.timestamp; viewingSavedResult = true;
    presentedTimestamp=saved.timestamp;
    cachedVideoFrames = savedFrameCount(); frameTiming = undefined;
  }

  function scheduleNext() {
    if (
      timer ||
      busy ||
      !completionQueue.canAdmit ||
      refreshPending ||
      !running ||
      status !== "ready" ||
      disposed ||
      trimming ||
      sourceKind === "none" ||
      sourceKind === "image"
    )
      return;
    if (sourceKind === "camera" && !cameraQueue.length) {
      if (captureStopped) { running = false; void finishMoveSearch(); }
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

  function projectionForFrame(frame?: FrameResult): [number, number][][] {
    if (!frame?.fittedPose?.pose) return [];
    const sourceFrame = frame.fittedPose.sourceFrame ?? Math.round(position * sourceFps);
    if (poseIsFlagged(frame.fittedPose,reconstructionFlags.get(sourceFrame))) return [];
    const staticEnd = moveSearch?.config.turning_start_frame;
    return projectedWireframe(frame.fittedPose, reconstructionChoices.get(sourceFrame),
      staticEnd != null && sourceFrame <= staticEnd);
  }
  function drawOverlay(
    ctx: OffscreenCanvasRenderingContext2D,
    width: number,
    height: number,
    frame = result,
  ) {
    if (!frame) return;
    const sx = width / frame.transform.sourceWidth,
      sy = height / frame.transform.sourceHeight;
    const radius = Math.max(3.5, width / 230);
    if (showRoi && frame.roi) {
      const t = frame.transform;
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
    if (displayPose() && frame.fittedPose?.pose) {
      ctx.strokeStyle = "#64f6dd";
      ctx.lineWidth = Math.max(1.5, width / 650);
      ctx.lineJoin = "round";
      const polygons = projectionForFrame(frame);
      for (const polygon of polygons) {
        if (!polygon.length) continue;
        ctx.beginPath();
        ctx.moveTo(polygon[0][0] * sx, polygon[0][1] * sy);
        for (const point of polygon.slice(1)) ctx.lineTo(point[0] * sx, point[1] * sy);
        ctx.closePath();
        ctx.stroke();
      }
    }
  }

  function redrawOverlay() {
    if (result) presenter.redraw(stage, showRoi || displayPose() ? drawOverlay : undefined);
  }
  function changePoseDisplay() {
    poseError = "";
    cachedVideoFrames = savedFrameCount();
    if (!needsPose()) void flushPose();
    const saved = sourceKind === "video" ? findSavedResult(position) : undefined;
    if (saved) { result = saved.result; viewingSavedResult = true; }
    redrawOverlay();
    if (showPose && !result?.fittedPose) {
      resetPose = true;
      if (busy || completionQueue.size) refreshPending = true;
      else void inferFrame();
      continuePipeline();
    }
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
    if (moveSearch) { moveSearch.dispose(); moveSearch = undefined; searchError = "Detection settings changed. Restart move search to use the new results."; }
    resetTracking = true;
    resetPose = true;
    cachedVideoFrames = savedFrameCount();
    if (busy || completionQueue.size) refreshPending = true;
    else void inferFrame();
    continuePipeline();
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

  async function requestReview(timestamp:number) {
    if(!reconstructionComplete||trimming)return;
    requestedReviewTimestamp=timestamp;
    running=false;seekResume=false;cancelScheduled();
    if(reviewInProgress)return;
    reviewInProgress=true;const token=generation;
    try {
      while(requestedReviewTimestamp!=null&&token===generation){
        const target:number=requestedReviewTimestamp;
        await seekFile(target);
        if(requestedReviewTimestamp===target)requestedReviewTimestamp=undefined;
      }
    } finally { if(token===generation)reviewInProgress=false; }
  }
  function stepReview(direction:-1|1) {
    if(sourceKind!=="video"||!reconstructionComplete||trimming)return;
    const timestamp=steppedTimestamp(reviewRows,requestedReviewTimestamp??presentedTimestamp,direction);
    if(timestamp!=null)void requestReview(timestamp);
  }
  function reviewKeydown(event:KeyboardEvent) {
    if(sourceKind!=="video"||!reconstructionComplete||trimming||event.defaultPrevented)return;
    const direction=frameArrow(event,event.target instanceof Element?event.target:null);
    if(direction){event.preventDefault();stepReview(direction);}
  }
  async function seekFile(timestamp: number) {
    if (!fileSource) return;
    timestamp = Math.max(solveStart, Math.min(timestamp, solveEnd - 1 / (sourceFps || 30)));
    const request = ++seekVersion;
    if (!seeking) {
      seekResume = running;
      seeking = true;
    }
    running = false;
    cancelScheduled();
    const token = generation;
    await waitForPipelineIdle();
    if (token !== generation || request !== seekVersion || !fileSource) return;
    refreshPending = false;
    const saved = findSavedResult(timestamp, 0.5 / Math.min(30, sourceFps || 30) + 0.001);
    fileSource.seek(saved?.timestamp ?? timestamp);
    position = saved?.timestamp ?? timestamp;
    processedFrames = 0;
    lastResultTime = 0;
    lastStarted = -Infinity;
    previousFrameStart = 0;
    frameTiming = undefined;
    fps = 0;
    fileEnded = false;
    resetTracking = true;
    resetPose = true;
    if (saved) {
      busy = true;
      try {
        const frame = await fileSource.next();
        if (token !== generation || request !== seekVersion || !fileSource) { frame?.bitmap.close(); return; }
        if (frame && Math.abs(frame.timestamp - saved.timestamp) < 0.00001) {
          if (saved.result.fittedPose) moveSearch?.submit(saved.timestamp, saved.result, Promise.resolve(saved.result.fittedPose));
          presentSavedFrame(frame.bitmap, saved);
        }
        else {
          frame?.bitmap.close();
          fileSource.seek(timestamp);
          throw new Error("The saved frame could not be decoded at its original timestamp.");
        }
      } catch (cause) {
        if (token === generation && request === seekVersion) mediaError = cause instanceof Error ? cause.message : String(cause);
      } finally { if (token === generation) markIdle(); }
      if (token !== generation || request !== seekVersion) return;
      running = seekResume; seeking = false; scheduleNext();
      return;
    }
    // Display/process the first selected frame at the seek point even when paused.
    running = true;
    await processNext();
    if (request !== seekVersion) return;
    running = seekResume;
    seeking = false;
  }
  async function togglePlayback() {
    if (sourceKind === "video" && !reconstructionComplete) return;
    if (running) {
      running = false;
      cancelScheduled();
      capture?.pause();
      if (!busy) void flushPose();
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
    previewVersion++; thumbnails.forEach(url => URL.revokeObjectURL(url));
    capture?.stop();
    cameraQueue.clear();
    fileSource?.close();
    worker?.terminate();
    poseFitter?.dispose();
    moveSearch?.dispose();
    completionQueue.clear();
    stream?.getTracks().forEach((track) => track.stop());
    markIdle();
  }
  onMount(() => {
    void loadModel();
    return cleanup;
  });
</script>

<svelte:head><title>Live Recognition · Detector</title></svelte:head>
<svelte:window onpagehide={cleanup} onkeydown={reviewKeydown} />

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
        <div class="reconstruction-pair" class:with-projection={sourceKind === "video" && reconstructionComplete && !trimming}>
        <div
          class="stage"
          class:video-view={sourceKind === "video"}
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
        {#if sourceKind === "video" && reconstructionComplete && !trimming}
          <ProjectedCube pose={coloredProjection} sourceTimestamp={presentedTimestamp} timestamp={timecode(presentedTimestamp)} solved={cubeSolved}
            timeline={projectedTimeline} steps={searchResult?.steps??[]} {sourceFps}
            sourceFrame={result?.fittedPose?.sourceFrame??Math.round(presentedTimestamp*sourceFps)} />
        {/if}
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
              >{#if sourceKind === "camera" || sourceKind === "video" && reconstructionComplete && !trimming}<button
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
              {#if sourceKind === "video" && reconstructionComplete && !trimming}<button class="small" onclick={editSolveRange}>Adjust range</button>{/if}
        </div>
        {#if sourceKind === "video" && duration > 0 && reconstructionComplete && !trimming}<div class="scrubber">
            <span>{position.toFixed(1)}s</span><input
              aria-label="Video position"
              type="range"
              min={solveStart}
              max={solveEnd}
              step="0.01"
              value={position}
              oninput={(e) => {
                void requestReview(Number(e.currentTarget.value));
              }}
            /><span>{solveEnd.toFixed(1)}s</span>
          </div>
          <div class="frame-review" aria-label="Frame review">
            <button class="small" aria-label="Previous frame" title="Previous frame (←)" disabled={currentReviewIndex<=0} onclick={()=>stepReview(-1)}>←</button>
            <span aria-label="Frame counter">Frame {currentReviewIndex+1} / {reviewRows.length}
              <span class="source-frame">· Source frame {result?.fittedPose?.sourceFrame??Math.round(presentedTimestamp*sourceFps)}</span></span>
            <button class="small" aria-label="Next frame" title="Next frame (→)" disabled={currentReviewIndex>=reviewRows.length-1} onclick={()=>stepReview(1)}>→</button>
            <span class="frame-key-hint">← / → to step</span>
          </div>{/if}
        {#if sourceKind === "video" && trimming}<section class="solve-setup" aria-label="Solve setup">
          <div class="trim-heading"><strong>Trim your solve</strong><span>{timecode(solveEnd - solveStart)} selected</span></div>
          <div class="trim-timeline" role="group" aria-label="Solve range">
            <div class="trim-thumbnails">{#each Array(10) as _, index}<div>{#if thumbnails[index]}<img src={thumbnails[index]} alt="" />{/if}</div>{/each}</div>
            <div class="trim-shade" style:left="0" style:width={`${duration ? solveStart / duration * 100 : 0}%`}></div>
            <div class="trim-shade" style:right="0" style:width={`${duration ? (1 - solveEnd / duration) * 100 : 0}%`}></div>
            <div class="trim-selection" style:left={`${duration ? solveStart / duration * 100 : 0}%`} style:right={`${duration ? (1 - solveEnd / duration) * 100 : 0}%`}></div>
            <input class="trim-handle" type="range" aria-label="Solve start" min="0" max={duration} step={1 / (sourceFps || 30)} value={solveStart} oninput={e => changeTrim("start", Number(e.currentTarget.value))} />
            <input class="trim-handle trim-end" type="range" aria-label="Solve end" min="0" max={duration} step={1 / (sourceFps || 30)} value={solveEnd} oninput={e => changeTrim("end", Number(e.currentTarget.value))} />
          </div>
          <div class="trim-times"><label>Start <input aria-label="Solve start time (seconds)" type="number" min="0" max={solveEnd} step="0.01" value={solveStart.toFixed(2)} onchange={e => changeTrim("start", Number(e.currentTarget.value))} /></label>
            <span>{timecode(solveStart)} → {timecode(solveEnd)}</span>
            <label>End <input aria-label="Solve end time (seconds)" type="number" min={solveStart} max={duration} step="0.01" value={solveEnd.toFixed(2)} onchange={e => changeTrim("end", Number(e.currentTarget.value))} /></label></div>
          <label class="field-label" for="solve-scramble">Initial scramble</label>
          <input id="solve-scramble" aria-label="Initial scramble" bind:value={initialScramble} placeholder="R U R' U'…" />
          <div class="trim-actions"><p class="hint">Include a clear cube before the first move.</p>
            <button class="primary" disabled={status !== "ready" || searchEnabled && !initialScramble.trim()} onclick={beginReconstruction}>Reconstruct selected range</button></div>
        </section>{/if}
        {#if sourceKind === "video" && !trimming && !reconstructionComplete}<p class="hint" role="status" aria-label="Reconstruction progress">
          Reconstructing first pass… Playback and scrubbing will be available when it finishes.
        </p>{/if}
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
                : ""}{fitBufferedFrames ? ` · ${fitBufferedFrames} fitting/presentation buffered` : ""}</span
            >
            {#if sourceKind === "video"}<span>{cachedVideoFrames} saved frames{viewingSavedResult ? " · Showing saved result" : ""}</span>{/if}
          </div>
        {/if}
        <div class="pipeline-status" role="status">
          <span class="status-dot" class:ready={!!result?.roi}></span><span
            >{!result
              ? sourceKind==="video"&&videoInputComplete?"Reviewing video · no saved detections for this frame":"Cube acquisition → pose tracking"
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
          <h2>Fitted pose</h2>
          <label class="toggle">
            <input type="checkbox" bind:checked={showPose} onchange={changePoseDisplay} />
            <span>Show fitted pose</span>
          </label>
          <p class="hint">{sourceKind === "video" ? "Show the fitted wireframe on each processed frame, including the first pass. The colored cube appears when reconstruction finishes." : "Fit and display a cube wireframe. Points remain hidden."}</p>
          {#if showPose}
            <p class="hint" role="status" aria-label="Pose fitting status">
              {poseError || (displayedPoseFlagged ? "Flagged pose hidden" : result?.fittedPose?.message) || (poseFitting ? "Fitting pose…" : "Waiting for a cube")}
              {#if result?.fittedPose}
                · Fit {result.fittedPose.elapsedMs.toFixed(1)}ms
                · Draw {(result.pipeline?.presentationMs ?? 0).toFixed(1)}ms
                · {result.fittedPose.bank.length} options
              {/if}
            </p>
          {/if}
        </section>
        <section class="control-section">
          <h2>Move search</h2>
          {#if !trimming}<label class="field-label" for="initial-scramble">Initial scramble</label>
          <input id="initial-scramble" aria-label="Search scramble" bind:value={initialScramble} placeholder="R U R' U'…" />{/if}
          <label class="field-label" for="opening-hold">Opening hold (seconds)</label>
          <input id="opening-hold" aria-label="Opening hold (seconds)" type="number" min="0" step="0.01" bind:value={openingHoldSeconds} />
          <label class="toggle"><input type="checkbox" bind:checked={searchEnabled} onchange={changeMoveSearch} />
            <span>Reconstruct moves</span></label>
          <label class="toggle"><input type="checkbox" bind:checked={searchBidirectional} onchange={changeMoveSearch} />
            <span>Bidirectional video search</span></label>
          <label class="toggle"><input type="checkbox" bind:checked={searchCleanup} onchange={changeMoveSearch} />
            <span>Remove move/undo artifacts</span></label>
          <label class="field-label" for="cleanup-tolerance">Cleanup color tolerance</label>
          <input id="cleanup-tolerance" aria-label="Cleanup color tolerance" type="number" min="0" max="1" step="0.005"
            disabled={!searchCleanup} bind:value={searchCleanupTolerance} />
          <p class="hint">After solving, remove state-return sections when holding raises the segment's average color error by at most this amount. Applies to the next run.</p>
          <label class="toggle"><input type="checkbox" bind:checked={searchDiagnostics} />
            <span>Record search diagnostics</span></label>
          <p class="hint">Turn off for faster search. Applies to the next run.</p>
          <p class="hint">{searchBidirectional ? "Video search starts after all poses are fitted and searches from both ends. Camera search stays forward." : "Forward search starts as poses arrive."} Shared 54 TPS limit · five fits plus flat.</p>
          <button class="small" disabled={!searchEnabled || sourceKind === "none" || trimming} onclick={changeMoveSearch}>Restart move search</button>
          <p class="hint" role="status" aria-label="Move search status">
            {searchError || (searchResult ? ({running:"Searching",waiting_for_frames:"Waiting for fitted frames",solved:"Solved",already_solved:"Already solved",exhausted:"Search exhausted",state_limit:"Search state limit reached"}[searchResult.status]) : searchEnabled ? "Waiting for a source" : "Move search is off")}
            {#if searchResult?.bidirectional_fallback}<span> · {searchResult.bidirectional_fallback}</span>{/if}
            {#if searchResult} · {searchResult.frames_prepared} prepared
              {#if searchResult.search_direction === "bidirectional"} · {(searchResult.stats.forward_expanded ?? 0).toLocaleString()} forward · {(searchResult.stats.backward_expanded ?? 0).toLocaleString()} backward
              {:else} · {searchResult.stats.furthest_frames} searched{/if}
              · {searchResult.stats.evaluated_states.toLocaleString()} states
            {/if}
          </p>
          {#if searchResult?.steps.length}<p class="hint" aria-label="Move search path">
            {searchResult.status === "solved" ? "Solution" : "Current candidate"}: {searchResult.steps.flatMap(step => step.state_moves).join(" ") || "Hold"}
          </p>{/if}
          {#if searchResult?.postprocess}<p class="hint" role="status" aria-label="Solution cleanup">
            {#if searchResult.postprocess.error}Cleanup skipped: {searchResult.postprocess.error}
            {:else if searchResult.postprocess.removed_moves}
              {#if searchResult.postprocess.removed_sections}Removed {searchResult.postprocess.removed_sections} move/undo {searchResult.postprocess.removed_sections===1?"section":"sections"}.{/if}
              {#if searchResult.postprocess.simplified_frames}Simplified moves in {searchResult.postprocess.simplified_frames} {searchResult.postprocess.simplified_frames===1?"frame":"frames"}.{/if}
            {:else}No removable move/undo sections found.{/if}
          </p>{/if}
          {#if searchResult?.unprocessed_steps?.length}<details>
            <summary>Original move list</summary>
            <p class="hint">{searchResult.unprocessed_steps.flatMap(step=>step.state_moves).join(" ")}</p>
          </details>{/if}
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
