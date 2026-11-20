<script lang="ts">
  import { onMount, untrack } from "svelte";
  import { FramePresenter } from "./lib/frame-presenter";
  import { compactDetectionCapture } from "./lib/diagnostic-capture";
  import { DetectionRecording } from "./lib/detection-recording";
  import type { RecordingContext } from "./lib/detection-recording";
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
  import { PLAYBACK_SPEEDS, playbackTiming } from "./lib/playback-timing";
  import type { PlaybackSpeed } from "./lib/playback-timing";
  import TpsGraph from "./lib/TpsGraph.svelte";
  import ProjectedCube from "./lib/ProjectedCube.svelte";
  import {createSearchConfig,ADMISSION_CAPACITY,poseWorkerCount,POLICY_VERSION,CAMERA_CLOCK_HZ} from "../../pose-fitter/runtime/reconstruction-policy.mjs";
  import {runtimeFingerprints} from './lib/capture-provenance';
  import CubeTimer from './lib/CubeTimer.svelte';
  import LatestTimerReplay from './lib/LatestTimerReplay.svelte';
  import { TimerReconstruction } from './lib/timer-reconstruction';
  import type { TimerReconstructionResult } from './lib/cube-timer';
  import { TimerDiagnostics } from './lib/timer-diagnostics';
  import { DiagnosticWriter } from './lib/diagnostic-writer';
  import type {CameraCaptureMetrics} from './lib/camera-metrics';
  import {TimerReplayStore,replayReconstruction,type TimerReplay} from './lib/timer-replay';
  import {LocalReplayStorage,TimerReplayCache} from './lib/timer-replay-cache';
  import {TimerCaptureGate} from './lib/timer-capture';
  import {workspacePath,workspaceFromPath,type WorkspaceTab} from './lib/workspace-route';
  import {offlineCacheReady} from './lib/offline-cache';
  import {hasWebGpuAdapter} from './lib/runtime-config';

  const workspaceBase=new URL(import.meta.env.BASE_URL,location.origin).pathname;
  let activeTab = $state<WorkspaceTab>(workspaceFromPath(location.pathname,workspaceBase));
  let timerBusy = $state(false);
  let timerCamera = $state(false);
  let cameraClockOriginMs = $state<number>();
  let timerReconstructions = $state<Record<string, TimerReconstructionResult>>({});
  let timerReplays=$state.raw<Record<string,TimerReplay>>({});
  let latestTimerSolveId=$state<string>(),latestReplayPosition=$state<number>();
  let latestTimerReplay=$state.raw<TimerReplay>();
  let latestReplaySeek=$state<{timestamp:number;revision:number}>(),latestReplaySeekRevision=0;
  const latestReplayReconstruction=$derived(latestTimerSolveId?timerReconstructions[latestTimerSolveId]??latestTimerReplay?.reconstruction:undefined);
  function seekLatestTimerReplay(id:string,timestamp:number){
    if(id!==latestTimerSolveId||!latestTimerReplay)return false;
    latestReplaySeek={timestamp,revision:++latestReplaySeekRevision};return true;
  }
  let replayLoading=$state(false),replayStorageError=$state('');
  const timerReplayCache=new TimerReplayCache(new LocalReplayStorage(),(clip,loading=false,error='')=>{
    timerReplays=clip?{[clip.id]:clip}:{};replayLoading=loading;replayStorageError=error;
  });
  const timerReplayStore=new TimerReplayStore(clip=>{
    timerReplayCache.publish(clip);
    if(clip.id===latestTimerSolveId)latestTimerReplay=clip;
  });
  const timerCaptureGate=new TimerCaptureGate();
  let timerCapturePhase=$state<'idle'|'capturing'|'draining'>('idle');
  const timerReconstruction = new TimerReconstruction(
    (config, onResult, onError, id) => {
      const search = new MoveSearch(baseUrl, config, (value, ended) => {
        timerDiagnostics.search(id, config, value, search.actions, ended);
        onResult(value, ended);
      }, onError);
      return search;
    },
    (id, value) => { timerReconstructions = { ...timerReconstructions, [id]: timerDiagnostics.update(id, value) }; },
    id=>{
      const playback=replayReconstruction($state.snapshot(timerReconstructions[id]));
      timerReplayCache.finish(id,playback);timerReconstructions={...timerReconstructions,[id]:playback};
      void timerDiagnostics.finish(id).then(()=>{savedFrames=recording.length;}).catch(error=>{mediaError=String(error);});
    });
  function selectTab(tab:WorkspaceTab,pushHistory=true) {
    if (timerBusy || tab === activeTab) return;
    if (tab === 'timer') {
      if (sourceKind !== 'none' && sourceKind !== 'camera') stopSource();
      moveSearch?.dispose(); moveSearch = undefined; searchResult = undefined; searchError = '';
      activeRunId = undefined;
      if (sourceKind === 'camera') timerCamera = true;
    }
    if (tab === 'detector' && sourceKind === 'camera') stopSource();
    activeTab = tab;
    if(pushHistory)history.pushState(null,'',workspacePath(tab,workspaceBase));
    document.getElementById('workspace-panel')?.focus({ preventScroll: true });
  }
  function navigateTab(event:MouseEvent,tab:WorkspaceTab){
    if(timerBusy){event.preventDefault();return;}
    if(event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;
    event.preventDefault();selectTab(tab);
  }
  function followHistory(){
    if(timerBusy){history.replaceState(history.state,'',workspacePath(activeTab,workspaceBase));return;}
    selectTab(workspaceFromPath(location.pathname,workspaceBase),false);
  }
  function startTimerSolve(id: string, scramble: string, clockMs: number) {
    if (cameraClockOriginMs === undefined) return;
    const config = timerReconstruction.start(id, scramble, (clockMs - cameraClockOriginMs) / 1000, sourceFps || 30, searchDiagnostics);
    latestTimerSolveId=id;latestTimerReplay=undefined;latestReplayPosition=undefined;latestReplaySeek=undefined;
    timerReplayCache.begin(id);
    timerReplayStore.start(id,config.requested_start_seconds!,config.fps);
    timerCaptureGate.start(id,config.requested_start_seconds!);timerCapturePhase='capturing';capture?.resume();
    const runId = timerDiagnostics.start(id, generation, config, {
      policyVersion: POLICY_VERSION, appOrigin: location.origin, poseWorkers: poseWorkerCount(navigator.hardwareConcurrency),
      admissionCapacity: ADMISSION_CAPACITY,
      detector: { poseSize: size, backend, preprocessing: detectorPreprocessing, settings: settings(), models: detectorIdentity },
      timer: { solveId: id, startClockMs: clockMs, cameraClockOriginMs },
    });
    if (config.diagnostics !== false) void recording.setFingerprints(runId, runtimeFingerprints()).catch(() => {});
    timerReconstructions = { ...timerReconstructions, [id]: { status: 'collecting', diagnosticRunId: runId } };
  }
  function stopTimerSolve(id: string, clockMs: number) {
    if (cameraClockOriginMs !== undefined) {
      const timestamp=(clockMs-cameraClockOriginMs)/1000;
      timerReconstruction.stop(id,timestamp);timerReplayStore.stop(id,timestamp);
      timerCaptureGate.stop(id,timestamp);
    }
  }
  function deleteTimerSolve(id:string){
    if(latestTimerSolveId===id){latestTimerSolveId=undefined;latestTimerReplay=undefined;latestReplayPosition=undefined;latestReplaySeek=undefined;}
    timerReconstruction.cancelSolve(id);timerReplayStore.remove(id);timerReplayCache.remove(id);
    void timerDiagnostics.forget(id).catch(error=>{mediaError=String(error);});
    const remaining={...timerReconstructions};delete remaining[id];timerReconstructions=remaining;
  }
  function clearTimerSession(ids:string[]){
    latestTimerSolveId=undefined;latestTimerReplay=undefined;latestReplayPosition=undefined;latestReplaySeek=undefined;
    timerReconstruction.cancel('Session cleared');timerReplayStore.clear();timerReplayCache.clear();
    for(const id of ids){timerReplayCache.remove(id);void timerDiagnostics.forget(id).catch(error=>{mediaError=String(error);});}
    timerReconstructions={};
  }

  let manifest = $state<ModelManifest>();
  const size = 256;
  const backend = "webgpu";
  let detectorIdentity:Extract<WorkerResponse,{type:'ready'}>['models'],detectorPreprocessing='gpu';
  let activeRunId:string|undefined;
  let status = $state<"loading" | "ready" | "error">("loading");
  let loadingText = $state("Checking WebGPU…");
  let error = $state("");
  let mediaError = $state("");
  let sourceName = $state("No source selected");
  let sourceKind = $state<"none" | "image" | "video" | "camera">("none");
  let running = $state(false);
  let playbackSpeed=$state<PlaybackSpeed>(1);
  let processedFrames = $state(0);
  let queuedFrames = $state(0);
  let sourceFps = $state(0);
  let busy = false,
    refreshPending = false,
    resetTracking = true;
  let bboxThreshold = $state(0.25),
    roiPadding = $state(0.4);
  let showRoi = $state(false);
  const showPose = true;
  let searchEnabled = $state(true), initialScramble = $state("");
  let openingHoldSeconds = $state(0);
  let searchDiagnostics = $state(true);
  let searchBidirectional = $state(true);
  let searchCleanup = $state(true);
  let searchCleanupTolerance = $state(0.08);
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
  let reviewInitialized = $state(false);
  const reconstructionChoices = $derived(new Map(searchResult?.steps.map(step => [step.frame_index, step.geometry_option_index] as const) ?? []));
  const reconstructionFlags = $derived(new Map(searchResult?.steps.map(step => [step.frame_index,step.pose_flagged === true] as const) ?? []));
  const reconstructionComplete = $derived(videoInputComplete && (!searchEnabled || !!searchError ||
    !!searchResult && ["solved", "already_solved", "exhausted", "state_limit"].includes(searchResult.status)));
  const recording = new DetectionRecording();
  const timerDiagnostics = new TimerDiagnostics(recording, persistDiagnostics);
  let diagnosticsWorker:Worker|undefined;
  let diagnosticResponse:{resolve():void;reject(error:Error):void}|undefined;
  const diagnosticWriter=new DiagnosticWriter(id=>({id,...recording.exportRun(id),model:$state.snapshot(manifest)}),sendDiagnostics);
  let captureFps=$state(0),cameraDelayMs=$state(0),skippedCameraFrames=$state(0);
  let savedFrames = $state(0);
  const latestTimerReconstruction=$derived(Object.values(timerReconstructions).at(-1));
  const timerBuffering=$derived(timerCamera && timerCapturePhase !== 'capturing' &&
    !!latestTimerReconstruction && ['waiting','queued','searching'].includes(latestTimerReconstruction.status));
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
  let stream=$state.raw<MediaStream>();
  let cameraPreview=$state<HTMLVideoElement>();
  $effect(()=>{
    const element=cameraPreview,source=stream;
    if(!element)return;
    element.srcObject=source??null;
    if(source)void element.play().catch(cause=>{
      if(element.srcObject===source)mediaError=cause instanceof Error?cause.message:String(cause);
    });
    return()=>{if(element.srcObject===source)element.srcObject=null;};
  });
  let fileSource: VideoFileSource | undefined;
  let capture: CameraCapture | undefined;
  let seekVersion = 0;
  let seekResume = false,
    seeking = false;
  let captureStopped = $state(false);
  const showCameraPreview=$derived(sourceKind==='camera'&&timerCamera&&timerCapturePhase==='idle'&&!captureStopped);
  let fileEnded = false;
  type CameraEnd={timerEnd:true;id:string;timestamp:number;close():void};
  type CameraFrame=SourceFrame&{close():void;timerReset?:boolean};
  const cameraQueue = new FrameQueue<CameraFrame|CameraEnd>();
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
        captureMetrics?:CameraCaptureMetrics;
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
    fitting?: Promise<FittedPose>; fitter?: PoseFitter; fitStartedAtMs?: number; fitCompletedAtMs?: number };
  type CachedFrame = { saved: SavedVideoResult; context: Pick<FrameContext, "snapshot" | "generation" | "timestamp"> };
  type TimerEnd={timerEnd:true;id:string;timestamp:number;generation:number};
  // Recovery needs 180ms of future source context, plus the adjacent initial fit.
  const completionQueue = new OrderedStage<DetectedFrame | CachedFrame | TimerEnd>(ADMISSION_CAPACITY, completeFrame,
    (packet) => {if('context' in packet)presenter.releaseFrame(packet.context.snapshot);},
    (size) => { fitBufferedFrames = size; continuePipeline(); },
    (cause) => {
      mediaError = cause instanceof Error ? cause.message : String(cause);
      running = false;
      capture?.pause();
    });
  const idleWaiters: (() => void)[] = [];
  const baseUrl = new URL(import.meta.env.BASE_URL, location.origin).href;
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
  const needsPose = () => timerCamera || activeTab === 'timer' || showPose || searchEnabled;
  const inferenceAllowed = () => (timerCamera || !moveSearch?.solved) && !(sourceKind === "video" && videoInputComplete);
  const displayPose = () => showPose && (sourceKind !== "video" || !trimming);
  $effect(() => {
    if (!reviewInitialized && sourceKind === "video" && reconstructionComplete && !trimming) {
      reviewInitialized = true;
      // Enter review paused at the first saved frame, once per reconstruction.
      // Later result updates and playback reaching the end must not rewind it.
      untrack(() => { void requestReview(reviewRows[0]?.timestamp ?? solveStart); });
    }
  });
  const poseFittingOptions = () => timerCamera ? {scramble:'',turningStartFrame:null,originSeconds:0,camera:true} : ({scramble:moveSearch?.config.scramble ?? initialScramble.trim(),
    camera:sourceKind==='camera',
    turningStartFrame:moveSearch ? moveSearch.config.turning_start_frame : openingHoldSeconds>0 ? Math.floor((searchOriginTimestamp+openingHoldSeconds)*(sourceKind==='camera'?CAMERA_CLOCK_HZ:sourceFps||30)) : null,
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
    if (timerCamera || activeTab === 'timer' || !searchEnabled || sourceKind === "none") return;
    if (!initialScramble.trim()) { searchError = "Enter the initial scramble to start move search."; return; }
    if(searchCleanup&&(!Number.isFinite(searchCleanupTolerance)||searchCleanupTolerance<0||searchCleanupTolerance>1)){
      searchError="Cleanup color tolerance must be between 0 and 1.";return;
    }
    const token = generation;
    const config = createSearchConfig({scramble:initialScramble,fps:sourceFps||30,camera:sourceKind==='camera',
      openingHoldSeconds,diagnostics:searchDiagnostics,bidirectional:searchBidirectional&&sourceKind==="video",originSeconds:sourceKind==="video"?searchOriginTimestamp:0,
      postprocessIdentityLoops:searchCleanup,postprocessColorTolerance:searchCleanup?searchCleanupTolerance:0.08,
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
      searchError = message;
      if(sourceKind === 'video'){
        running=false;seekResume=false;videoInputComplete=true;fileEnded=true;cancelScheduled();
      }
      recording.saveError(runId,message,moveSearch?.actions??[]);void persistDiagnostics(runId);
    } }, sourceKind === "video");
  }
  async function finishMoveSearch() {
    const token = generation, activeSearch = moveSearch;
    activeSearch?.startDeadline();
    await waitForPipelineIdle();
    if (token !== generation || disposed || activeSearch !== moveSearch) return;
    if (timerCamera && captureStopped) {timerReplayStore.closeInput();timerReconstruction.closeInput();}
    if (sourceKind === "video") videoInputComplete = true;
    if(activeRunId)void persistDiagnostics(activeRunId);
    try { await activeSearch?.end(); } catch (cause) { if (token === generation) searchError = cause instanceof Error ? cause.message : String(cause); }
  }
  async function persistDiagnostics(runId:string) {
    if((!import.meta.env.DEV&&!['localhost','127.0.0.1','[::1]'].includes(location.hostname))||!recording.diagnosticsEnabled(runId))return true;
    try {
      await recording.ready(runId);
      await diagnosticWriter.write(runId);
      return true;
    }catch(cause){console.error('Local diagnostic capture failed',cause);return false;}
  }
  function sendDiagnostics(capture:unknown):Promise<void> {
    if(disposed)return Promise.resolve();
    diagnosticsWorker??=new Worker(new URL('./lib/diagnostics.worker.ts',import.meta.url),{type:'module'});
    const active=diagnosticsWorker;
    return new Promise((resolve,reject)=>{
      diagnosticResponse={resolve,reject};
      active.onmessage=({data})=>{
        diagnosticResponse=undefined;
        if(data.ok)resolve();else reject(new Error(data.error));
      };
      active.onerror=event=>{diagnosticResponse=undefined;active.terminate();diagnosticsWorker=undefined;reject(new Error(event.message||'Diagnostic writer stopped'));};
      active.postMessage({baseUrl,capture});
    });
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
      reviewInitialized = false;
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
    updateSearchPressure();
    if(!inferenceAllowed())refreshPending=false;
    if (!busy && (refreshPending || !running || status !== "ready")) void flushPose();
    if (refreshPending && !busy && !completionQueue.size && !seeking && status === "ready") {
      refreshPending = false;
      void inferFrame();
    } else scheduleNext();
  }
  function updateSearchPressure(){
    if(!timerCamera||queuedFrames<=2)timerReconstruction.setBackpressure(false);
    else if(queuedFrames>=8)timerReconstruction.setBackpressure(true);
  }
  async function completeFrame(packet: DetectedFrame | CachedFrame | TimerEnd) {
    if('timerEnd' in packet){
      if(packet.generation===generation&&!disposed){
        timerReplayStore.finishThrough(packet.timestamp);timerReconstruction.finishThrough(packet.timestamp);
        if(timerCaptureGate.finish(packet.id)){
          timerCapturePhase='idle';presentationTimes=[];lastResultTime=0;fps=0;previousFrameStart=0;
        }
      }
      return;
    }
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
          packet.result.fittingError = poseError;
          // Old queued rejections must not dispose a newly recovered fitter.
          if (poseFitter === packet.fitter) {
            poseFitter?.dispose();
            poseFitter = undefined;
            resetPose = true;
          }
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
      ...(completed.camera&&cameraClockOriginMs!==undefined?{camera:{captureFps:completed.captureMetrics?.captureFps??captureFps,
        copyMs:completed.captureMetrics?.copyMs,deliveryMs:completed.captureMetrics?.deliveryMs,
        skippedFrames:completed.captureMetrics?.skippedFrames??0,browserSkippedFrames:completed.captureMetrics?.browserSkippedFrames??0,
        queuedFrames,bufferedFits:fitBufferedFrames,
        sourceAgeMs:Math.max(0,performance.timeOrigin+presentedAtMs-cameraClockOriginMs-completed.timestamp*1000)}}:{}),
    };
    if(packet.result.pipeline.camera)cameraDelayMs=packet.result.pipeline.camera.sourceAgeMs;
    result = packet.result;
    presentedTimestamp=completed.timestamp;
    viewingSavedResult = false;
    if (completed.recording.sourceKind === "video" && (!completed.fitPose || packet.result.fittedPose))
      cachedVideoFrames = videoResults.put(completed.cacheKey, completed.timestamp, packet.result);
    const retainDetection = completed.recording.sourceKind !== 'camera' || !timerCamera || timerReconstruction.includes(completed.timestamp);
    if (retainDetection && (completed.counted || completed.recording.sourceKind === "image" || completed.recording.sourceKind === "video")) {
      const sequence = recording.append(packet.result, completed.recording);
      if (completed.camera && timerCamera) {
        for (const id of timerReconstruction.solveIds(completed.timestamp)) timerDiagnostics.frame(id, sequence);
      }
      savedFrames = recording.length;
    }
    if (completed.camera && timerCamera) {
      timerReplayStore.capture(completed.timestamp,packet.result,()=>completed.snapshot.convertToBlob({type:'image/jpeg',quality:.8}));
      timerReconstruction.accept(completed.timestamp, packet.result);
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
        "WebGPU is required. Open this app in a browser that supports WebGPU.";
      status = "error";
      return;
    }
    loadingText = backend === "webgpu" ? "Checking WebGPU…" : "Preparing app files…";
    try {
      if (backend === "webgpu") {
        const available = await hasWebGpuAdapter(navigator.gpu);
        if (disposed || request !== modelGeneration) return;
        if (!available) throw new Error("WebGPU is unavailable on this device or browser.");
      }
      loadingText = "Preparing app files…";
      await offlineCacheReady();
      if (disposed || request !== modelGeneration) return;
      loadingText = "Reading detector configuration…";
      if (!manifest) {
        const response = await fetch(new URL("models/detector.json", baseUrl));
        if (
          !response.ok ||
          !response.headers.get("content-type")?.includes("json")
        )
          throw new Error(
            "Detector files are missing. Add the files described in web/README.md.",
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
              packet.fitter = poseFitter;
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
              if(completed.camera&&timerCamera&&completed.generation===generation)
                timerReconstruction.submit(completed.timestamp,data.result,packet.fitting);
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
    timerCaptureGate.clear();timerCapturePhase='idle';
    timerReplayStore.closeInput();
    timerReconstruction.closeInput('Camera disconnected before the complete solve window was processed');
    captureFps=0;cameraDelayMs=0;skippedCameraFrames=0;
    timerCamera = false; cameraClockOriginMs = undefined;
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
    reviewInitialized = false;
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
    timerCamera = activeTab === 'timer';
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
          cameraClockOriginMs = frame.clockOriginMs;
          let timerReset=false;
          if(timerCamera){
            const admission=timerCaptureGate.admit(frame.timestamp);
            if(admission.type!=='frame'){
              frame.bitmap.close();capture?.acknowledge();
              if(admission.type==='end'){
                capture?.idle();timerCapturePhase='draining';
                cameraQueue.push({timerEnd:true,id:admission.id,timestamp:admission.timestamp,close(){}});
                queuedFrames=cameraQueue.length;scheduleNext();
              }else if(timerCapturePhase==='idle')capture?.idle();
              return;
            }
            timerReset=admission.reset;
          }
          cameraQueue.push({ ...frame,timerReset, close: () => frame.bitmap.close() });
          queuedFrames = cameraQueue.length;
          updateSearchPressure();
          scheduleNext();
        },
        (message) => {
          if (token !== generation || disposed) return;
          captureStopped = true;
          stream?.getTracks().forEach((track) => track.stop());
          if (message) mediaError = message;
          if (!cameraQueue.length && !busy) running = false;
        },
        metrics=>{if(token===generation&&!disposed){captureFps=metrics.captureFps;skippedCameraFrames=metrics.skippedFrames+metrics.browserSkippedFrames;}},
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
      if (activeTab === 'timer') document.getElementById('workspace-panel')?.focus({ preventScroll: true });
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

  function dispatch(bitmap: ImageBitmap, timestamp: number, counted: boolean,captureMetrics?:CameraCaptureMetrics) {
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
      captureMetrics,
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
    if(timerCamera)return;
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
      if(frame&&'timerEnd' in frame){
        if(token===generation&&activeWorker===worker&&!disposed){
          // No later idle-camera work enters the fitter. Flush its remaining
          // lookahead and close the solve after ordered presentation drains.
          void flushPose();completionQueue.enqueue({timerEnd:true,id:frame.id,timestamp:frame.timestamp,generation:token});
        }
        markIdle();continuePipeline();return;
      }
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
      if('timerReset' in frame&&frame.timerReset){resetPose=true;resetTracking=true;}
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
      } else dispatch(frame.bitmap, frame.timestamp, true,frame.captureMetrics);
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
    // Camera timestamps already pace frame admission. Drain its FIFO as fast as
    // inference/backpressure permit, so short UI stalls can be caught up.
    const interval = sourceKind==='camera'?0:playbackTiming(sourceFps,sourceKind==="video"&&reconstructionComplete?playbackSpeed:1).frameIntervalMs;
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
  async function downloadDetections() {
    if (!recording.length) return;
    try { await recording.readyAll(); }
    catch (cause) { mediaError = `Unable to prepare diagnostic metadata: ${String(cause)}`; return; }
    const blob = new Blob(
      [
        JSON.stringify({
          ...compactDetectionCapture(recording.toJSON()),
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
  async function downloadTimerDiagnostics(id: string) {
    let data:unknown;
    try{
      if(timerDiagnostics.released(id)){
        const url=new URL('__diagnostics/run',baseUrl);url.searchParams.set('id',timerDiagnostics.runId(id)!);
        const response=await fetch(url);if(!response.ok)throw Error('Saved diagnostic capture is unavailable');data=await response.json();
      }else {
        await recording.ready(timerDiagnostics.runId(id)!);
        data=timerDiagnostics.export(id);
      }
    }catch(error){mediaError=error instanceof Error?error.message:String(error);return;}
    const capture=compactDetectionCapture(data as ReturnType<DetectionRecording["toJSON"]>);
    const url = URL.createObjectURL(new Blob([JSON.stringify({...capture,model:$state.snapshot(manifest)})], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `cube-timer-diagnostics-${id}.json`; link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
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
  function changePlaybackSpeed(event:Event){
    const speed=Number((event.currentTarget as HTMLSelectElement).value);
    if(!PLAYBACK_SPEEDS.includes(speed as PlaybackSpeed))return;
    playbackSpeed=speed as PlaybackSpeed;
    cancelScheduled();scheduleNext();
  }
  function drop(event: DragEvent) {
    event.preventDefault();
    dragging = false;
    if (activeTab === 'timer' || timerBusy) return;
    void openFile(event.dataTransfer?.files[0]);
  }
  function cleanup() {
    timerReconstruction.cancel('Page closed before reconstruction finished');
    timerReplayStore.clear();timerReplayCache.clear();latestTimerReplay=undefined;
    timerCaptureGate.clear();
    disposed = true;
    generation++;
    cancelScheduled();
    previewVersion++; thumbnails.forEach(url => URL.revokeObjectURL(url));
    capture?.stop();
    cameraQueue.clear();
    fileSource?.close();
    worker?.terminate();
    diagnosticsWorker?.terminate();diagnosticResponse?.resolve();diagnosticResponse=undefined;
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

<svelte:head><title>Magic Stew · {activeTab === 'timer' ? 'Cube timer' : 'Detector'}</title></svelte:head>
<svelte:window onpagehide={cleanup} onkeydown={reviewKeydown} onpopstate={followHistory} />

<div class="shell" class:timer-shell={activeTab === 'timer'}>
  <div class="beta-banner" role="note"><strong>Early beta — barely functional.</strong> Cube detection and reconstruction will likely fail in many cases.</div>
  <header class="topbar">
    <a class="brand" href={workspacePath('detector',workspaceBase)} aria-label="Magic Stew home">
      <span class="brand-logo-frame"><img class="brand-logo" src={new URL('magic-stew-logo.png',baseUrl).href} alt="" width="64" height="64" /></span>
      <span class="brand-label">Magic Stew</span>
    </a>
    <div class="app-tabs" role="tablist" aria-label="Workspace">
      <a id="detector-tab" role="tab" aria-selected={activeTab === 'detector'} aria-controls="workspace-panel" aria-disabled={timerBusy} href={workspacePath('detector',workspaceBase)} onclick={event=>navigateTab(event,'detector')}>Detector</a>
      <a id="timer-tab" role="tab" aria-selected={activeTab === 'timer'} aria-controls="workspace-panel" aria-disabled={timerBusy} href={workspacePath('timer',workspaceBase)} onclick={event=>navigateTab(event,'timer')}>Cube timer</a>
    </div>
  </header>

  <main>
  <div id="workspace-panel" role="tabpanel" tabindex="0" aria-labelledby={activeTab === 'timer' ? 'timer-tab' : 'detector-tab'}>
    <div class:timer-layout={activeTab === 'timer'} class:with-inline-replay={activeTab === 'timer' && !!latestTimerReplay}>
    <div class="timer-controls" hidden={activeTab !== 'timer'}>
      <CubeTimer active={activeTab === 'timer'} cameraReady={sourceKind === 'camera' && running && !captureStopped && status === 'ready' && cameraClockOriginMs !== undefined}
        reconstructions={timerReconstructions} replays={timerReplays} {replayLoading} replayError={replayStorageError}
        latestReplayPosition={latestTimerReplay?latestReplayPosition:undefined} onlatestseek={seekLatestTimerReplay}
        onreplayselect={id=>{void timerReplayCache.select(id);}}
        ondelete={deleteTimerSolve} onclearsession={clearTimerSession} onstart={startTimerSolve} onstop={stopTimerSolve} onbusychange={value => timerBusy = value}
        ondownloaddiagnostics={downloadTimerDiagnostics} />
    </div>
    <div class="workspace" class:timer-workspace={activeTab === 'timer'}>
      {#if activeTab === 'detector'}<p class="local-processing-note">Everything runs locally on your device.</p>{/if}
      <section class="viewer" aria-label="Detection viewer">
        {#if sourceKind === 'video' && !trimming}
          <div class="reconstruction-status" class:failed={!!searchError || searchResult?.status === 'exhausted' || searchResult?.status === 'state_limit'} role="status" aria-live="polite">
            {#if searchError || searchResult?.status === 'exhausted' || searchResult?.status === 'state_limit'}
              <strong>Reconstruction failed</strong>
              <span>{searchError || (searchResult?.status === 'state_limit' ? 'Search limit reached.' : 'No full reconstruction found.')}</span>
              <span>Partial reconstruction{searchResult?.steps.length ? ' retained.' : ' · No moves recovered.'}</span>
            {:else if cubeSolved}<strong>Full reconstruction</strong>{/if}
          </div>
        {/if}
        <div class="viewer-toolbar">
          <span class="source-label"
            ><span class="status-dot" class:ready={sourceKind !== "none"}
            ></span>{sourceName}</span
          >
        </div>
        <div class="reconstruction-pair" class:with-timer-replay={activeTab === 'timer'} class:with-projection={sourceKind === "video" && reconstructionComplete && !trimming}>
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
            hidden={showCameraPreview}
            class:empty={sourceKind === "none"}
            aria-label="Processed source frame"
          ></canvas>
          {#if sourceKind==='camera'&&timerCamera}
            <video bind:this={cameraPreview} hidden={!showCameraPreview} autoplay muted playsinline aria-label="Camera preview"></video>
          {/if}
          {#if sourceKind === "none"}
            <div class="empty-state">
              {#if activeTab === 'timer'}
                <button class="primary" onclick={camera} disabled={status !== "ready"}>Connect camera</button>
              {:else}
                <p>Drop an image or video here</p>
                <button class="primary" onclick={() => fileInput.click()}>Upload file</button>
              {/if}
            </div>
          {/if}
          {#if status === "loading"}<div class="loading-badge" role="status">
              <span class="spinner"></span>{loadingText}
            </div>{/if}
          {#if sourceKind === 'video' && !trimming && !reconstructionComplete || timerBuffering}
            <div class="buffering-overlay" role="status" aria-label="Reconstructing">
              <span class="spinner"></span><span>Allowing up to 30 seconds to find a solution…</span>
            </div>
          {/if}
        </div>
        {#if activeTab === 'timer'}
          <LatestTimerReplay clip={latestTimerReplay} reconstruction={latestReplayReconstruction}
            disabled={timerBusy} seekRequest={latestReplaySeek} onpositionchange={timestamp=>latestReplayPosition=timestamp} />
        {/if}
        {#if sourceKind === "video" && reconstructionComplete && !trimming}
          <ProjectedCube pose={coloredProjection} sourceTimestamp={presentedTimestamp} timestamp={timecode(presentedTimestamp)} solved={cubeSolved}
            timeline={projectedTimeline} steps={searchResult?.steps??[]} {sourceFps} playbackSpeed={running?playbackSpeed:1}
            sourceFrame={result?.fittedPose?.sourceFrame??Math.round(presentedTimestamp*sourceFps)} />
        {/if}
        </div>
        {#if sourceKind !== "none"}<div class="transport">
          <div class="transport-actions">
            {#if activeTab === 'detector'}<button class="small" onclick={() => fileInput.click()}>Upload file</button>{/if}
            {#if sourceKind === "video" && reconstructionComplete && !trimming}
              <button class="small" onclick={togglePlayback}>{running ? "Pause" : "Play"}</button>
              <label class="playback-speed">Speed
                <select aria-label="Playback speed" bind:value={playbackSpeed} onchange={changePlaybackSpeed}>
                  {#each PLAYBACK_SPEEDS as speed}<option value={speed}>{speed}×</option>{/each}
                </select>
              </label>
              <button class="small" onclick={editSolveRange}>Adjust range</button>
            {/if}
            <button class="small subtle" onclick={stopSource} disabled={timerBusy}>{timerCamera ? 'Disconnect camera' : 'Clear'}</button>
          </div>
        </div>{/if}
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
        <div class="viewer-statusbar">
          <span class="fps" aria-label="Frames per second">{(sourceKind === 'camera' && timerCapturePhase === 'idle' ? captureFps : fps)?.toFixed(1) || '0.0'} FPS</span>
          <div class="detection-data" aria-label="Detection data">
            <span>{savedFrames} frames</span>
            <button class="small" onclick={downloadDetections} disabled={!savedFrames || timerBusy}>Download detection data</button>
          </div>
        </div>
        {#if mediaError}<div class="error" role="alert">{mediaError}</div>{/if}
        {#if status === "error"}<div class="error" role="alert">
            <p>{error}</p>
            <button class="small" onclick={loadModel}>Reload detector</button>
          </div>{/if}
      </section>
      {#if activeTab === 'detector' && sourceKind === 'video' && reconstructionComplete && !trimming}
        <TpsGraph steps={searchResult?.steps??[]} fps={sourceFps} start={solveStart} end={solveEnd}
          cursor={presentedTimestamp} complete={cubeSolved} onseek={timestamp=>{void requestReview(timestamp);}} />
      {/if}

    </div>
    </div>
  </div>

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
