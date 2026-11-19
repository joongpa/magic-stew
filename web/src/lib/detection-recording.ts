import type { FrameResult, InferenceBackend, Settings } from "./detector";
import type { MoveSearchConfig,MoveSearchResult } from "./move-search-types";
import type {RunMetadata} from './run-provenance';
import type {SearchAction} from '../../../pose-fitter/runtime/incremental-search.mjs';

export interface RecordingContext {
  sourceId: number;
  sourceName: string;
  sourceKind: "image" | "video" | "camera";
  sourceFps: number;
  segment: number;
  timestampSeconds: number;
  backend: InferenceBackend;
  settings: Settings;
  runId?:string;
}

/** Retains results without per-frame serialization or copying the detections. */
export class DetectionRecording {
  private readonly startedAt = new Date().toISOString();
  private revision=0;
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
    runId?:string;
  }[] = [];
  private readonly searches = new Map<number,{sourceId:number;config:MoveSearchConfig;result:MoveSearchResult}>();
  private readonly runs=new Map<string,{id:string;sourceId:number;config:MoveSearchConfig;metadata:RunMetadata;
    result?:MoveSearchResult;error?:string;actions:SearchAction[];frameSequences:number[]}>();
  private readonly files=new Map<number,Promise<NonNullable<RunMetadata['source']>>>();
  private readonly readiness=new Map<string,Promise<void>>();
  setSourceFile(sourceId:number,file:File) {
    const value={name:file.name,bytes:file.size,lastModified:file.lastModified};
    const pending=file.arrayBuffer().then(bytes=>crypto.subtle.digest('SHA-256',bytes)).then(hash=>
      ({...value,sha256:[...new Uint8Array(hash)].map(v=>v.toString(16).padStart(2,'0')).join('')}));
    this.files.set(sourceId,pending);void pending.catch(()=>{});
  }
  startRun(sourceId:number,config:MoveSearchConfig,metadata:RunMetadata) {
    const id=`${this.startedAt}-${sourceId}-${this.runs.size}`;
    this.runs.set(id,{id,sourceId,config:structuredClone(config),metadata:structuredClone(metadata),actions:[],frameSequences:[]});return id;
  }
  linkCachedFrame(id:string,timestamp:number,result:FrameResult) {
    const run=this.runs.get(id);if(!run)return;
    const sequence=this.frames.findIndex(row=>row.sourceId===run.sourceId&&row.result===result&&Math.abs(row.timestampSeconds-timestamp)<0.00001);
    if(sequence<0)throw Error('Cached reconstruction frame lacks its original raw detection record');
    if(!run.frameSequences.includes(sequence))run.frameSequences.push(sequence);
  }
  setFingerprints(id:string,fingerprints:Promise<NonNullable<RunMetadata['fingerprints']>>) {
    const pending=(async()=>{const run=this.runs.get(id);
      if(run){run.metadata.fingerprints=await fingerprints;run.metadata.source=await this.files.get(run.sourceId);}
    })();this.readiness.set(id,pending);return pending;
  }
  ready(id:string){return this.readiness.get(id)??Promise.resolve();}
  diagnosticsEnabled(id:string){return this.runs.get(id)?.config.diagnostics!==false;}
  exportRun(id:string) {
    const run=this.runs.get(id);if(!run)throw Error('Unknown reconstruction run');
    const all=this.toJSON(),frames=run.frameSequences.map(sequence=>all.frames[sequence]);
    return {...all,sources:all.sources.filter(source=>source.id===run.sourceId),frames,
      runs:[{...run,frameSequences:frames.map((_,index)=>index)}],
      moveSearches:run.result?[{sourceId:run.sourceId,config:run.config,result:run.result}]:[]};
  }
  saveSearch(sourceId:number,config:MoveSearchConfig,result:MoveSearchResult,runId?:string,actions:SearchAction[]=[]) {
    this.searches.set(sourceId,{sourceId,config,result});
    this.revision++;
    const run=runId?this.runs.get(runId):undefined;if(run){run.result=result;run.actions=actions;}
  }
  saveError(runId:string,error:string,actions:SearchAction[]) {
    const run=this.runs.get(runId);if(run){run.error=error;run.actions=actions;this.revision++;}
  }

  get length() {
    return this.frames.length;
  }

  append(result: FrameResult, context: RecordingContext) {
    this.revision++;
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
      runId:context.runId,
    });
    if(context.runId)this.runs.get(context.runId)?.frameSequences.push(this.frames.length-1);
  }

  toJSON() {
    return {
      version: 2,
      revision:this.revision,
      startedAt: this.startedAt,
      confidenceFields: {
        "result.detections[].score": "Pose per-object detection/class score",
        "result.detections[].presence": "Keypoint-presence score",
        "result.cube.score":
          "Separate cube bbox detector score from the last acquisition",
      },
      fittingFields: {
        "result.fittedPose.wireframes": "Projected meshes in the same order as all retained geometry options",
        "result.fittedPose.wireframeFaces": "Physical face IDs in each projected mesh, for reconstructed sticker colors",
        "result.fittedPose.wireframeDepths": "Camera depths for far-to-near face rendering",
        "moveSearches[].result.steps[].state_score": "Balanced geometry/color cost; lower is better, not detector confidence",
        "moveSearches[].config.motion_cost_weight": "Explicit nonnegative weight for the latest eight processed-frame motion requirements; legacy absent values mean zero",
        "moveSearches[].result.steps[].recent_motion_penalty": "Queue-cost addition from this path's last eight normalized motion requirements, including the current step",
        "moveSearches[].result.steps[].required_tps_ratio": "Minimum endpoint motion requirement divided by the configured TPS budget; not measured physical TPS",
        "moveSearches[].result.steps[].color_match_count": "Unique projected sticker positions with valid color assignments, not raw detection count or number of correct colors",
        "moveSearches[].result.steps[].color_error": "Color mismatch fraction; with zero assigned stickers, zero is a neutral placeholder and does not indicate observed agreement",
        "moveSearches[].result.steps[].orientation_symmetry_index": "Search-selected cube-frame symmetry from pose-fitter/tables/orientations.json; the six fitted geometries are unchanged",
        "result.fittedPose.stage": "cpu_pipeline means startup, initial, neighbors, orientation and recovery are complete",
        "result.fittedPose.pipelineVersion": "Content-addressed Rust/WASM asset loaded by the fitter",
        "result.fittedPose.pose_flag_reason": "Display/evidence flag: geometry may be unreliable or assigned sticker-color evidence may be insufficient; fitted data is retained",
        "result.fittedPose.pose_geometry_flag_reason": "Geometry-specific reliability flag, separate from insufficient sticker-color evidence",
        "result.fittedPose.continuity_source_pose": "Original finalized geometric winner used for camera continuity before output-only weak-layer alternatives",
        "result.fittedPose.final_bank_diagnostics": "Counts of weak-layer projection attempts and retained alternatives; no extra solver, sweep or PnP passes",
        "result.fittedPose.bank[].metrics.layer_support": "Geometric proximity support per layer, not detection confidence, keypoint presence, or distinct assigned-observation counts",
        "result.fittedPose.startup_reference_core": "Screen-plane core of the latest eligible finalized pose with reliable geometry; color-only flags retain geometry eligibility; null during reference warmup",
        "result.fittedPose.startup_motion_allowance": "Pixels of normal core motion allowed for the finalized reference's lag, estimated from its last two unflagged cores",
        "result.fittedPose.fittingConfig": "Frame sequence configuration and buffered fitting segment for standalone replay",
        "result.fittedPose.bank[].score": "Geometry cost, or recovery cost when recovery_score_components is present; lower is better, not detector confidence",
        "result.fittedPose.bank[].recovery_score_components": "Capped reprojection loss, pose-prior loss, active-layer count and projection weight used to rank sparse depth-recovery proposals",
        "result.fittedPose.bank[].nearest": "Projected sticker-face ID for each observed sticker keypoint",
        "result.fittedPose.bank[].valid": "Whether that sticker keypoint has an eligible projected-face assignment",
        "result.fittedPose.bank[].flat_fallback": "Reserved zero-angle option outside the five assignment-distinct slots",
      },
      sources: [...this.sources.values()],
      frames: this.frames,
      moveSearches: [...this.searches.values()],
      runs:[...this.runs.values()],
    };
  }
}
