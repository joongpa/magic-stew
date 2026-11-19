import type { FrameResult } from "./detector";
import type { FittedPose } from "./pose-types";
import type { MoveSearchConfig,MoveSearchResult } from "./move-search-types";
import {videoFrameIdentity} from '../../../pose-fitter/runtime/reconstruction-policy.mjs';
import type {SearchAction} from '../../../pose-fitter/runtime/incremental-search.mjs';
import {ParallelMoveSearch} from './parallel-move-search';
const COLORS=["corner","corner-edge","corner-edge-center","white","yellow","orange","red","blue","green"];

/** Frame sequence is assigned on admission, before fitting promises finish.
 * Rust can precompute out-of-order arrivals while withholding missing depths. */
export class MoveSearch {
  readonly actions:SearchAction[]=[];
  private worker:Worker|ParallelMoveSearch;private sequence=0;private seen=new Set<number>();private lastSequence=-1;
  private pending=new Set<Promise<void>>();private closed=false;private ended=false;private endRequested=false;
  private solutionFound=false;
  get solved(){return this.solutionFound;}
  constructor(baseUrl:string,readonly config:MoveSearchConfig,onResult:(value:MoveSearchResult,ended:boolean)=>void,onError:(message:string)=>void,private video=false){
    this.onError=onError;
    this.worker=config.bidirectional?new ParallelMoveSearch(baseUrl):new Worker(new URL("./move-search.worker.ts",import.meta.url),{type:"module"});
    this.worker.onmessage=({data})=>{
      if(this.closed)return;
      if(data.type==="result"){
        if(data.result.status==="solved"||data.result.status==="already_solved")this.solutionFound=true;
        if(this.config.diagnostics!==false)this.actions.push(...(data.actions??[]));onResult(data.result,!!data.ended);
      }
      if(data.type==="error"){this.dispose();onError(data.message);}
    };
    this.worker.onerror=()=>{this.dispose();onError("The Rust move-search worker stopped");};
    this.worker.postMessage({type:"init",baseUrl,config});
  }
  private onError:(message:string)=>void;
  submit(timestamp:number,frame:FrameResult,fitting:Promise<FittedPose>){
    if(this.closed||this.ended||this.solutionFound)return;
    const identity=videoFrameIdentity(timestamp,this.config),sourceFrame=identity.sourceFrame;
    const sequence=this.video?identity.sequence:this.sequence++;
    if(sequence<0)return;
    if(this.seen.has(sequence))return;
    this.seen.add(sequence);this.lastSequence=Math.max(this.lastSequence,sequence);
    const task=fitting.then(pose=>{
      if(this.closed||this.ended||this.solutionFound)return;
      let landmarkCount=0;const colors:string[]=[];
      for(const point of frame.detections){if(point.classId<3)landmarkCount++;else colors.push(COLORS[point.classId]);}
      this.worker.postMessage({type:"frame",frame:{sequence,frame:sourceFrame,bank:pose.bank,
        landmark_count:landmarkCount,colors,
        flagged:pose.pose_flag_reason===undefined?undefined:pose.pose_flag_reason!==null,
        geometry_flagged:pose.pose_geometry_flag_reason===undefined?undefined:pose.pose_geometry_flag_reason!==null}});
    }).catch(cause=>{
      if(!this.closed&&!this.solutionFound){this.dispose();this.onError(cause instanceof Error?cause.message:String(cause));}
    });
    this.pending.add(task);void task.finally(()=>{this.pending.delete(task);this.maybeEnd();}).catch(()=>{});
  }
  async end(){
    if(this.closed||this.ended)return;this.endRequested=true;
    if(!this.solutionFound)await Promise.all([...this.pending]);this.maybeEnd();
  }
  private maybeEnd(){
    if(!this.closed&&!this.ended&&this.endRequested&&
      (this.solutionFound||!this.pending.size&&this.seen.size===this.lastSequence+1)){
      this.ended=true;this.worker.postMessage({type:"end"});
    }
  }
  dispose(){this.closed=true;this.worker.terminate();}
}
