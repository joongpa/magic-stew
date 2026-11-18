import type { FrameResult } from "./detector";
import type { FittedPose } from "./pose-types";
import type { MoveSearchConfig,MoveSearchResult } from "./move-search-types";
const COLORS=["corner","corner-edge","corner-edge-center","white","yellow","orange","red","blue","green"];

/** Frame sequence is assigned on admission, before fitting promises finish.
 * Rust can precompute out-of-order arrivals while withholding missing depths. */
export class MoveSearch {
  private worker:Worker;private sequence=0;private seen=new Set<number>();private lastSequence=-1;
  private pending=new Set<Promise<void>>();private closed=false;private ended=false;private endRequested=false;
  constructor(baseUrl:string,readonly config:MoveSearchConfig,onResult:(value:MoveSearchResult)=>void,onError:(message:string)=>void,private video=false){
    this.onError=onError;
    this.worker=new Worker(new URL("./move-search.worker.ts",import.meta.url),{type:"module"});
    this.worker.onmessage=({data})=>{
      if(this.closed)return;
      if(data.type==="result")onResult(data.result);
      if(data.type==="error"){this.dispose();onError(data.message);}
    };
    this.worker.onerror=()=>{this.dispose();onError("The Rust move-search worker stopped");};
    this.worker.postMessage({type:"init",baseUrl,config});
  }
  private onError:(message:string)=>void;
  submit(timestamp:number,frame:FrameResult,fitting:Promise<FittedPose>){
    if(this.closed||this.ended)return;
    const sourceFrame=Math.round(timestamp*this.config.fps);
    const elapsed=timestamp-(this.config.source_origin_seconds??0);
    const sequence=this.video?(this.config.fps>30.001?Math.floor(elapsed*30+0.0001):Math.round(elapsed*this.config.fps)):this.sequence++;
    if(sequence<0)return;
    if(this.seen.has(sequence))return;
    this.seen.add(sequence);this.lastSequence=Math.max(this.lastSequence,sequence);
    const task=fitting.then(pose=>{
      if(this.closed)return;
      this.worker.postMessage({type:"frame",frame:{sequence,frame:sourceFrame,bank:pose.bank,
        colors:frame.detections.filter(p=>p.classId>=3).map(p=>COLORS[p.classId])}});
    }).catch(cause=>{
      if(!this.closed){this.dispose();this.onError(cause instanceof Error?cause.message:String(cause));}
    });
    this.pending.add(task);void task.finally(()=>{this.pending.delete(task);this.maybeEnd();}).catch(()=>{});
  }
  async end(){
    if(this.closed||this.ended)return;this.endRequested=true;
    await Promise.all([...this.pending]);this.maybeEnd();
  }
  private maybeEnd(){
    if(!this.closed&&!this.ended&&this.endRequested&&!this.pending.size&&this.seen.size===this.lastSequence+1){
      this.ended=true;this.worker.postMessage({type:"end"});
    }
  }
  dispose(){this.closed=true;this.worker.terminate();}
}
