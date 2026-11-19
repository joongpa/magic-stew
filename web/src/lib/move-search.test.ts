import assert from "node:assert/strict";
import test from "node:test";
import { MoveSearch } from "./move-search";
import type { MoveSearchConfig } from "./move-search-types";
import type { FrameResult } from "./detector";
import type { FittedPose } from "./pose-types";
const config:MoveSearchConfig={scramble:"R U",fps:60,max_tps:54,turning_start_frame:null,branch_widths:[3],
 branch_stage_expansions:10000,state_limit:1000,mismatch_window_weight:1,unrestricted_sticker_count:3};
const frame={detections:[{classId:3,x:2,y:3,score:0.94,presence:0.99},{classId:0,x:3,y:4,score:0.8,presence:0.99}]} as FrameResult;
const pose={bank:[{flat_fallback:true}]} as FittedPose;
class WorkerStub {
 static instances:WorkerStub[]=[];messages:any[]=[];terminated=false;
 onmessage:any;onerror:any;
 constructor(){WorkerStub.instances.push(this);}
 postMessage(value:any){this.messages.push(value);}
 terminate(){this.terminated=true;}
}
function setup(t:any){
 const original=Object.getOwnPropertyDescriptor(globalThis,"Worker");
 Object.defineProperty(globalThis,"Worker",{value:WorkerStub,configurable:true});WorkerStub.instances=[];
 t.after(()=>{if(original)Object.defineProperty(globalThis,"Worker",original);else delete (globalThis as any).Worker;});
 const errors:string[]=[];const search=new MoveSearch("http://localhost/",config,()=>{},error=>errors.push(error),true);
 return {search,worker:WorkerStub.instances[0],errors};
}
const tick=()=>new Promise<void>(resolve=>setImmediate(resolve));
function gate(){let resolve!:(pose:FittedPose)=>void;const promise=new Promise<FittedPose>(r=>{resolve=r;});return{promise,resolve};}
test("completed future poses are prepared immediately with their original sequence",async t=>{
 const {search,worker}=setup(t),first=gate(),second=gate();
 search.submit(0,frame,first.promise);search.submit(1/30,frame,second.promise);
 second.resolve(pose);await tick();
 assert.deepEqual(worker.messages.filter(m=>m.type==="frame").map(m=>m.frame.sequence),[1]);
 assert.deepEqual(worker.messages.at(-1).frame.colors,["white"]);
 assert.equal(worker.messages.at(-1).frame.landmark_count,1);
 const ending=search.end();await tick();assert.equal(worker.messages.some(m=>m.type==="end"),false);
 first.resolve(pose);await ending;
 assert.deepEqual(worker.messages.filter(m=>m.type==="frame").map(m=>m.frame.sequence),[1,0]);
 assert.equal(worker.messages.at(-1).type,"end");search.dispose();
});
test("scrubbing cannot duplicate cached input or close a gap as if it were complete",async t=>{
 const {search,worker}=setup(t);
 search.submit(0,frame,Promise.resolve(pose));search.submit(2/30,frame,Promise.resolve(pose));await search.end();
 assert.equal(worker.messages.some(m=>m.type==="end"),false);
 search.submit(0,frame,Promise.resolve(pose));search.submit(1/30,frame,Promise.resolve(pose));await tick();
 assert.deepEqual(worker.messages.filter(m=>m.type==="frame").map(m=>m.frame.sequence),[0,2,1]);
 assert.equal(worker.messages.filter(m=>m.type==="end").length,1);search.dispose();
});
test("source replacement ignores late fits and releases the search worker",async t=>{
 const {search,worker}=setup(t),pending=gate();search.submit(0,frame,pending.promise);search.dispose();
 pending.resolve(pose);await tick();assert.equal(worker.messages.length,1);assert.equal(worker.terminated,true);
});
test("a solution closes frame admission before the result callback, including late fitted frames",async t=>{
 setup(t);
 for(const status of ["solved","already_solved"]){
  const pending=gate(),received:boolean[]=[];
  const search=new MoveSearch("http://localhost/",config,()=>{
   received.push(search.solved);
   // A completion callback must not be able to restart frame admission.
   search.submit(2/30,frame,Promise.resolve(pose));
  },()=>{},true);
  const worker=WorkerStub.instances.at(-1)!;
  search.submit(0,frame,Promise.resolve(pose));await tick();
  search.submit(1/30,frame,pending.promise);
  worker.onmessage({data:{type:"result",result:{status},ended:false}});
  pending.resolve(pose);await tick();await search.end();
  assert.deepEqual(received,[true]);
  assert.deepEqual(worker.messages.map(message=>message.type),["init","frame","end"]);
  search.dispose();
 }
});
test("a solved search can end without waiting for unused fits or turning their failures into search errors",async t=>{
 const {search,worker,errors}=setup(t);
 let reject!:(cause:Error)=>void;
 search.submit(0,frame,new Promise<FittedPose>((_resolve,fail)=>{reject=fail;}));
 worker.onmessage({data:{type:"result",result:{status:"already_solved"},ended:false}});
 await search.end();
 assert.equal(worker.messages.at(-1).type,"end");
 reject(new Error("unused fit cancelled"));await tick();
 assert.deepEqual(errors,[]);assert.equal(worker.terminated,false);
 search.dispose();
});
test("a fresh search admits frames after the previous search found a solution",async t=>{
 const previous=setup(t);
 previous.worker.onmessage({data:{type:"result",result:{status:"solved"},ended:false}});
 previous.search.dispose();
 const search=new MoveSearch("http://localhost/",config,()=>{},()=>{},true),worker=WorkerStub.instances.at(-1)!;
 assert.equal(search.solved,false);
 search.submit(0,frame,Promise.resolve(pose));await search.end();
 assert.deepEqual(worker.messages.map(message=>message.type),["init","frame","end"]);
 search.dispose();
});
test("trimmed video starts its ordinal at zero and preserves all six geometry options",async t=>{
 setup(t);const origin=6.000716;
 const search=new MoveSearch("http://localhost/",{...config,source_origin_seconds:origin},()=>{},()=>{},true);
 const bank=Array.from({length:6},(_,i)=>({flat_fallback:i===5}));
 search.submit(origin,frame,Promise.resolve({...pose,bank} as FittedPose));
 search.submit(origin+1/30,frame,Promise.resolve({...pose,bank} as FittedPose));await search.end();
 const rows=WorkerStub.instances.at(-1)!.messages.filter(m=>m.type==="frame");
 assert.deepEqual(rows.map(m=>m.frame.sequence),[0,1]);assert.equal(rows[0].frame.bank.length,6);assert.equal(rows[0].frame.bank[5].flat_fallback,true);
 search.dispose();
});
test("end acknowledgement exposes its complete journal before notifying diagnostic capture",t=>{
 setup(t);
 const received:{ended:boolean;actions:number}[]=[];
 const search=new MoveSearch("http://localhost/",config,(_value,ended)=>received.push({ended,actions:search.actions.length}),()=>{},true);
 const worker=WorkerStub.instances.at(-1)!;
 worker.onmessage({data:{type:"result",result:{status:"running"},ended:false,actions:[{type:"frame",sequence:0}]}});
 worker.onmessage({data:{type:"result",result:{status:"running"},ended:true,actions:[{type:"end"}]}});
 assert.deepEqual(received,[{ended:false,actions:1},{ended:true,actions:2}]);
 search.dispose();
});
test("disabled diagnostics retains reconstruction results without retaining a journal",t=>{
 setup(t);const received:any[]=[];
 const search=new MoveSearch("http://localhost/",{...config,diagnostics:false},value=>received.push(value),()=>{},true);
 const worker=WorkerStub.instances.at(-1)!;
 assert.equal(worker.messages[0].config.diagnostics,false);
 const result={status:"solved",steps:[{frame_index:1,geometry_option_index:0,state_moves:["R"],pose_flagged:false}]};
 worker.onmessage({data:{type:"result",result,ended:true,actions:[{type:"end"}]}});
 assert.deepEqual(received,[result]);assert.deepEqual(search.actions,[]);search.dispose();
});
test("search receives finalized pose flags and leaves legacy flag derivation available",async t=>{
 const {search,worker}=setup(t);
 for(const [i,reason] of [undefined,null,"collapsed_landmark_and_sticker_support"].entries())
  search.submit(i/30,frame,Promise.resolve({...pose,pose_flag_reason:reason}));
 await search.end();
 assert.deepEqual(worker.messages.filter(message=>message.type==="frame").map(message=>message.frame.flagged),[undefined,false,true]);
 search.dispose();
});
test("color-only warnings keep good geometry authoritative while legacy flags remain untyped",async t=>{
 const {search,worker}=setup(t);
 const cases=[
  {pose_flag_reason:"insufficient_assigned_sticker_evidence",pose_geometry_flag_reason:null},
  {pose_flag_reason:"high_robust_geometric_error",pose_geometry_flag_reason:"high_robust_geometric_error"},
  {pose_flag_reason:null,pose_geometry_flag_reason:null},
  {pose_flag_reason:"legacy_bad_pose"},
 ];
 for(const [i,flags] of cases.entries())search.submit(i/30,frame,Promise.resolve({...pose,...flags}));
 await search.end();
 assert.deepEqual(worker.messages.filter(message=>message.type==="frame").map(message=>
  [message.frame.flagged,message.frame.geometry_flagged]),[[true,false],[true,true],[false,false],[true,undefined]]);
 search.dispose();
});
