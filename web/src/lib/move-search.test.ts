import assert from "node:assert/strict";
import test from "node:test";
import { MoveSearch } from "./move-search";
import type { MoveSearchConfig } from "./move-search-types";
import type { FrameResult } from "./detector";
import type { FittedPose } from "./pose-types";
const config:MoveSearchConfig={scramble:"R U",fps:60,max_tps:54,turning_start_frame:null,branch_widths:[3],
 branch_stage_expansions:10000,state_limit:1000,mismatch_window_weight:1,unrestricted_sticker_count:3};
const frame={detections:[{classId:3,x:2,y:3,score:0.94,presence:0.99}]} as FrameResult;
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
