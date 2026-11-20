import assert from 'node:assert/strict';
import test from 'node:test';
import {TimerReplayStore,replayResult,type TimerReplay} from './timer-replay';
import type {FrameResult} from './detector';
const result={} as FrameResult;
const tick=()=>new Promise<void>(resolve=>setImmediate(resolve));
test('an empty completed capture publishes its boundary so the replay working cache can release it',async()=>{
 const clips:TimerReplay[]=[],store=new TimerReplayStore(clip=>clips.push(clip));
 store.start('empty',10,1000);store.stop('empty',11);store.finishThrough(11.3);await tick();
 assert.equal(clips.length,1);assert.equal(clips[0].complete,true);assert.deepEqual(clips[0].frames,[]);
});
test('replay encodes only processed solve frames including the tail, with exact source timestamps',async()=>{
  const clips:TimerReplay[]=[],store=new TimerReplayStore(clip=>clips.push(clip));let encoded=0;
  const encode=()=>{encoded++;return Promise.resolve(new Blob(['frame']));};
  store.capture(0,result,encode);store.start('a',10,1000);
  store.capture(9.99,result,encode);store.capture(10.042,result,encode);
  store.stop('a',11);store.capture(11.299,result,encode);store.capture(11.3,result,encode);store.capture(12,result,encode);
  await tick();assert.equal(encoded,2);assert.equal(clips.length,1);
  assert.deepEqual(clips[0].frames.map(row=>row.timestamp),[10.042,11.299]);
  assert.equal(clips[0].start,10);assert.equal(clips[0].end,11.3);assert.equal(clips[0].clockFps,1000);
  assert.ok(clips[0].frames.every(row=>row.result!==result&&row.image?.size===5));
});

test('replay drops raw detections, assignments, fitting metrics and diagnostic arrays while keeping render geometry',()=>{
 const pose={bank:[{axis:1,params:[1,2,3,4,5,6],angles:[0,0,5],flat_fallback:false,score:.1,
   nearest:[1,2,3],valid:[true,true,true],metrics:{large:[1,2,3]},assignment_objects:[[1,2,3]]}],
   pose:{rvec:[1,2,3],tvec:[4,5,6]},wireframe:[],wireframes:[[]],sourceFrame:640007,
   continuity_core:[4,5],final_bank_diagnostics:{large:[1,2,3]}};
 const compact=replayResult({detections:[{classId:1}],fittedPose:pose} as unknown as FrameResult);
 assert.deepEqual(Object.keys(compact),['fittedPose']);
 assert.equal(compact.fittedPose?.sourceFrame,640007);
 assert.deepEqual(compact.fittedPose?.bank[0].params,pose.bank[0].params);
 assert.deepEqual(compact.fittedPose?.bank[0].nearest,[]);
 assert.equal(compact.fittedPose?.bank[0].metrics,undefined);
 assert.equal(compact.fittedPose?.continuity_core,undefined);
 assert.notEqual(compact.fittedPose,pose);assert.notEqual(compact.fittedPose?.bank[0],pose.bank[0]);
});
test('slow replay encoding never blocks capture and cannot publish a deleted solve',async()=>{
  const clips:TimerReplay[]=[],store=new TimerReplayStore(clip=>clips.push(clip));let resolve!:(blob:Blob)=>void;
  store.start('a',0,1000);store.stop('a',1);
  store.capture(.2,result,()=>new Promise(value=>{resolve=value;}));store.capture(1.3,result,()=>{throw Error('Outside solve');});
  assert.equal(clips.length,0);store.remove('a');resolve(new Blob(['old']));await tick();assert.equal(clips.length,0);
});
test('replay survives an image encoding failure and disconnect preserves captured frames',async()=>{
  const clips:TimerReplay[]=[],store=new TimerReplayStore(clip=>clips.push(clip));store.start('a',0,1000);
  store.capture(.1,result,()=>{throw Error('No encoder');});
  store.capture(.2,result,()=>Promise.reject(Error('Encoding failed')));store.closeInput();await tick();
  assert.equal(clips.length,1);assert.equal(clips[0].frames.length,2);assert.equal(clips[0].missingImages,2);
});
test('consecutive solve replays discard idle frames and retain separate images/results',async()=>{
  const clips:TimerReplay[]=[],store=new TimerReplayStore(clip=>clips.push(clip));let encoded=0;
  const encode=()=>{encoded++;return Promise.resolve(new Blob());};
  for(const [id,start] of [['a',0],['b',10]] as const){
    store.start(id,start,1000);store.stop(id,start+1);store.capture(start+.5,result,encode);store.capture(start+1.3,result,encode);
  }
  store.capture(15,result,encode);await tick();assert.equal(encoded,2);assert.deepEqual(clips.map(clip=>clip.frames.length),[1,1]);
});
test('stopping publishes available pose replay immediately and appends tail frames before input closes',async()=>{
  const clips:TimerReplay[]=[],store=new TimerReplayStore(clip=>clips.push(clip));
  store.start('a',10,1000);store.capture(10.2,result,()=>Promise.resolve(new Blob(['first'])));await tick();
  assert.equal(clips.length,0,'replay is not mounted while the timer is running');
  store.stop('a',11);assert.equal(clips.length,1,'finished poses are available synchronously on stop');
  assert.equal(clips[0].complete,false);assert.equal(clips[0].frames.length,1);
  store.capture(11.2,result,()=>Promise.resolve(new Blob(['tail'])));
  await new Promise(resolve=>setTimeout(resolve,130));
  assert.equal(clips.at(-1)?.frames.length,2);assert.equal(clips.at(-1)?.complete,false);
  store.finishThrough(11.3);await tick();assert.equal(clips.at(-1)?.complete,true);
  assert.deepEqual(clips.at(-1)?.frames.map(row=>row.timestamp),[10.2,11.2]);
});

test('starting another solve drops old pending image encodes while retaining their projected replay frames',async()=>{
 const clips:TimerReplay[]=[],store=new TimerReplayStore(clip=>clips.push(clip));let resolve!:(image:Blob)=>void;
 store.start('old',0,1000);store.stop('old',1);
 store.capture(.2,result,()=>new Promise(value=>{resolve=value;}));
 store.start('latest',10,1000);resolve(new Blob(['old footage']));
 store.finishThrough(1.3);await tick();
 assert.equal(clips.at(-1)?.id,'old');assert.equal(clips.at(-1)?.frames.length,1);
 assert.equal(clips.at(-1)?.frames[0].image,undefined);
});
