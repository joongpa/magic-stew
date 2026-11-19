import assert from 'node:assert/strict';
import test from 'node:test';
import {TimerReplayStore,type TimerReplay} from './timer-replay';
import type {FrameResult} from './detector';
const result={} as FrameResult;
const tick=()=>new Promise<void>(resolve=>setImmediate(resolve));
test('replay encodes only processed solve frames including the tail, with exact source timestamps',async()=>{
  const clips:TimerReplay[]=[],store=new TimerReplayStore(clip=>clips.push(clip));let encoded=0;
  const encode=()=>{encoded++;return Promise.resolve(new Blob(['frame']));};
  store.capture(0,result,encode);store.start('a',10,1000);
  store.capture(9.99,result,encode);store.capture(10.042,result,encode);
  store.stop('a',11);store.capture(11.299,result,encode);store.capture(11.3,result,encode);store.capture(12,result,encode);
  await tick();assert.equal(encoded,2);assert.equal(clips.length,1);
  assert.deepEqual(clips[0].frames.map(row=>row.timestamp),[10.042,11.299]);
  assert.equal(clips[0].start,10);assert.equal(clips[0].end,11.3);assert.equal(clips[0].clockFps,1000);
  assert.ok(clips[0].frames.every(row=>row.result===result&&row.image?.size===5));
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
