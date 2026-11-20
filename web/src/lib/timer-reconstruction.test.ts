import assert from 'node:assert/strict';
import test, {type TestContext} from 'node:test';
import { TimerReconstruction } from './timer-reconstruction';
import { TIMER_SEARCH_LIMIT_MS, TIMER_TAIL_SECONDS } from './cube-timer';
import type { TimerSearch, TimerSearchFactory } from './timer-reconstruction';
import type { FrameResult } from './detector';
import type { MoveSearchConfig, MoveSearchResult } from './move-search-types';
import type { TimerReconstructionResult } from './cube-timer';
import type {FittedPose} from './pose-types';
import {fakeTimers} from './test-clock';
const frame = { detections: [], fittedPose: { bank: [] } } as unknown as FrameResult;
let clock:ReturnType<typeof fakeTimers>;
test.beforeEach(t=>{clock=fakeTimers(t as TestContext);});
function setup() {
  const updates = new Map<string, TimerReconstructionResult>();
  const searches: { config: MoveSearchConfig; rows: number[]; ended: boolean; disposed: boolean;paused:boolean;pacingMs:number;
    result: Parameters<TimerSearchFactory>[1]; fail: Parameters<TimerSearchFactory>[2] }[] = [];
  const pipeline = new TimerReconstruction((config, result, fail): TimerSearch => {
    const search = { config, result, fail, rows: [] as number[], ended: false, disposed: false,paused:false,pacingMs:0 }; searches.push(search);
    return { submit: timestamp => { search.rows.push(timestamp); }, end: async () => { search.ended = true; }, dispose: () => { search.disposed = true; },
      pause(){search.paused=true;},resume(){search.paused=false;},setPacing(delayMs){search.pacingMs=delayMs;} };
  }, (id, update) => updates.set(id, update));
  return { pipeline, updates, searches };
}
const result = (status: MoveSearchResult['status']): MoveSearchResult => ({ status, steps: [{ frame_index: 1, geometry_option_index: 0, state_moves: ['R'] }] }) as MoveSearchResult;
test('working-data cleanup runs once after final input and search completion, including timeouts and early failures',()=>{
 const finished:string[]=[],callbacks:Parameters<TimerSearchFactory>[1][]=[];
 const timer=new TimerReconstruction((_config,onResult)=>{
   callbacks.push(onResult);return {submit(){},async end(){},dispose(){}};
 },()=>{},id=>finished.push(id));
 timer.start('solved','R',0,30);timer.accept(0,frame);timer.stop('solved',1);
 callbacks[0](result('solved'),false);assert.deepEqual(finished,[],'retain diagnostic inputs until the tail drains');
 timer.finishThrough(1.3);callbacks[0](result('solved'),true);assert.deepEqual(finished,['solved']);
 timer.start('timeout','R',2,30);timer.accept(2,frame);timer.stop('timeout',3);timer.finishThrough(3.3);
 clock.tick(30_300);assert.deepEqual(finished,['solved','timeout']);
 timer.start('failed','R',4,30);timer.stop('failed',5);timer.accept(4,{detections:[]} as unknown as FrameResult);
 timer.finishThrough(5.3);timer.cancel();assert.deepEqual(finished,['solved','timeout','failed']);
});
test('the post-stop deadline terminates a stalled search even before fitting and input closure finish',t=>{
  const {pipeline,searches,updates}=setup();
  pipeline.start('a','R',10,30);pipeline.accept(10,frame);
  clock.tick(60_000);assert.equal(searches[0].disposed,false,'solving itself has no reconstruction deadline');
  searches[0].result(result('running'),false);pipeline.stop('a',20);
  clock.tick(TIMER_TAIL_SECONDS*1000+TIMER_SEARCH_LIMIT_MS-1);
  assert.equal(searches[0].disposed,false);
  clock.tick(1);assert.equal(searches[0].disposed,true);
  assert.equal(updates.get('a')?.status,'failed');
  assert.equal(updates.get('a')?.message,'Reconstruction timed out after 30 seconds');
  assert.deepEqual(updates.get('a')?.moves,['R'],'last candidate remains available for inspection');
  searches[0].result(result('solved'),true);assert.equal(updates.get('a')?.status,'failed','late worker results cannot overwrite a timeout');
  assert.equal(pipeline.includes(10),true,'remaining processed solve frames still belong to diagnostics/replay');
  pipeline.finishThrough(20.3);assert.equal(pipeline.includes(10),false);
});
test('closed-input searches retain the same deadline and a subsequent solve gets a fresh worker',t=>{
  const {pipeline,searches,updates}=setup();
  pipeline.start('a','R',0,30);pipeline.accept(0,frame);pipeline.stop('a',1);pipeline.finishThrough(1.3);
  clock.tick(TIMER_TAIL_SECONDS*1000+TIMER_SEARCH_LIMIT_MS);
  assert.equal(searches[0].disposed,true);assert.equal(updates.get('a')?.status,'failed');
  pipeline.start('b','U',20,30);pipeline.accept(20,frame);pipeline.stop('b',21);pipeline.finishThrough(21.3);
  assert.equal(searches[1].disposed,false);
  searches[1].result(result('solved'),true);
  clock.tick(20_000);assert.equal(updates.get('b')?.status,'solved');
});
test('a solution clears the deadline even before the end acknowledgement',t=>{
  const {pipeline,searches,updates}=setup();
  pipeline.start('a','R',0,30);pipeline.accept(0,frame);pipeline.stop('a',1);
  searches[0].result(result('solved'),false);clock.tick(30_000);
  assert.equal(updates.get('a')?.status,'solved');assert.equal(searches[0].disposed,false);
  pipeline.finishThrough(1.3);searches[0].result(result('solved'),true);assert.equal(searches[0].disposed,true);
});
test('search starts at the first finalized solve pose and streams every pose until stop + 0.3s', () => {
  const { pipeline, searches, updates } = setup();
  pipeline.start('a', 'R U', 10, 30);
  pipeline.accept(9.99, frame); pipeline.accept(10, frame);
  assert.equal(searches.length,1);assert.deepEqual(searches[0].rows,[10]);assert.equal(searches[0].ended,false);
  pipeline.stop('a', 11);
  // These arrive after timer stop, but remain within its source-time window.
  pipeline.accept(10.5, frame); pipeline.accept(11, frame); pipeline.accept(11.299, frame);
  assert.equal(searches.length, 1); assert.equal(updates.get('a')?.status, 'waiting');assert.equal(searches[0].ended,false);
  pipeline.accept(11.3, frame);
  assert.deepEqual(searches[0].rows, [10, 10.5, 11, 11.299]);
  assert.equal(searches[0].config.requested_start_seconds, 10);
  assert.equal(searches[0].config.source_end_seconds, 11.3);
  assert.equal(searches[0].config.source_origin_seconds, 10);
  assert.equal(searches[0].ended, true);
});
test('source origin uses first admitted frame, and consecutive searches cannot absorb each other’s frames', () => {
  const { pipeline, searches, updates } = setup();
  pipeline.start('a', 'R', 0.01, 30); pipeline.stop('a', 1);
  pipeline.accept(0, frame); pipeline.accept(0.0333, frame); pipeline.accept(1.3, frame);
  assert.equal(searches[0].config.source_origin_seconds, 0.0333);
  pipeline.start('b', 'U', 2, 30); pipeline.stop('b', 3);
  pipeline.accept(2, frame); pipeline.accept(3.3, frame);
  assert.equal(searches.length, 2); assert.equal(searches[0].paused,true);assert.equal(searches[1].paused,false);
  searches[0].result(result('solved'), false);
  assert.equal(searches[0].disposed, false, 'retain worker until cleanup end acknowledgement');
  searches[0].result(result('solved'), true);
  assert.equal(searches[0].disposed, true); assert.equal(searches.length, 2);
  assert.deepEqual(searches[1].rows, [2]); assert.equal(searches[1].config.scramble, 'U');
  searches[0].result(result('state_limit'), true);
  assert.equal(updates.get('a')?.status, 'solved', 'late callbacks from disposed search are ignored');
});
test('missing fits, interrupted capture and worker failures report explicit failure', () => {
  const { pipeline, searches, updates } = setup();
  pipeline.start('missing', 'R', 0, 30); pipeline.stop('missing', 1);
  pipeline.accept(0, { detections: [] } as unknown as FrameResult); pipeline.accept(1.3, frame);
  assert.equal(updates.get('missing')?.status, 'failed'); assert.equal(searches.length, 0);
  pipeline.start('interrupted', 'R', 2, 30); pipeline.stop('interrupted', 3); pipeline.closeInput();
  assert.equal(updates.get('interrupted')?.status, 'failed');
  pipeline.start('worker', 'R', 4, 30); pipeline.stop('worker', 5); pipeline.accept(4, frame); pipeline.accept(5.3, frame);
  searches[0].fail('worker failed'); assert.equal(updates.get('worker')?.message, 'worker failed');
  assert.equal(searches[0].disposed, true);
});
test('page cleanup cancels all searches and ignores stale results', () => {
  const { pipeline, searches, updates } = setup();
  for (const [id, start] of [['a', 0], ['b', 2]] as const) {
    pipeline.start(id, 'R', start, 30); pipeline.stop(id, start + 1);
    pipeline.accept(start, frame); pipeline.accept(start + 1.3, frame);
  }
  pipeline.cancel(); searches[0].result(result('solved'), true);
  assert.equal(searches[0].disposed, true); assert.equal(searches[1].disposed,true);assert.equal(searches.length, 2);
  assert.equal(updates.get('a')?.status, 'failed'); assert.equal(updates.get('b')?.status, 'failed');
});

test('disconnecting camera preserves complete searches and fails only unfinished input windows',()=>{
 const {pipeline,searches,updates}=setup();
 pipeline.start('complete','R',0,30);pipeline.accept(0,frame);pipeline.stop('complete',1);pipeline.accept(1.3,frame);
 pipeline.start('unfinished','U',2,30);pipeline.accept(2,frame);
 assert.equal(searches[0].paused,true);
 pipeline.closeInput('Camera disconnected');
 assert.equal(searches[0].disposed,false);assert.equal(searches[0].paused,false);
 assert.equal(searches[1].disposed,true);assert.equal(updates.get('unfinished')?.message,'Camera disconnected');
 searches[0].result(result('solved'),true);assert.equal(updates.get('complete')?.status,'solved');
});

test('newest solve searches immediately while an older expansion is paused, then resumes it',()=>{
 const {pipeline,searches}=setup();
 pipeline.start('first','R',0,30);pipeline.accept(0,frame);pipeline.stop('first',1);pipeline.accept(1.3,frame);
 pipeline.start('second','U',2,30);pipeline.accept(2,frame);pipeline.accept(2.1,frame);
 assert.deepEqual(searches[1].rows,[2,2.1]);assert.equal(searches[0].paused,true);
 pipeline.stop('second',3);pipeline.accept(3.3,frame);searches[1].result(result('solved'),true);
 assert.equal(searches[0].paused,false);assert.equal(searches[0].disposed,false);
});
test('camera backpressure spaces expansion chunks without starving search or delaying pose admission',()=>{
 const {pipeline,searches}=setup();pipeline.start('a','R',0,30);pipeline.accept(0,frame);
 pipeline.setBackpressure(true);pipeline.accept(.033,frame);pipeline.accept(.066,frame);
 assert.equal(searches[0].paused,false);assert.equal(searches[0].pacingMs,25);assert.deepEqual(searches[0].rows,[0,.033,.066]);
 pipeline.setBackpressure(false);assert.equal(searches[0].pacingMs,0);
 pipeline.stop('a',1);pipeline.accept(1.3,frame);assert.equal(searches[0].ended,true);
});
test('a closed solve removes camera pacing even while unrelated live frames keep the camera buffer full',()=>{
 const {pipeline,searches}=setup();pipeline.start('a','R',0,30);pipeline.accept(0,frame);
 pipeline.setBackpressure(true);assert.equal(searches[0].pacingMs,25);
 pipeline.stop('a',1);pipeline.accept(1.3,frame);
 assert.equal(searches[0].ended,true);assert.equal(searches[0].pacingMs,0);
 pipeline.setBackpressure(false);pipeline.setBackpressure(true);assert.equal(searches[0].pacingMs,0);
});
test('a rejected fit reports its original error before display without dropping solve-only recording membership',async()=>{
 const {pipeline,searches,updates}=setup();
 pipeline.start('a','R',10,30);
 pipeline.submit(10,frame,Promise.reject(new Error('CPU pose prepare: test failure')));
 await new Promise<void>(resolve=>setImmediate(resolve));
 assert.equal(searches[0].disposed,true);
 assert.equal(updates.get('a')?.message,'Pose fitting failed at 10.000s: CPU pose prepare: test failure');
 assert.equal(pipeline.includes(10),true,'failed in-flight frames still need to be linked to diagnostics by display completion');
 pipeline.stop('a',11);pipeline.accept(10,{detections:[],fittingError:'CPU pose prepare: test failure'} as unknown as FrameResult);
 assert.equal(updates.get('a')?.frames,1,'presentation does not count the frame twice');
 pipeline.accept(11.3,frame);assert.equal(pipeline.includes(10),false);
});
test('source cleanup ignores fitting promises that finalize after their solve was cancelled',async()=>{
 const {pipeline,searches,updates}=setup();
 let release!:(pose:FittedPose)=>void;
 const fitting=new Promise<FittedPose>(resolve=>{release=resolve;});
 pipeline.start('a','R',0,30);pipeline.submit(0,frame,fitting);pipeline.cancel('Cancelled');
 release(frame.fittedPose!);await fitting;
 assert.equal(searches[0].disposed,true);assert.equal(updates.get('a')?.message,'Cancelled');
});
test('detection recording excludes idle time and uses source timestamps for delayed solve frames', () => {
  const { pipeline } = setup();
  assert.equal(pipeline.includes(0), false, 'connected camera before the first solve');
  pipeline.start('a', 'R', 10, 30);
  assert.equal(pipeline.includes(9.99), false, 'an idle frame still in the inference queue');
  assert.equal(pipeline.includes(10), true, 'inclusive start');
  pipeline.stop('a', 11);
  assert.equal(pipeline.includes(10.5), true, 'solve frame completing after timer stop');
  assert.equal(pipeline.includes(11.299), true, 'post-stop margin');
  assert.equal(pipeline.includes(11.3), false, 'exclusive end');
  assert.equal(pipeline.includes(12), false, 'idle time after this solve');
  pipeline.accept(10, frame); pipeline.accept(11.3, frame);
  assert.equal(pipeline.includes(12), false, 'idle while search runs');
  pipeline.start('b', 'U', 20, 30);
  assert.equal(pipeline.includes(19.99), false, 'idle backlog from between solves');
  assert.equal(pipeline.includes(20), true, 'next solve');
  pipeline.cancel(); assert.equal(pipeline.includes(21), false, 'cancelled solve');
});
