import assert from 'node:assert/strict';
import test from 'node:test';
import {IncrementalSearch} from '../../../pose-fitter/runtime/incremental-search.mjs';
import {createSearchConfig,videoFrameIdentity} from '../../../pose-fitter/runtime/reconstruction-policy.mjs';
import {validateRun,recordingRows} from './run-provenance';

const settings={thresholds:[.25,.4,.25,.9,.9,.9,.9,.9,.9],presenceThreshold:.05,bboxThreshold:.25,roiPadding:.4};
const fingerprints={scheduler:'scheduler-sha',policy:'policy-sha',searchDriver:'driver-sha',frameAdapter:'adapter-sha',colors:'colors-sha'};
function capture(size=256){
 const config=createSearchConfig({scramble:'R U',fps:60,originSeconds:6,openingHoldSeconds:.1,startSeconds:6,endSeconds:8});
 const run={id:'run-a',sourceId:1,config,metadata:{policyVersion:1,appOrigin:'http://localhost:5177',poseWorkers:3,admissionCapacity:12,
  detector:{poseSize:size,backend:'webgpu',preprocessing:'gpu',settings,
   models:{pose:{file:'pose.onnx',size,bytes:10,sha256:'pose-sha'},bbox:{file:'bbox.onnx',size:640,bytes:10,sha256:'bbox-sha'}}},
  source:{name:'solve.mp4',bytes:10,lastModified:1,sha256:'file-sha'},fingerprints},
  result:{pipelineVersion:'cpu-pipeline.test.wasm'}};
 const frames=Array.from({length:3},(_,i)=>({sourceId:1,runId:'run-a',segment:1,timestampSeconds:6+i/30,
  backend:'webgpu',settings,result:{detectorHashes:{pose:'pose-sha',bbox:'bbox-sha'},transform:{size},profile:{pose:{preprocessing:'gpu'}},
   fittedPose:{pipelineVersion:'cpu-pipeline.test.wasm',fittingConfig:{fps:60,turningStartFrame:366}}}}));
 return {recording:{sources:[{id:1,name:'solve.mp4',fps:60}],frames},run};
}
test('trim clock and hold use one absolute source-frame convention',()=>{
 const {run}=capture();assert.equal(run.config.turning_start_frame,366);
 assert.deepEqual(videoFrameIdentity(6,run.config),{sourceFrame:360,sequence:0});
 assert.deepEqual(videoFrameIdentity(6+1/30,run.config),{sourceFrame:362,sequence:1});
 const {recording}=capture();assert.equal(validateRun(recording,run,{fingerprints,runtime:'cpu-pipeline.test.wasm'}).poseSize,256);
});
test('new runs record motion cost explicitly while retaining source clocks and TPS limits',()=>{
 const config=createSearchConfig({scramble:'R U',fps:60});
 assert.equal(config.motion_cost_weight,.025);
 assert.equal(config.max_tps,54);assert.equal(config.max_rotation_tps,20);
 assert.equal(createSearchConfig({scramble:'R U',motionCostWeight:0}).motion_cost_weight,0);
 for(const motionCostWeight of [-1,Infinity,NaN])assert.throws(()=>createSearchConfig({scramble:'R U',motionCostWeight}));
});
test('new runs enable post-solve identity cleanup with a configurable segment mean color tolerance',()=>{
 const config=createSearchConfig({scramble:'R U'});
 assert.equal(config.postprocess_identity_loops,true);
 assert.equal(config.postprocess_color_tolerance,.02);
 assert.equal(config.postprocess_pose_cleanup,true);
 assert.equal(createSearchConfig({scramble:'R U',postprocessIdentityLoops:false}).postprocess_identity_loops,false);
 assert.equal(createSearchConfig({scramble:'R U',postprocessIdentityLoops:false}).postprocess_pose_cleanup,false);
 assert.equal(createSearchConfig({scramble:'R U',postprocessColorTolerance:0}).postprocess_color_tolerance,0);
 for(const postprocessColorTolerance of [-.01,1.01,Infinity,NaN])
  assert.throws(()=>createSearchConfig({scramble:'R U',postprocessColorTolerance}));
});
test('bidirectional workers buffer fitted frames without advancing until input closes',()=>{
 const requests:any[]=[],tasks:(()=>void)[]=[];
 const driver=new IncrementalSearch(request=>{requests.push(request);return {status:request.type==='end'?'running':'waiting_for_frames'};},
  {report:()=>{},error:e=>{throw e;},now:()=>0,
   schedule:((task:()=>void)=>{tasks.push(task);return tasks.length;}) as any,cancel:(()=>{}) as any});
 const config=createSearchConfig({scramble:'R U',bidirectional:true});
 assert.equal(config.bidirectional,true);
 driver.request({type:'init',config});
 driver.request({type:'frame',frame:{sequence:0}});
 driver.request({type:'frame',frame:{sequence:1}});
 assert.equal(tasks.length,0);
 assert.equal(requests.some(request=>request.type==='advance'),false);
 driver.request({type:'end'});assert.equal(tasks.length,1);
 tasks.shift()!();assert.equal(requests.at(-1).type,'advance');driver.dispose();
});
test('old 160px detections cannot silently stand in for a 256px browser run',()=>{
 const {recording,run}=capture(160);
 assert.throws(()=>validateRun(recording,run,{expectedSize:256}),/160px/);
 run.metadata.detector.poseSize=256;
 assert.throws(()=>validateRun(recording,run),/saved run declares 256/);
});
test('changed source FPS, thresholds, preprocessing and runtime fail before fitting',()=>{
 for(const modify of [
  (r:any)=>r.run.config.fps=30,
  (r:any)=>r.recording.frames[1].settings={...settings,roiPadding:.5},
  (r:any)=>r.recording.frames[1].result.profile.pose.preprocessing='cpu',
  (r:any)=>r.recording.frames[1].result.fittedPose.pipelineVersion='old-runtime',
  (r:any)=>r.recording.frames[1].result.detectorHashes.pose='other-weights',
 ]){const r=capture();modify(r);assert.throws(()=>validateRun(r.recording,r.run));}
 const r=capture();assert.throws(()=>validateRun(r.recording,r.run,{fingerprints:{...fingerprints,policy:'old-policy'}}),/differs/);
});
test('legacy data and missing frames are explicit diagnostic cases',()=>{
 const {recording,run}=capture();delete (run as any).metadata;
 assert.throws(()=>validateRun(recording,run),/provenance/);
 assert.equal(validateRun(recording,run,{diagnostic:true}).mode,'diagnostic-ablation');
 recording.frames.splice(1,1);assert.throws(()=>validateRun(recording,run,{diagnostic:true}),/Missing\/duplicated/);
});
test('a cached restart links exact raw frames without merging other runs',()=>{
 const {recording,run}=capture();recording.frames.push({...recording.frames[0],runId:'other-run'});
 assert.equal(recordingRows(recording,run).length,3);
 assert.equal(recordingRows(recording,{...run,frameSequences:[0,1,2]}).length,3);
});
test('the shared driver journals frame arrival and expansions for fast exact replay',()=>{
 function engine(){let received=0,expanded=0,ended=false;return (r:any)=>{
  if(r.type==='frame')received++;if(r.type==='advance')expanded+=received;if(r.type==='end')ended=true;
  return {status:ended?'solved':'waiting_for_frames',received,expanded};
 };}
 const tasks:(()=>void)[]=[],actions:any[]=[],reports:any[]=[];
 const live=new IncrementalSearch(engine(),{report:m=>{reports.push(m.result);actions.push(...m.actions);},error:e=>{throw e;},
  schedule:((fn:()=>void)=>{tasks.push(fn);return tasks.length;}) as any,cancel:(()=>{}) as any,now:()=>0});
 live.request({type:'init',config:{}});live.request({type:'frame',frame:{sequence:0}});tasks.shift()!();
 live.request({type:'frame',frame:{sequence:1}});live.request({type:'end'});tasks.shift()!();
 const replayReports:any[]=[];
 const replay=new IncrementalSearch(engine(),{live:false,now:()=>0,report:m=>replayReports.push(m.result),error:e=>{throw e;}});
 replay.request({type:'init',config:{}});
 for(const action of actions)replay.request(action.type==='frame'?{type:'frame',frame:{sequence:action.sequence}}:action);
 assert.deepEqual(replayReports.at(-1),reports.at(-1));
 assert.deepEqual(actions.map(a=>a.type),['frame','advance','frame','end','advance']);
});
test('host timers receive the global receiver, as browser WebIDL timers require',()=>{
 let cancelled=false;
 const driver=new IncrementalSearch(()=>({status:'waiting_for_frames'}),{report:()=>{},error:e=>{throw e;},
  schedule:(function(this:unknown){assert.equal(this,globalThis);return 7;}) as any,
  cancel:(function(this:unknown,id:number){assert.equal(this,globalThis);assert.equal(id,7);cancelled=true;}) as any});
 driver.request({type:'frame',frame:{sequence:0}});driver.dispose();assert.equal(cancelled,true);
});
test('diagnostics can be disabled without changing worker scheduling or results',()=>{
 assert.equal(createSearchConfig({scramble:'R U'}).diagnostics,true);
 assert.equal(createSearchConfig({scramble:'R U',diagnostics:false}).diagnostics,false);
 const reports:any[]=[],requests:any[]=[],tasks:(()=>void)[]=[];
 let ticks=0;
 const driver=new IncrementalSearch(request=>{requests.push(request);return {status:request.type==='end'?'solved':'waiting_for_frames',steps:[]};},
  {report:m=>reports.push(m),error:e=>{throw e;},now:()=>{ticks++;return 0;},
   schedule:((fn:()=>void)=>{tasks.push(fn);return tasks.length;}) as any,cancel:(()=>{}) as any});
 driver.request({type:'init',config:{diagnostics:false}});
 driver.request({type:'frame',frame:{sequence:0}});tasks.shift()!();
 driver.request({type:'end'});tasks.shift()!();
 assert.deepEqual(requests.map(r=>r.type),['init','frame','advance','end','advance']);
 assert.equal(ticks,requests.length,'Only publication throttling needs a clock read');
 assert.ok(reports.every(m=>m.actions.length===0&&m.result.timings===undefined));
 assert.ok(reports.some(m=>m.ended&&m.result.status==='solved'));
 driver.request({type:'init',config:{}});
 driver.request({type:'frame',frame:{sequence:0}});
 assert.deepEqual(reports.at(-1).actions,[{type:'frame',sequence:0}]);
 assert.ok(reports.at(-1).result.timings);
 driver.dispose();
});
