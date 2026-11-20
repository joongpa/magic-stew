import assert from 'node:assert/strict';
import test from 'node:test';
import { DetectionRecording } from './detection-recording';
import { compactDetectionCapture, searchSummary } from './diagnostic-capture';
import { letterbox, type FrameResult } from './detector';
import { pipelineFrame } from './pose-fitter';
import { validateRun, recordingRows } from './run-provenance';
import { createSearchConfig, POLICY_VERSION } from '../../../pose-fitter/runtime/reconstruction-policy.mjs';
import type { MoveSearchResult } from './move-search-types';

const settings={thresholds:[.25,.4,.25,.9,.9,.9,.9,.9,.9],presenceThreshold:.05,bboxThreshold:.25,roiPadding:.4};
const fingerprints={scheduler:'scheduler',policy:'policy',searchDriver:'search',frameAdapter:'adapter',colors:'colors'};
const runtime='cpu-pipeline.test.wasm';
function setup(){
  const recording=new DetectionRecording(),config=createSearchConfig({scramble:'R',fps:30});
  const metadata={policyVersion:POLICY_VERSION,appOrigin:'http://localhost',poseWorkers:3,admissionCapacity:12,
    source:{name:'solve.mp4',bytes:12,lastModified:1,sha256:'source'},fingerprints,
    detector:{poseSize:256,backend:'webgpu',preprocessing:'gpu',settings,
      models:{pose:{file:'pose.onnx',size:256,bytes:10,sha256:'pose'},bbox:{file:'bbox.onnx',size:640,bytes:10,sha256:'bbox'}}}};
  const id=recording.startRun(1,config,metadata);
  const frame={detections:[{classId:0,x:40,y:50,score:.65,presence:.99,box:[30,40,50,60]},
      {classId:3,x:45,y:55,score:.94,presence:.98,box:[35,45,55,65]}],
    transform:letterbox(640,480,256,false),detectorHashes:{pose:'pose',bbox:'bbox'},
    inputCrop:{x:320,y:120,width:640,height:480,sourceWidth:1920,sourceHeight:1080},
    cube:{box:[0,0,100,100],score:.85},roi:[0,0,100,100],roiSource:'tracker',trackingLost:false,
    profile:{pose:{preprocessing:'gpu'}},pipeline:{presentedAtMs:100},
    fittedPose:{sourceFrame:0,pipelineVersion:runtime,pipelineAssets:{rust:runtime,geometry:'geometry',opencvModule:'cv.mjs',opencvWasm:'cv.wasm'},
      fittingConfig:{segment:1,fps:30,turningStartFrame:null,centers:[]},
      bank:[{state:'large-derived-state',params:Array(10_000).fill(12345)}],wireframes:Array(100).fill(['mesh'])}} as unknown as FrameResult;
  recording.append(frame,{sourceId:1,sourceName:'solve.mp4',sourceKind:'video',sourceFps:30,segment:1,
    timestampSeconds:0,backend:'webgpu',settings,runId:id});
  const result={status:'running',pipelineVersion:runtime,steps:[{frame_index:0,geometry_option_index:0,state_moves:['R'],
    orientation_symmetry_index:2,state_score:.1}],stats:{evaluated_states:128},solved_frame_index:null,
    unprocessed_steps:[{state:'unneeded-derived-path'}],frontier:Array(10_000).fill('unneeded-frontier')} as unknown as MoveSearchResult;
  const actions=[{type:'frame' as const,sequence:0},{type:'advance' as const,budget:128}];
  recording.saveSearch(1,config,result,id,actions);recording.saveError(id,'Reconstruction timed out after 30 seconds');
  return {recording,frame,config,result,actions};
}

test('compact downloads retain identical fitting inputs, clocks, scores, provenance and deterministic operation order',()=>{
  const {recording,frame,result,actions}=setup(),raw=recording.toJSON();
  const capture=JSON.parse(JSON.stringify(compactDetectionCapture(raw)));
  const run=capture.runs[0];
  assert.equal(capture.format,'replay-inputs');assert.equal(capture.reconstructionTimeoutMs,30_000);
  assert.deepEqual(pipelineFrame(capture.frames[0].result,0),pipelineFrame(frame,0));
  assert.deepEqual(capture.frames[0].result.detections,frame.detections);
  assert.deepEqual(capture.frames[0].result.inputCrop,frame.inputCrop);
  assert.equal(capture.frames[0].result.cube.score,.85);
  assert.match(capture.confidenceFields['result.detections[].score'],/per-object/);
  assert.match(capture.confidenceFields['result.detections[].presence'],/presence/);
  assert.equal(recordingRows(capture,run).length,1);
  assert.equal(validateRun(capture,run,{runtime,fingerprints}).runtime,runtime);
  assert.deepEqual(run.actions,actions);assert.deepEqual(run.result,JSON.parse(JSON.stringify(searchSummary(result))));
  assert.equal(run.error,'Reconstruction timed out after 30 seconds');
  assert.equal(capture.frames[0].result.fittedPose.bank,undefined);
  assert.equal(capture.frames[0].result.fittedPose.wireframes,undefined);
  assert.equal(capture.moveSearches,undefined);
  const text=JSON.stringify(capture);
  for(const value of ['large-derived-state','unneeded-frontier','unneeded-derived-path'])assert.equal(text.includes(value),false);
  assert.ok(text.length<JSON.stringify(raw).length/10);
  assert.ok(frame.fittedPose?.bank.length,'export must not mutate the live viewer');
});

test('failed fitting frames and unlinked image records are retained without inventing a pose',()=>{
  const {recording,frame}=setup();
  const failed={...frame,fittedPose:undefined,fittingError:'Original fitting failure'};
  recording.append(failed,{sourceId:2,sourceName:'image.png',sourceKind:'image',sourceFps:0,segment:7,
    timestampSeconds:1.123456,backend:'webgpu',settings});
  const capture=compactDetectionCapture(recording.toJSON());
  assert.equal(capture.sources.length,2);assert.equal(capture.frames.length,2);
  assert.equal(capture.frames[1].result.fittingError,'Original fitting failure');
  assert.equal(capture.frames[1].timestampSeconds,1.123456);
  assert.equal(capture.frames[1].result.fittedPose,undefined);
});
