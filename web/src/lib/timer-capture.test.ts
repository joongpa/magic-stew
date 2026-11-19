import assert from 'node:assert/strict';
import test from 'node:test';
import {TimerCaptureGate} from './timer-capture';
import {TimerReconstruction} from './timer-reconstruction';
import {PoseFitter} from './pose-fitter';
import {OrderedStage} from './ordered-stage';
import type {FrameResult} from './detector';

test('timer capture is idle until start and emits one boundary after the complete stop tail',()=>{
 const gate=new TimerCaptureGate();assert.deepEqual(gate.admit(0),{type:'idle'});
 gate.start('a',10);assert.equal(gate.admit(9.999).type,'idle');
 assert.deepEqual(gate.admit(10.01),{type:'frame',id:'a',reset:true});
 assert.deepEqual(gate.admit(10.04),{type:'frame',id:'a',reset:false});
 gate.stop('a',11);assert.equal(gate.admit(11.299).type,'frame');
 assert.deepEqual(gate.admit(11.3),{type:'end',id:'a',timestamp:11.3});
 for(const stamp of [11.33,12,13])assert.equal(gate.admit(stamp).type,'idle');
 assert.equal(gate.finish('a'),true);assert.equal(gate.admit(14).type,'idle');
});
test('an earlier draining solve cannot stop or reset a newer capture window',()=>{
 const gate=new TimerCaptureGate();gate.start('a',0);gate.stop('a',1);gate.admit(1.3);
 gate.start('b',2);assert.equal(gate.finish('a'),false);gate.stop('a',3);
 assert.deepEqual(gate.admit(2.01),{type:'frame',id:'b',reset:true});
 gate.clear();assert.equal(gate.admit(3).type,'idle');
});
test('stopping detector input flushes buffered lookahead and closes search after all solve frames are presented',async()=>{
 const camera={rvec:[0,0,0],tvec:[0,0,500]},bank=[{params:[0,0,0,0,0,500,0,0],angles:[0,0,0]}];
 let jobs=0,ended=false,submitted=0,presented=0;
 const fitter=new PoseFitter('http://localhost/',{async execute(job){jobs++;
   if(job.type==='prepare')return {candidates:[camera]};
   if(job.type==='select_startup')return {source:camera,anchor:true};
   if(job.type==='target')return {target:500};
   return {bank,pose:camera,reference:{frame:0,rvec:[0,0,0]},continuity_core:[0,0]};
 },dispose(){}});
 const search=new TimerReconstruction(()=>({submit(){submitted++;},async end(){ended=true;},dispose(){}}),()=>{});
 const frame={detections:[0,1,2,3].map(classId=>({classId,x:10,y:10,presence:1})),
   transform:{sourceWidth:640,sourceHeight:480,cropWidth:400,cropHeight:400}} as FrameResult;
 const gate=new TimerCaptureGate();gate.start('a',0);gate.stop('a',.2);search.start('a','R',0,30);search.stop('a',.2);
 const queue=new OrderedStage<any>(24,async item=>{
   if(item.end){search.finishThrough(item.timestamp);gate.finish('a');}
   else{const fittedPose=await item.fitting;presented++;search.accept(item.timestamp,{...frame,fittedPose});}
 },()=>{});
 for(let i=0;i<=15;i++){
   const timestamp=i/30,admission=gate.admit(timestamp);
   if(admission.type==='frame'){
     const fitting=fitter.fit(frame,admission.reset,timestamp,30,{camera:true,scramble:'',turningStartFrame:null});
     search.submit(timestamp,frame,fitting);queue.enqueue({timestamp,fitting});
   }else if(admission.type==='end'){void fitter.flush();queue.enqueue({end:true,timestamp});}
 }
 await queue.whenIdle();assert.equal(submitted,15);assert.equal(presented,15);assert.equal(ended,true);
 const completedJobs=jobs;for(let i=16;i<90;i++)assert.equal(gate.admit(i/30).type,'idle');
 assert.equal(jobs,completedJobs,'idle camera frames start no detector/fitting work');fitter.dispose();
});
