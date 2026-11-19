import assert from 'node:assert/strict';
import test from 'node:test';
import {CameraFrameRate,CameraFrameBuffer} from './camera-metrics';
test('measured capture FPS follows source timestamps independently of processing delays or nominal FPS',()=>{
 const rate=new CameraFrameRate();let measured=0;
 for(let i=0;i<240;i++)measured=rate.observe(i/24);
 assert.ok(Math.abs(measured-24)<1e-8);
 for(let i=0;i<90;i++)measured=rate.observe(10+i/30);
 assert.ok(Math.abs(measured-30)<1e-8);rate.reset();assert.equal(rate.observe(0),0);
});
test('a full camera buffer keeps draining at consumer speed and resumes admission without stopping or evicting frames',()=>{
 const buffer=new CameraFrameBuffer(60),queued:number[]=[],processed:number[]=[];let admitted=0;
 // A 30fps producer and 10fps consumer must eventually saturate the buffer.
 for(let frame=0;frame<300;frame++){
  if(buffer.admit()){queued.push(frame);admitted++;}
  if(frame%3===0){processed.push(queued.shift()!);buffer.acknowledge();}
  assert.ok(buffer.pending<=60);
 }
 assert.ok(buffer.skipped>0);assert.ok(processed.at(-1)!>0,'processing continues during saturation');
 while(queued.length){processed.push(queued.shift()!);buffer.acknowledge();}
 assert.equal(processed.length,admitted);assert.ok(processed.every((frame,i)=>!i||frame>processed[i-1]));
 assert.equal(buffer.pending,0);assert.equal(buffer.admit(),true,'capture resumes admitting frames automatically');
});
