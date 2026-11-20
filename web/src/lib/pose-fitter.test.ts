import assert from "node:assert/strict";
import test from "node:test";
import { PoseFitter, pipelineFrame, poseObservations } from "./pose-fitter";
import type { FrameResult } from "./detector";
import cameraTiming from './fixtures/camera-timer-timestamps.json';
import cameraCollision from './fixtures/camera-clock-collision.json';
import {CameraFrameBuffer} from './camera-metrics';

test("pose fitting receives source coordinates and already-filtered object classes", () => {
  const frame = {
    transform: { sourceWidth: 1080, sourceHeight: 1920 },
    detections: [
      { classId: 0, x: 430, y: 910, localX: 2, localY: 3, score: 0.5, presence: 0.9 },
      { classId: 2, x: 450, y: 920, localX: 4, localY: 5, score: 0.7, presence: 0.95 },
      { classId: 7, x: 440, y: 915, localX: 3, localY: 4, score: 0.92, presence: 0.99 },
    ],
  } as unknown as FrameResult;
  assert.deepEqual(poseObservations(frame), {
    width: 1080, height: 1920,
    observations: { points: [[430, 910], [450, 920]], classes: [0, 2], stickers: [[440, 915]], uncertainties: [1] },
  });
  assert.equal(frame.detections[0].score, 0.5);
  assert.equal(frame.detections[0].presence, 0.9);
});

test('the failed camera recording preserves all 187 irregular frames in one fitting segment', async () => {
  const admitted: number[] = [];
  const pool = { dispose() {}, async execute(job: Record<string, any>) {
    if (job.type === 'prepare') { admitted.push(job.frame.frame); return { candidates: [] }; }
    if (job.type === 'select_startup') return { source: null, anchor: false };
    if (job.type === 'orient') return { bank: [], reference: null };
    return { bank: [], wireframe: [], pose: null };
  } };
  const fitter = new PoseFitter('http://localhost/', pool);
  const frame = { transform: { sourceWidth: 640, sourceHeight: 480, cropWidth: 200, cropHeight: 200 }, detections: [] } as unknown as FrameResult;
  const nominal = cameraTiming.timestamps.map(time => Math.round(time * cameraTiming.sourceFps));
  assert.equal(nominal.filter((value, i) => i > 0 && value <= nominal[i - 1]).length, 2, 'reproduces both original collisions');
  const pending = cameraTiming.timestamps.map((time, index) => fitter.fit(frame, index === 0, time, cameraTiming.sourceFps,
    { scramble: '', turningStartFrame: null, camera: true }));
  await fitter.flush();
  const poses = await Promise.all(pending);
  assert.equal(poses.length, 187); assert.equal(admitted.length, 187);
  assert.ok(admitted.every((value, i) => !i || value > admitted[i - 1]));
  assert.deepEqual([...new Set(poses.map(pose => pose.fittingConfig?.segment))], [1]);
  for (const [index, pose] of poses.entries()) {
    assert.equal(pose.fittingConfig?.fps, 1000); assert.equal(pose.fittingConfig?.sourceFps, 30);
    assert.ok(Math.abs(pose.sourceFrame! / pose.fittingConfig!.fps - cameraTiming.timestamps[index]) <= .0005);
  }
  fitter.dispose();
});

test("CPU pipeline gets ROI pixel scale, source frame number, colors and presence scores", () => {
  const frame = { transform: { sourceWidth: 1920, sourceHeight: 1080, cropWidth: 240, cropHeight: 200 },
    detections: [{ classId: 1, x: 20, y: 30, score: 0.45, presence: 0.98 },
      { classId: 5, x: 40, y: 50, score: 0.91, presence: 0.99 }] } as FrameResult;
  const input = pipelineFrame(frame, 392);
  assert.equal(input.frame, 392);
  assert.equal(input.pixel_scale, 0.6);
  assert.deepEqual(input.landmark_presence, [0.98]);
  assert.deepEqual(input.colors, ["orange"]);
  assert.deepEqual(input.observations.stickers, [[40, 50]]);
});

test('the 203-frame failed camera solve fits through the end after source admission filters the clock collision',async()=>{
  const admitted:number[]=[],buffer=new CameraFrameBuffer();
  const pool={dispose(){},async execute(job:Record<string,any>){
    if(job.type==='prepare'){admitted.push(job.frame.frame);return {candidates:[]};}
    if(job.type==='select_startup')return {source:null,anchor:false};
    if(job.type==='orient')return {bank:[],reference:null};
    return {bank:[],wireframe:[],pose:null};
  }};
  const fitter=new PoseFitter('http://localhost/',pool);
  const frame={transform:{sourceWidth:640,sourceHeight:480,cropWidth:200,cropHeight:200},detections:[]} as unknown as FrameResult;
  const pending:ReturnType<PoseFitter['fit']>[]=[],times:number[]=[];
  for(const timestamp of cameraCollision.timestamps){
    if(!buffer.admit(timestamp))continue;
    times.push(timestamp);pending.push(fitter.fit(frame,pending.length===0,timestamp,cameraCollision.sourceFps,
      {scramble:'',turningStartFrame:null,camera:true}));
    buffer.acknowledge();
  }
  await fitter.flush();const poses=await Promise.all(pending);
  assert.equal(cameraCollision.timestamps.length,203);assert.equal(buffer.skipped,1);
  assert.equal(poses.length,202);assert.equal(admitted.length,202);
  assert.ok(admitted.every((value,index)=>!index||value>admitted[index-1]));
  assert.ok(!times.includes(640.007205));
  assert.deepEqual(poses.map(pose=>pose.sourceFrame),times.map(timestamp=>Math.round(timestamp*1000)));
  assert.equal(poses.at(-1)?.sourceFrame,Math.round(cameraCollision.timestamps.at(-1)!*1000));
  assert.deepEqual([...new Set(poses.map(pose=>pose.fittingConfig?.segment))],[1]);
  fitter.dispose();
});
