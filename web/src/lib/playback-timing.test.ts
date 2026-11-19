import assert from "node:assert/strict";
import test from "node:test";
import { PLAYBACK_SPEEDS, playbackTiming } from "./playback-timing";

test("slow motion preserves selected frames and stretches video and cube motion together",()=>{
 for(const fps of [24,30,60]){
  const normal=playbackTiming(fps,1),selectedFrames=Math.min(fps,30);
  for(const speed of PLAYBACK_SPEEDS){
   const timing=playbackTiming(fps,speed);
   assert.ok(Math.abs(selectedFrames*timing.frameIntervalMs-1000/speed)<1e-9);
   assert.equal(timing.transitionMs,normal.transitionMs/speed);
   assert.ok(Math.abs(timing.frameIntervalMs/timing.transitionMs-normal.frameIntervalMs/normal.transitionMs)<1e-12);
  }
 }
});

test("missing source FPS and unsupported rates retain a valid normal playback clock",()=>{
 for(const fps of [0,-1,NaN,Infinity])assert.deepEqual(playbackTiming(fps,.5),playbackTiming(30,.5));
 for(const speed of [0,-1,NaN,Infinity,.75])assert.deepEqual(playbackTiming(30,speed),playbackTiming(30,1));
 assert.deepEqual(playbackTiming(30),playbackTiming(30,1));
});
