import assert from "node:assert/strict";
import test from "node:test";
import type { FrameResult, Settings } from "./detector";
import { VideoResultCache, videoResultKey } from "./video-result-cache";

const settings: Settings = { thresholds: [0.25, 0.4, 0.25, 0.9], presenceThreshold: 0.05, bboxThreshold: 0.25, roiPadding: 0.4 };
test("scrubbing uses the nearest completed selected frame without crossing an unprocessed gap", () => {
  const cache = new VideoResultCache(), result = { fittedPose: { stage: "cpu_pipeline" } } as FrameResult;
  cache.put("a", 6.567, result); cache.put("a", 6.534, result);
  assert.equal(cache.get("a", 6.54, 1 / 60)?.timestamp, 6.534);
  assert.equal(cache.get("a", 6.56, 1 / 60)?.timestamp, 6.567);
  assert.equal(cache.get("a", 7, 1 / 60), undefined);
  assert.equal(cache.get("a", 6.54), undefined, "playback requires an exact decoded timestamp");
  assert.equal(cache.get("a", 6.534)?.result, result, "full fitted result is retained by reference");
});
test("fitted caches distinguish scramble center frames and opening holds",()=>{
 const key=(scramble:string,turningStartFrame:number|null)=>videoResultKey(256,"webgpu",settings,true,{scramble,turningStartFrame});
 assert.notEqual(key("R",null),key("Rw",null));
 assert.notEqual(key("R",null),key("R",49));
 assert.notEqual(videoResultKey(256,"webgpu",settings,true,{scramble:"R",turningStartFrame:null,originSeconds:0}),
  videoResultKey(256,"webgpu",settings,true,{scramble:"R",turningStartFrame:null,originSeconds:6}),
  "a different trimmed prefix cannot reuse poses seeded by an earlier prefix");
 assert.notEqual(videoResultKey(256,"webgpu",settings,true,{scramble:"R",turningStartFrame:null,originSeconds:0,endSeconds:4}),
  videoResultKey(256,"webgpu",settings,true,{scramble:"R",turningStartFrame:null,originSeconds:0,endSeconds:3}),
  "a different ending changes the fitter's available future recovery window");
 assert.equal(videoResultKey(256,"webgpu",settings,false,{scramble:"R",turningStartFrame:null}),
  videoResultKey(256,"webgpu",settings,false,{scramble:"Rw",turningStartFrame:49}));
});
test("replacement does not duplicate results and configurations cannot reuse stale fits", () => {
  const cache = new VideoResultCache(), key = videoResultKey(256, "webgpu", settings, true);
  const first = {} as FrameResult, replacement = { trackingLost: true } as FrameResult;
  cache.put(key, 1, first); cache.put(key, 1, replacement);
  assert.equal(cache.count(key), 1); assert.equal(cache.get(key, 1)?.result, replacement);
  for (const changed of [videoResultKey(160, "webgpu", settings, true), videoResultKey(256, "wasm", settings, true),
    videoResultKey(256, "webgpu", { ...settings, roiPadding: 0.5 }, true), videoResultKey(256, "webgpu", settings, false)]) {
    assert.equal(cache.get(changed, 1), undefined);
  }
  cache.clear(); assert.equal(cache.count(key), 0);
});
