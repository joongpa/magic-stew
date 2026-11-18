import fs from 'node:fs';
import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';
import {makeCubePose,poseSurfaces,layerTransition} from '../src/lib/cube-motion.ts';
import {buildColorTimeline,stateAtFrame} from '../src/lib/cube-colors.ts';
import {projectedGeometryIndex} from '../src/lib/reconstruction-view.ts';
if(!process.argv[2])throw new Error('Usage: tsx tools/check-cube-motion.mts recording.json [report.json]');
const recording=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const search=recording.moveSearches.at(-1),timeline=buildColorTimeline(search.config.scramble,search.result.steps);
const choices=new Map(search.result.steps.map(s=>[s.frame_index,s.geometry_option_index]));
const poses=recording.frames.map(row=>{
 const fitted=row.result.fittedPose;if(!fitted?.pose)return;
 const frame=fitted.sourceFrame??Math.round(row.timestampSeconds*search.config.fps);
 const fit=fitted.bank[projectedGeometryIndex(fitted,choices.get(frame),search.config.turning_start_frame!=null&&frame<=search.config.turning_start_frame)];
 const state=stateAtFrame(timeline,frame,true);return state&&makeCubePose(fit,state);
}).filter(Boolean);
let joins=0,samples=0,maxBoundaryError=0,transitions=0;const start=performance.now();
function boundary(a,b){
 const targets=new Map(b.map(face=>[face.sticker,face]));
 for(const face of a){const target=targets.get(face.sticker);assert.ok(target);assert.equal(target.color,face.color);
  for(const point of face.points){const error=Math.min(...target.points.map(p=>Math.hypot(...point.map((v,i)=>v-p[i]))));maxBoundaryError=Math.max(maxBoundaryError,error);assert.ok(error<1e-6,`discontinuous sticker ${face.sticker}: ${error}`);}}
 joins++;
}
function check(from,to){
 const transition=layerTransition(from,to);transitions++;let last=poseSurfaces(from);
 for(const phase of transition.phases){
  boundary(last,poseSurfaces({...from,axis:phase.axis,angles:phase.start,state:phase.state}));
  last=poseSurfaces({...from,axis:phase.axis,angles:phase.end,state:phase.state});
 }
 boundary(last,poseSurfaces({...to,rotation:from.rotation}));
 for(const t of [.1,.3,.7,.99]){const sample=transition.sample(t);assert.equal(sample.angles.length,3);assert.ok(sample.angles.every(Number.isFinite));samples++;}
 return transition;
}
for(const rows of [poses,[...poses].reverse()]){
 let displayed=rows[0];
 for(const target of rows.slice(1)){
  const transition=check(displayed,target);
  // Four screen refreshes per input frame; the unfinished 100ms curve is cut.
  for(let i=1;i<=4;i++){displayed=transition.sample(i/(120*.1));samples++;}
 }
}
const report={input:process.argv[2],simulatedRefreshRate:120,inputRate:30,easeDurationMs:100,savedFrames:poses.length,transitions,boundaryChecks:joins,sampledStates:samples,maxBoundaryErrorCubeUnits:maxBoundaryError,elapsedMs:performance.now()-start};
const output=JSON.stringify(report,null,2)+'\n';
if(process.argv[3])fs.writeFileSync(process.argv[3],output);
console.log(output);
