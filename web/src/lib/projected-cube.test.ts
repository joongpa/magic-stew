import assert from "node:assert/strict";
import test from "node:test";
import { buildColorTimeline, stateAtFrame } from "./cube-colors";
import { makeCubePose, poseSurfaces } from "./cube-motion";
import type { CubePose } from "./cube-motion";
import type { FrameResult } from "./detector";
import type { MoveStep } from "./move-search-types";
import type { FittedPose } from "./pose-types";
import { buildProjectedCubeTimeline, projectedCubeAtFrame } from "./projected-cube";
import type { SavedVideoResult } from "./video-result-cache";
import turns from "../../../pose-fitter/tables/render-turns.json";

function row(frame:number,degrees:number|null,flagged=false):SavedVideoResult {
 const fit={axis:0,params:[0,0,(degrees??0)*Math.PI/180],angles:[0,0,0],
  flat_fallback:false,score:0,nearest:[],valid:[]};
 const fittedPose:FittedPose={sourceFrame:frame,bank:[fit],wireframe:[],bootstrap:false,message:"Test",elapsedMs:0,
  pose:degrees==null?null:{rvec:fit.params,tvec:[0,0,500]},pose_flag_reason:flagged?"rejected":null};
 return {timestamp:frame/60,result:{fittedPose} as FrameResult};
}
function step(frame_index:number,state_moves:string[]=[],pose_flagged=false):MoveStep {
 return {frame_index,state_moves,pose_flagged,geometry_option_index:0,state_score:0,color_error:0,queue_priority:0};
}
function timeline(rows:SavedVideoResult[],steps:MoveStep[]) {
 const colors=buildColorTimeline("",steps);
 return buildProjectedCubeTimeline(rows,colors,{fps:60,solved:true,steps});
}
function samePhysicalPose(actual:CubePose,expected:CubePose) {
 const surfaces=poseSurfaces(expected);
 for(const face of poseSurfaces(actual)){
  const other=surfaces.find(f=>f.sticker===face.sticker)!;
  assert.equal(face.color,other.color);
  for(const point of face.points)assert.ok(other.points.some(p=>
   point.reduce((sum,v,i)=>sum+(v-p[i])**2,0)<1e-14),`sticker ${face.sticker}`);
 }
}

test("missing, rejected and search-flagged frames bridge accepted rotations in source time",()=>{
 const rows=[row(600,0),row(602,null),row(604,170,true),row(606,-130),row(608,90)];
 const steps=rows.map(r=>step(r.result.fittedPose!.sourceFrame!,[],r.timestamp===606/60));
 const saved=JSON.stringify(rows),frames=timeline(rows,steps),initial=stateAtFrame(buildColorTimeline("",[]),0,true)!;
 assert.deepEqual(frames.map(r=>r.frame),[600,608]);
 for(const frame of [606,602,604,600,608,604]){
  const expected=makeCubePose({axis:0,params:[0,0,(frame-600)/8*Math.PI/2],angles:[0,0,0]},initial)!;
  samePhysicalPose(projectedCubeAtFrame(frames,frame)!,expected);
 }
 assert.equal(JSON.stringify(rows),saved,"interpolation must leave diagnostic fits intact");
});

test("a move across a pose gap animates rigid layers along with global rotation",()=>{
 const steps=[step(600),step(602),step(604,["R"])],frames=timeline([row(600,0),row(602,null),row(604,90)],steps);
 const initial=stateAtFrame(buildColorTimeline("",steps),600,true)!;
 const expected=makeCubePose({axis:turns.R.axis,params:[0,0,Math.PI/4],angles:turns.R.angles.map(v=>v/2)},initial)!;
 samePhysicalPose(projectedCubeAtFrame(frames,602)!,expected);
 assert.equal(projectedCubeAtFrame(frames,600),frames[0].pose);
 assert.equal(projectedCubeAtFrame(frames,604),frames.at(-1)!.pose);
});

test("a committed move on a rejected frame still reaches its own source-frame target",()=>{
 const steps=[step(600),step(602,["R"],true),step(604)],colors=buildColorTimeline("",steps);
 const frames=timeline([row(600,0),row(602,170,true),row(604,90)],steps);
 assert.deepEqual(frames.map(r=>r.frame),[600,602,604]);
 const commit=projectedCubeAtFrame(frames,602)!;
 assert.deepEqual(commit.state.moves,["R"]);
 assert.deepEqual(commit.angles,[0,0,0]);
 samePhysicalPose(commit,makeCubePose({axis:0,params:[0,0,Math.PI/4],angles:[0,0,0]},stateAtFrame(colors,602,true)!)!);
 const half=makeCubePose({axis:turns.R.axis,params:[0,0,Math.PI/8],angles:turns.R.angles.map(v=>v/2)},
  stateAtFrame(colors,600,true)!)!;
 samePhysicalPose(projectedCubeAtFrame(frames,601)!,half);
});

test("leading and trailing pose gaps retain orientation and continue reconstructed moves",()=>{
 const steps=[step(600),step(602,["R"]),step(604),step(606,["U"]),step(608,["F"])];
 const frames=timeline([row(600,null),row(602,null),row(604,45),row(606,null),row(608,null)],steps);
 assert.deepEqual(projectedCubeAtFrame(frames,600)!.state.moves,[],"do not show a future sticker state at the start");
 assert.deepEqual(projectedCubeAtFrame(frames,602)!.state.moves,["R"]);
 assert.deepEqual(projectedCubeAtFrame(frames,608)!.state.moves,["R","U","F"]);
 const colors=buildColorTimeline("",steps),afterR=stateAtFrame(colors,604,true)!;
 const halfU=makeCubePose({axis:turns.U.axis,params:[0,0,Math.PI/4],angles:turns.U.angles.map(v=>v/2)},afterR)!;
 samePhysicalPose(projectedCubeAtFrame(frames,605)!,halfU);
 assert.equal(projectedCubeAtFrame(frames,610),frames.at(-1)!.pose);
});

test("accepted anchors retain geometry selection, static holds, symmetry and the absolute source clock",()=>{
 const saved=row(1200,10),pose=saved.result.fittedPose!;
 delete pose.sourceFrame;
 pose.bank.push({...pose.bank[0],params:[.3,-.2,.1],flat_fallback:true});
 const steps=[{...step(1200),geometry_option_index:1,orientation_symmetry_index:3}],colors=buildColorTimeline("",steps);
 const options={fps:60,solved:true,steps};
 const frames=buildProjectedCubeTimeline([saved],colors,options);
 assert.equal(frames[0].frame,1200);
 assert.deepEqual(frames[0].pose,makeCubePose(pose.bank[1],stateAtFrame(colors,1200,true)!,3));
 steps[0].geometry_option_index=0;
 const hold=buildProjectedCubeTimeline([saved],colors,{...options,turningStartFrame:1200});
 assert.deepEqual(hold[0].pose,frames[0].pose);
});

test("without any usable anchor the cube stays unavailable instead of using rejected geometry",()=>{
 const invalid=row(604,0);invalid.result.fittedPose!.bank[0].params[0]=NaN;
 const frames=timeline([row(600,null),row(602,0,true),invalid],[step(600),step(602),step(604)]);
 assert.deepEqual(frames,[]);
 assert.equal(projectedCubeAtFrame(frames,602),undefined);
});
