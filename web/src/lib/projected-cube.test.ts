import assert from "node:assert/strict";
import test from "node:test";
import { buildColorTimeline, stateAtFrame } from "./cube-colors";
import { makeCubePose, poseSurfaces } from "./cube-motion";
import type { CubePose } from "./cube-motion";
import type { FrameResult } from "./detector";
import type { MoveStep } from "./move-search-types";
import type { FittedPose } from "./pose-types";
import { buildProjectedCubeTimeline, projectedCubeAtFrame, hideUncommittedMisalignments } from "./projected-cube";
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
function partial(frame:number,axis:number,angles:number[]):SavedVideoResult {
 const result=row(frame,0);result.result.fittedPose!.bank[0].axis=axis;result.result.fittedPose!.bank[0].angles=angles;
 return result;
}
test("solved playback removes unsupported layer offsets independently and keeps real pre-commit motion",()=>{
 const rows=[partial(600,0,[0,0,0]),partial(602,0,[12,6,25]),partial(604,0,[0,0,0])];
 const steps=[step(600),step(602),step(604,['R'])],saved=JSON.stringify(rows);
 const cleaned=timeline(rows,steps);
 assert.deepEqual(cleaned.find(r=>r.frame===602)!.pose.angles,[0,0,25]);
 assert.equal(JSON.stringify(rows),saved,'raw fitted poses stay unchanged');
 const original=buildProjectedCubeTimeline(rows,buildColorTimeline('',steps),{fps:60,solved:false,steps});
 assert.deepEqual(original.find(r=>r.frame===602)!.pose.angles,[12,6,25],'a running candidate has no future commitment guarantee');
 const unrelated=timeline(rows,[step(600),step(602),step(604,['U'])]);
 assert.deepEqual(unrelated.find(r=>r.frame===602)!.pose.angles,[0,0,0]);
});
test("a later commit cannot justify an earlier offset that already returned to flat",()=>{
 const rows=[partial(600,0,[0,0,0]),partial(602,0,[0,0,18]),partial(604,0,[0,0,0]),partial(606,0,[0,0,32]),partial(608,0,[0,0,0]),partial(610,0,[0,0,14])];
 const steps=rows.map(r=>step(r.result.fittedPose!.sourceFrame!,r.result.fittedPose!.sourceFrame===608?['R']:[]));
 const cleaned=timeline(rows,steps);
 assert.deepEqual(cleaned.map(r=>r.pose.angles[2]),[0,0,0,32,0,0]);
});
test("a commit wrapping positive to negative residual offsets preserves the entire movement",()=>{
 const rows=[partial(600,0,[0,0,0]),partial(602,0,[0,0,35]),partial(604,0,[0,0,-40]),partial(606,0,[0,0,-10]),partial(608,0,[0,0,0])];
 const steps=rows.map(r=>step(r.result.fittedPose!.sourceFrame!,r.result.fittedPose!.sourceFrame===604?['R']:[]));
 assert.deepEqual(timeline(rows,steps).map(r=>r.pose.angles[2]),[0,35,-40,-10,0]);
});
test("a slice commit keeps its middle offset; a cube rotation does not count as a layer commit",()=>{
 const rows=[partial(600,0,[0,0,0]),partial(602,0,[7,19,11]),partial(604,0,[0,0,0])];
 const middle=timeline(rows,[step(600),step(602),step(604,['M'])]);
 assert.deepEqual(middle.find(r=>r.frame===602)!.pose.angles,[0,19,0]);
 const rotation=timeline(rows,[step(600),step(602),step(604,['x'])]);
 assert.deepEqual(rotation.find(r=>r.frame===602)!.pose.angles,[0,0,0]);
 const saved=JSON.stringify(middle),again=hideUncommittedMisalignments(middle,[step(604,['M'])]);
 assert.equal(JSON.stringify(middle),saved);
 for(let i=0;i<again.length;i++){assert.equal(again[i].pose.rotation,middle[i].pose.rotation);assert.equal(again[i].pose.state,middle[i].pose.state);}
});
test("a known commit in a rejected pose gap still supports its preceding partial turn",()=>{
 const rows=[partial(600,0,[0,0,0]),partial(602,0,[0,0,28]),row(604,null),partial(606,0,[0,0,0])];
 const steps=[step(600),step(602),step(604,['R'],true),step(606)];
 const result=timeline(rows,steps);
 assert.deepEqual(result.find(r=>r.frame===602)!.pose.angles,[0,0,28]);
 assert.deepEqual(result.find(r=>r.frame===604)!.pose.angles,[0,0,0]);
});
test("a slice encoded as outer commits retains its middle-layer approach",()=>{
 const rows=[partial(600,0,[0,0,0]),partial(602,0,[7,24,9]),partial(604,0,[0,0,0])];
 rows[2].result.fittedPose!.bank[0].params=[Math.PI/2,0,0];
 const steps=[step(600),step(602),step(604,["L'",'R'])];
 const result=timeline(rows,steps);
 assert.deepEqual(result.find(row=>row.frame===602)!.pose.angles,[0,24,0]);
 assert.deepEqual(result.at(-1)!.pose.state.moves,["L'",'R'],'display classification never rewrites committed state');
});
