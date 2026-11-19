import assert from "node:assert/strict";
import test from "node:test";
import orientations from "../../../pose-fitter/tables/orientations.json";
import turns from "../../../pose-fitter/tables/render-turns.json";
import { applyCubeMove, buildColorTimeline, stateAtFrame } from "./cube-colors";
import type { CubeStateSnapshot } from "./cube-colors";
import { axisAngle, multiply, poseSurfaces, rotate } from "./cube-motion";
import type { CubePose, CubeSurface, Quaternion, V3 } from "./cube-motion";
import type { MoveStep } from "./move-search-types";
import { CAMERA_ORIENTATION, displayCubeBasis, reorientedCubePose, reorientedMove, reorientedMoveTimeline, rotateCubeView } from "./cube-orientation";

const initial=stateAtFrame(buildColorTimeline("",[]),0,true)!;
// Canonical cube faces: R = +X, U = +Z, F = -Y. Camera: right +X,
// screen-up -Y, toward the viewer -Z, so this pose aligns those frames.
const camera=axisAngle([Math.PI/2,0,0]);
const inverse=(q:Quaternion):Quaternion=>[-q[0],-q[1],-q[2],q[3]];
const pose=(rotation:Quaternion=camera,axis=0,angles:V3=[0,0,0]):CubePose=>({rotation,axis,angles,state:initial});
const quaternion=(index:number)=>axisAngle(orientations.rvecs[index] as V3);
const step=(frame_index:number,state_moves:string[]):MoveStep=>({frame_index,state_moves,geometry_option_index:0,state_score:0,color_error:0,queue_priority:0});
function close(actual:number[],expected:number[]) {
 actual.forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-9,`${actual} != ${expected}`));
}
function sameStickers(actual:CubeSurface[],expected:CubeSurface[]) {
 const bySticker=new Map(expected.map(face=>[face.sticker,face]));
 for(const face of actual){
  const target=bySticker.get(face.sticker)!;assert.equal(face.color,target.color);
  for(const point of face.points)assert.ok(target.points.some(p=>p.every((v,i)=>Math.abs(v-point[i])<1e-9)),`sticker ${face.sticker}`);
 }
}
function relabel(state:CubeStateSnapshot,basis:number):CubeStateSnapshot {
 const colors=[...state.colors],stickers=[...state.stickers];
 orientations.face_permutations[basis].forEach((target,source)=>{colors[target]=state.colors[source];stickers[target]=state.stickers[source];});
 return {colors,stickers,moves:[]};
}

test("U is always the face closest to display-up, then F is the closest perpendicular face to the viewer",()=>{
 const normals:V3[]=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
 const rotations=[...orientations.rvecs.map((_,i)=>multiply(camera,quaternion(i))),
  ...Array.from({length:80},(_,i)=>axisAngle([i*.37-1,i*.19+.2,i*.43-.3]))];
 for(const rotation of rotations){
  const basis=displayCubeBasis(rotation),matrix=orientations.matrices[basis];
  const up=matrix[2] as V3,front=matrix[1].map(v=>-v) as V3;
  const upScore=-rotate(rotation,up)[1],frontScore=-rotate(rotation,front)[2];
  for(const normal of normals){
   assert.ok(upScore>=-rotate(rotation,normal)[1]-1e-9);
   if(normal.reduce((sum,v,i)=>sum+v*up[i],0)===0)assert.ok(frontScore>=-rotate(rotation,normal)[2]-1e-9);
  }
 }
 for(let i=0;i<24;i++)assert.equal(displayCubeBasis(multiply(camera,quaternion(i))),i);
});

test("x/y/z controls use the currently displayed R/U/F faces even on tilted cubes",()=>{
 const upright=pose(),tilted=pose(axisAngle([.7,-.4,1.3]));
 for(const source of [upright,tilted])for(const view of [CAMERA_ORIENTATION,axisAngle([-.3,.2,.6])]){
  const displayed=multiply(view,source.rotation),matrix=orientations.matrices[displayCubeBasis(displayed)];
  for(const [move,axis] of [["x",matrix[0]],["y",matrix[2]],["z",matrix[1].map(v=>-v)]] as [string,V3][]){
   const rotated=rotateCubeView(view,source.rotation,move),after=multiply(rotated,source.rotation);
   close(rotate(after,axis),rotate(displayed,axis));
   assert.notDeepEqual(rotated,view);
   assert.ok(Math.abs(Math.hypot(...rotated)-1)<1e-12);
  }
 }
 // A further control follows the newly uppermost face, rather than the
 // original material U face. In the aligned display x then y maps right to front.
 let view=rotateCubeView(CAMERA_ORIENTATION,camera,"x");
 view=rotateCubeView(view,camera,"y");
 close(rotate(multiply(view,camera),[1,0,0]),rotate(camera,[0,-1,0]));
 assert.throws(()=>rotateCubeView(CAMERA_ORIENTATION,camera,"R"),/x, y or z/);
});

test("control direction follows standard cube notation for all 24 displayed orientations",()=>{
 for(let i=0;i<24;i++)for(const move of ["x","x'","y","y'","z","z'"]){
  const rotation=multiply(camera,quaternion(i)),view=rotateCubeView(CAMERA_ORIENTATION,rotation,move);
  const turn=turns[move as keyof typeof turns],vector:V3=[0,0,0];vector[turn.axis]=turn.angles[0]*Math.PI/180;
  const expected=multiply(camera,multiply(axisAngle(vector),quaternion(i)));
  for(const normal of [[1,0,0],[0,1,0],[0,0,1]] as V3[])close(rotate(multiply(view,rotation),normal),rotate(expected,normal));
 }
});

test("reorienting carries the entire tracked animation through one view transform",()=>{
 const view=rotateCubeView(CAMERA_ORIENTATION,axisAngle([.5,-.7,.3]),"y");
 for(const axis of [0,1,2])for(const rotation of [camera,axisAngle([-.2,.4,.8]),axisAngle([1,-.8,.1])]){
  const source=pose(rotation,axis,[23,-41,67]),saved=JSON.stringify(source),result=reorientedCubePose(source,view);
  const expected=poseSurfaces(source).map(face=>({...face,points:face.points.map(point=>rotate(view,point))}));
  sameStickers(poseSurfaces(result),expected);
  assert.equal(result.state,source.state,"view changes must not rewrite the animation's sticker history");
  assert.equal(JSON.stringify(source),saved);
  assert.equal(reorientedCubePose(source,CAMERA_ORIENTATION),source,"reset restores the exact camera pose, even mid-turn");
 }
});

test("every face, wide, slice and explicit cube rotation is converted in the displayed basis",()=>{
 for(let basis=0;basis<24;basis++)for(const move of Object.keys(turns)){
  const before=relabel(initial,basis),mapped=reorientedMove(move,basis);
  const actual=applyCubeMove(before,mapped,[]),expected=relabel(applyCubeMove(initial,move,[]),basis);
  assert.deepEqual(actual,expected,`${basis}: ${move} -> ${mapped}`);
  if(/^[xyz]/.test(move))assert.match(mapped,/^[xyz]/);
  if(/^[rludfb]/.test(move))assert.match(mapped,/^[rludfb]/);
 }
 assert.equal(reorientedMove("R2'",0),"R2");assert.equal(reorientedMove("R’",0),"R'");
 assert.throws(()=>reorientedMove("unknown",0),/Unknown cube turn/);
});

test("changing the view translates tracked cube rotations as well as face turns",()=>{
 const y=axisAngle([0,0,-Math.PI/2]),rows=[{frame:600,pose:pose()},{frame:602,pose:pose(multiply(camera,y))}];
 const steps=[step(600,["U"]),step(602,["R","x'"])];
 const original=reorientedMoveTimeline(rows,steps,CAMERA_ORIENTATION);
 assert.deepEqual(original[1],{frame:602,moves:["y"],rotation:true});
 const view=rotateCubeView(CAMERA_ORIENTATION,camera,"x"),changed=reorientedMoveTimeline(rows,steps,view);
 assert.deepEqual(changed[1],{frame:602,moves:["z'"],rotation:true},"tracked y must become z' after an x view change");
 for(const row of changed){
  if(row.rotation)continue;
  const source=rows.find(source=>source.frame===row.frame)!;
  const basis=displayCubeBasis(multiply(view,source.pose.rotation));
  assert.deepEqual(row.moves,steps.find(step=>step.frame_index===row.frame)!.state_moves.map(move=>reorientedMove(move,basis)));
 }
 assert.notDeepEqual(original[0].moves,changed[0].moves);
 assert.deepEqual(reorientedMoveTimeline(rows,steps,CAMERA_ORIENTATION),original,"reset must restore notation without accumulating rotations");
});

test("tracked rotations and converted moves replay to the same physical stickers in every view",()=>{
 const sourceMoves=["R","x'"],steps=[step(1,sourceMoves)];
 let finalState=initial;for(const move of sourceMoves)finalState=applyCubeMove(finalState,move,[]);
 for(let grip=0;grip<24;grip++)for(let setup=0;setup<24;setup++){
  const view=multiply(multiply(camera,quaternion(setup)),inverse(camera));
  const target={...pose(multiply(camera,quaternion(grip))),state:finalState};
  const rows=[{frame:0,pose:pose()},{frame:1,pose:target}];
  const events=reorientedMoveTimeline(rows,steps,view);
  let state=relabel(initial,setup);
  for(const move of events.flatMap(event=>event.moves))state=applyCubeMove(state,move,[]);
  sameStickers(poseSurfaces({...pose(),state}),poseSurfaces(reorientedCubePose(target,view)));
 }
});

test("the opening camera angle and missing frames use the same displayed faces as controls",()=>{
 const rotation=axisAngle([.3,-.5,.2]),source=pose(rotation),basis=displayCubeBasis(rotation);
 assert.notEqual(basis,0,"the original model U is not display-up in this camera pose");
 const rows=[{frame:600,pose:source},{frame:604,pose:source}],steps=[step(600,["U"]),step(602,["R"])];
 const actual=reorientedMoveTimeline(rows,steps,CAMERA_ORIENTATION);
 assert.deepEqual(actual,[
  {frame:600,moves:[reorientedMove("U",basis)],rotation:false},
  {frame:602,moves:[reorientedMove("R",basis)],rotation:false},
 ]);
 assert.deepEqual(reorientedMoveTimeline(rows,steps,CAMERA_ORIENTATION),actual);
 assert.deepEqual(reorientedMoveTimeline([],steps,CAMERA_ORIENTATION),[]);
});

test("exact face-assignment ties are deterministic under quaternion sign and numerical noise",()=>{
 const rotation=multiply(camera,axisAngle([0,0,Math.PI/4]));
 const basis=displayCubeBasis(rotation);
 assert.equal(displayCubeBasis(rotation.map(v=>-v) as Quaternion),basis);
 assert.equal(displayCubeBasis(multiply(rotation,axisAngle([0,0,1e-13]))),basis);
});
