import assert from "node:assert/strict";
import test from "node:test";
import {makeCubePose,poseSurfaces,projectSurfaces,layerTransition,easeOutCubic,relativeTurns} from "./cube-motion";
import type {CubePose,CubeSurface,V3} from "./cube-motion";
import {buildColorTimeline,stateAtFrame} from "./cube-colors";
import type {CubeStateSnapshot} from "./cube-colors";
import oracle from "../../../pose-fitter/tests/fixtures/cube-projection.json";
import turns from "../../../pose-fitter/tables/render-turns.json";
const initial=stateAtFrame(buildColorTimeline("",[]),0,true)!;
const pose=(axis:number,angles:number[],state:CubeStateSnapshot=initial,rvec=[.2,-.3,.1])=>makeCubePose({axis,params:rvec,angles},state)!;
const squared=(p:number[])=>p.reduce((sum,v)=>sum+v*v,0);
const near=(a:number,b:number)=>assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);
function sameStickers(a:CubeSurface[],b:CubeSurface[],context="") {
 for(const face of a){const other=b.find(f=>f.sticker===face.sticker)!;assert.ok(other);assert.equal(face.color,other.color);
  for(const point of face.points)assert.ok(other.points.some(p=>squared(point.map((v,i)=>v-p[i]))<1e-14),`${context}: sticker ${face.sticker}`);
 }
}
function joins(from:CubePose,to:CubePose){
 const transition=layerTransition(from,to),phases=transition.phases;
 let last=poseSurfaces(from);
 for(const phase of phases){
  const beginning=poseSurfaces({...from,axis:phase.axis,angles:phase.start,state:phase.state});
  sameStickers(last,beginning,"phase must start at the previous physical configuration");
  last=poseSurfaces({...from,axis:phase.axis,angles:phase.end,state:phase.state});
 }
 sameStickers(last,poseSurfaces({...to,rotation:from.rotation}),"last phase must reach the target configuration");
 return transition;
}
test("shared-layer projection preserves the fixed-camera OpenCV reference",()=>{
 for(const sample of oracle.cases){
  const faces=projectSurfaces(poseSurfaces(makeCubePose(sample,initial)!)).sort((a,b)=>a.face-b.face);
  assert.equal(faces.length,sample.expected.length);
  faces.forEach((face,i)=>{assert.equal(face.face,sample.expected[i].face);near(face.depth,sample.expected[i].depth);
   face.points.forEach((p,j)=>p.forEach((v,k)=>near(v,sample.expected[i].points[j][k])));});
 }
});
test("one global rotation takes the short arc, with exact endpoints",()=>{
 const scene=(angle:number)=>pose(0,[0,0,0],initial,[0,0,angle*Math.PI/180]);
 const start=scene(179),end=scene(-179),transition=layerTransition(start,end);
 assert.equal(transition.sample(0),start);assert.equal(transition.sample(1),end);
 near(easeOutCubic(.5),.875);
 sameStickers(poseSurfaces(transition.sample(.5)),poseSurfaces(scene(180.75)));
});
test("whole layers, including their centers, share signed angles through interrupted ease curves",()=>{
 for(const axis of [0,1,2])for(const side of [0,1,2])for(const angle of [-180,-120,-90,-60,60,90,120,180]){
  const scene=(value:number)=>{const angles=[0,0,0];angles[side]=value;return pose(axis,angles);};
  const transition=joins(scene(0),scene(angle));assert.equal(transition.phases.length,1);
  for(const progress of [.2,.5,.8])sameStickers(poseSurfaces(transition.sample(progress)),poseSurfaces(scene(angle*easeOutCubic(progress))));
  const interrupted=transition.sample(.2),retarget=layerTransition(interrupted,scene(angle+15));
  sameStickers(poseSurfaces(retarget.sample(.5)),poseSurfaces(scene(angle*easeOutCubic(.2)+(angle+15-angle*easeOutCubic(.2))*easeOutCubic(.5))));
 }
});
test("all native face, wide, slice and cube-rotation commits rebase into the same rigid slabs",()=>{
 for(const [move,turn] of Object.entries(turns)){
  const after=stateAtFrame(buildColorTimeline("",[{frame_index:1,state_moves:[move]}]),1,true)!;
  const from=pose(turn.axis,turn.angles.map(v=>v*.8)),to=pose(turn.axis,[0,0,0],after);
  const transition=joins(from,to);
  assert.equal(transition.phases.length,1,move);
  for(const progress of [.2,.5,.8])sameStickers(poseSurfaces(transition.sample(progress)),
   poseSurfaces(pose(turn.axis,turn.angles.map(v=>v*(.8+.2*easeOutCubic(progress))))),move);
  joins(to,from);
 }
});
test("perpendicular axis changes align the old layers before moving the next layer",()=>{
 const after=stateAtFrame(buildColorTimeline("",[{frame_index:1,state_moves:["R'"]}]),1,true)!;
 const from=pose(0,[0,0,80]),to=pose(2,[-25,0,30],after),transition=joins(from,to);
 assert.equal(transition.phases.length,2);
 assert.deepEqual(transition.phases.map(p=>p.axis),[0,2]);
 assert.deepEqual(transition.phases[0].start,[0,0,-10]);
 assert.deepEqual(transition.phases[0].end,[0,0,0]);assert.deepEqual(transition.phases[1].start,[0,0,0]);
 for(const progress of [.1,.3,.8]){
  const cut=transition.sample(progress);
  joins(cut,pose(1,[40,0,-35],after));
 }
 joins(to,from);
});
test("multiple committed axes and reverse scrubbing meet at matching physical boundaries",()=>{
 const history=["R'","U","F2","M'","Bw"],timeline=buildColorTimeline("",[{frame_index:1,state_moves:history}]);
 const to=pose(1,[5,0,-15],stateAtFrame(timeline,1,true)!),from=pose(0,[0,0,70]);
 const transition=joins(from,to);assert.ok(transition.phases.length>=3);
 for(const p of [.05,.2,.6,.9]){const cut=transition.sample(p);joins(cut,to);joins(cut,from);}
 joins(to,from);
 assert.deepEqual(relativeTurns(history,[]).map(action=>action.history.at(-1)),["M'","F2","U","R'",undefined]);
});
