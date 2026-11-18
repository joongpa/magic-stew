import assert from "node:assert/strict";
import test from "node:test";
import type { FittedPose } from "./pose-types";
import { projectedWireframe } from "./reconstruction-view";
import { projectedColoredFaces } from "./reconstruction-view";
import type { CubeColor } from "./cube-colors";
import { fixedCubeProjection, CUBE_VIEW_BOX } from "./cube-projection";
import oracle from "../../../pose-fitter/tests/fixtures/cube-projection.json";
test("the projected cube uses the search geometry, including reserved flat slot six",()=>{
 const wireframes=Array.from({length:6},(_,i)=>[[[i,0],[i,1]]]) as [number,number][][][];
 const pose={bank:Array.from({length:6},(_,i)=>({flat_fallback:i===5})),wireframe:wireframes[0],wireframes} as FittedPose;
 assert.equal(projectedWireframe(pose,3),wireframes[3]);
 assert.equal(projectedWireframe(pose,5),wireframes[5]);
 assert.equal(projectedWireframe(pose,3,true),wireframes[5]);
 assert.equal(projectedWireframe(pose,null),wireframes[0]);
 assert.equal(projectedWireframe(pose,9),wireframes[0]);
});
test("colored geometry uses physical face IDs and paints distant faces before near ones",()=>{
 const pose={bank:[oracle.cases[0],oracle.cases[1]],wireframe:[]} as unknown as FittedPose;
 const colors=Array(54).fill("white") as CubeColor[];colors[22]="green";colors[3]="red";
 const faces=projectedColoredFaces(pose,colors,1);
 assert.deepEqual(faces.map(face=>face.face).sort((a,b)=>a-b),oracle.cases[1].expected.map(face=>face.face));
 for (const [i,face] of faces.entries()) {
  assert.equal(face.color,colors[face.face]);
  if(i)assert.ok(faces[i-1].depth>=face.depth);
 }
});
test("fixed camera matches OpenCV rotation and residual layer geometry on every axis",()=>{
 assert.equal(CUBE_VIEW_BOX,"-1 -1 2 2");
 for(const sample of oracle.cases){
  const actual=fixedCubeProjection(sample);
  assert.deepEqual(actual.map(face=>face.face),sample.expected.map(face=>face.face));
  for(const [i,face] of actual.entries()){
   assert.ok(Math.abs(face.depth-sample.expected[i].depth)<1e-12);
   face.points.forEach((point,j)=>point.forEach((value,k)=>{
    assert.ok(Math.abs(value-sample.expected[i].points[j][k])<1e-12);
    assert.ok(Math.abs(value)<1,"fixed viewport contains the turned cube");
   }));
  }
 }
});
test("source translation and video mesh cannot move or resize the side-panel cube",()=>{
 const fit={...oracle.cases[1],flat_fallback:true,score:0,nearest:[],valid:[]};
 const pose={bank:[oracle.cases[0],fit],wireframe:[[[500,600],[800,900]]]} as unknown as FittedPose;
 const colors=Array(54).fill("white") as CubeColor[];
 const expected=projectedColoredFaces(pose,colors,1);
 assert.ok(expected.length);
 const moved={...pose,bank:[pose.bank[0],{...fit,params:[...fit.params.slice(0,3),-999,888,5,0,0]}],wireframe:[]} as FittedPose;
 assert.deepEqual(projectedColoredFaces(moved,colors,1),expected);
 assert.deepEqual(projectedColoredFaces(moved,colors,0,true),expected,"reserved flat selection is preserved");
 assert.equal(projectedWireframe(pose),pose.wireframe,"video overlay still uses source-space mesh");
 const rotated={...fit,params:[0,0,0,...fit.params.slice(3)]};
 assert.notDeepEqual(fixedCubeProjection(rotated),fixedCubeProjection(fit));
 assert.notDeepEqual(fixedCubeProjection({...fit,angles:[0,0,0]}),fixedCubeProjection(fit));
});
