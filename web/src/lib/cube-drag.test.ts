import assert from "node:assert/strict";
import test from "node:test";
import { axisAngle, multiply, rotate } from "./cube-motion";
import type { Quaternion, V3 } from "./cube-motion";
import { CAMERA_ORIENTATION, displayCubeBasis, reorientedMove } from "./cube-orientation";
import { dragCubeView, nudgeCubeView } from "./cube-drag";

function close(a:number[],b:number[]) {a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-9,`${a} != ${b}`));}
function sameRotation(a:Quaternion,b:Quaternion) {
 for(const axis of [[1,0,0],[0,1,0],[0,0,1]] as V3[])close(rotate(a,axis),rotate(b,axis));
}

test("center drags turn and tilt the cube in the pointer's direction",()=>{
 const right=dragCubeView(CAMERA_ORIENTATION,[0,0],[.6,0],"orbit");
 const down=dragCubeView(CAMERA_ORIENTATION,[0,0],[0,.6],"orbit");
 const facing:V3=[0,0,-1];
 assert.ok(rotate(right,facing)[0]>0);close([rotate(right,facing)[1]],[0]);
 assert.ok(rotate(down,facing)[1]>0);close([rotate(down,facing)[0]],[0]);
 assert.equal(dragCubeView(right,[.2,.3],[.2,.3],"orbit"),right);
});

test("the roll ring rotates through the third axis without changing the viewing direction",()=>{
 const clockwise=dragCubeView(CAMERA_ORIENTATION,[1,0],[0,1],"roll");
 close(rotate(clockwise,[1,0,0]),[0,1,0]);
 close(rotate(clockwise,[0,0,-1]),[0,0,-1]);
 const backward=dragCubeView(clockwise,[0,1],[1,0],"roll");
 sameRotation(backward,CAMERA_ORIENTATION);
 const crossing=dragCubeView(CAMERA_ORIENTATION,[-1,.01],[-1,-.01],"roll");
 assert.ok(Math.abs(crossing[2])<.02,"crossing the atan2 seam must not cause a full-turn jump");
 assert.equal(dragCubeView(clockwise,[1,0],[0,0],"roll"),clockwise);
});

test("gestures start from the saved view and remain independent of intermediate pointer events",()=>{
 const view=axisAngle([.3,-.5,.2]),from:[number,number]=[-.2,.3],to:[number,number]=[.6,-.5];
 const sampled=dragCubeView(view,from,to,"orbit");
 for(const point of [[.1,.1],[.2,-.1],[.4,-.3]] as [number,number][])
  dragCubeView(view,from,point,"orbit");
 sameRotation(dragCubeView(view,from,to,"orbit"),sampled);
 sameRotation(dragCubeView(sampled,to,from,"orbit"),view);
});

test("long and repeated mixed-axis drags stay finite, normalized and rigid",()=>{
 let view:Quaternion=CAMERA_ORIENTATION;
 for(let i=0;i<500;i++){
  const from:[number,number]=[Math.sin(i),Math.cos(i*.7)],to:[number,number]=[Math.cos(i*.3)*4,Math.sin(i*.8)*3];
  view=dragCubeView(view,from,to,i%3?"orbit":"roll");
  assert.ok(view.every(Number.isFinite));assert.ok(Math.abs(Math.hypot(...view)-1)<1e-12);
 }
 const x=rotate(view,[1,0,0]),y=rotate(view,[0,1,0]);
 close([Math.hypot(...x),Math.hypot(...y),x.reduce((sum,v,i)=>sum+v*y[i],0)],[1,1,0]);
 const long=dragCubeView(view,[-1e6,0],[1e6,0],"orbit");
 assert.ok(long.every(Number.isFinite));close([Math.hypot(...long)],[1]);
});

test("keyboard provides turn, tilt and roll with reversible fine adjustments",()=>{
 for(const [forward,backward] of [["ArrowRight","ArrowLeft"],["ArrowDown","ArrowUp"],["e","q"]]){
  const start=axisAngle([.4,.1,-.2]),next=nudgeCubeView(start,forward)!;
  assert.notDeepEqual(next,start);sameRotation(nudgeCubeView(next,backward)!,start);
  let fine=start;for(let i=0;i<5;i++)fine=nudgeCubeView(fine,forward,true)!;
  sameRotation(fine,next);
 }
 assert.equal(nudgeCubeView(CAMERA_ORIENTATION,"Tab"),undefined);
 sameRotation(nudgeCubeView(CAMERA_ORIENTATION,"E")!,nudgeCubeView(CAMERA_ORIENTATION,"e")!);
});

test("continuous drag views feed the existing display-relative move conversion",()=>{
 const camera=axisAngle([Math.PI/2,0,0]);
 const view=dragCubeView(CAMERA_ORIENTATION,[1,0],[0,1],"roll");
 const before=displayCubeBasis(camera),after=displayCubeBasis(multiply(view,camera));
 assert.notEqual(before,after);
 assert.notEqual(reorientedMove("U",before),reorientedMove("U",after));
 assert.notEqual(reorientedMove("y",before),reorientedMove("y",after));
});
