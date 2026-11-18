import assert from "node:assert/strict";
import test from "node:test";
import {cubeMesh} from "./cube-renderer";
import {makeCubePose} from "./cube-motion";
import type {V3} from "./cube-motion";
import {buildColorTimeline,stateAtFrame} from "./cube-colors";
const state=stateAtFrame(buildColorTimeline("",[]),0,true)!;
const cross=(a:V3,b:V3):V3=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const sub=(a:V3,b:V3)=>a.map((v,i)=>v-b[i]) as V3;
const dot=(a:V3,b:V3)=>a.reduce((s,v,i)=>s+v*b[i],0);
function hit(origin:V3,direction:V3,a:V3,b:V3,c:V3){
 const e1=sub(b,a),e2=sub(c,a),p=cross(direction,e2),det=dot(e1,p);if(Math.abs(det)<1e-10)return Infinity;
 const t=sub(origin,a),u=dot(t,p)/det;if(u<0||u>1)return Infinity;
 const q=cross(t,e1),v=dot(direction,q)/det;if(v<0||u+v>1)return Infinity;
 const distance=dot(e2,q)/det;return distance>0?distance:Infinity;
}
test("closed opaque slab bodies occlude exposed internal gaps during a turn",()=>{
 const pose=makeCubePose({axis:0,params:[0,0,0],angles:[0,0,45]},state)!,mesh=cubeMesh(pose);
 assert.equal(mesh.filter(q=>q.kind==="body").length,18);
 assert.equal(mesh.filter(q=>q.kind==="sticker").length,54);
 const eye:V3=[0,0,-4],direction=sub([1/6+.00005,.6,0],eye);
 const hits=mesh.map(quad=>({quad,distance:Math.min(...[[0,1,2],[0,2,3]].map(t=>hit(eye,direction,...t.map(i=>quad.points[i]) as [V3,V3,V3])))}))
  .filter(entry=>Number.isFinite(entry.distance)).sort((a,b)=>a.distance-b.distance);
 assert.ok(hits.length);assert.equal(hits[0].quad.kind,"body","exposed cap must be black plastic, occluding stickers behind it");
 assert.ok(mesh.every(q=>q.points.every(p=>p.every(Number.isFinite))));
});
