import orientations from "../../../pose-fitter/tables/orientations.json";
import turns from "../../../pose-fitter/tables/render-turns.json";
import { axisAngle, multiply, rotate } from "./cube-motion";
import type { CubePose, Quaternion, V3 } from "./cube-motion";
import type { MoveStep } from "./move-search-types";
import { projectedCubeAtFrame } from "./projected-cube";
import type { ProjectedCubeFrame } from "./projected-cube";

const moveTurns=turns as Record<string,{axis:number;angles:number[]}>;
const clean=(move:string)=>move.replaceAll("’","'").replace(/2'$/,"2");
const inverse=(q:Quaternion):Quaternion=>[-q[0],-q[1],-q[2],q[3]];
const quarter=(angle:number)=>((angle%360)+360)%360;
const moveCache=new Map<string,string>();
const identity=orientations.identity_index;
export const CAMERA_ORIENTATION:Quaternion=[0,0,0,1];
const orientationQuaternions=orientations.rvecs.map(vector=>axisAngle(vector as V3));
const orientationIndices=new Map(orientations.matrices.map((matrix,index)=>[matrix.flat().join(","),index]));
const normals:V3[]=[[0,0,1],[0,0,-1],[0,-1,0],[0,1,0],[1,0,0],[-1,0,0]];
const dot=(a:V3,b:V3)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
const cross=(a:V3,b:V3):V3=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];

/** Map model axes to displayed cube faces. Choose U first, then the most
 * viewer-facing perpendicular face as F. OpenCV screen-up is -Y, toward us -Z.
 * Fixed candidate order resolves exact ties without floating-point chatter. */
export function displayCubeBasis(rotation:Quaternion):number {
 const best=(candidates:V3[],direction:V3)=>candidates.reduce((chosen,normal)=>
  dot(rotate(rotation,normal),direction)>dot(rotate(rotation,chosen),direction)+1e-10?normal:chosen);
 const up=best(normals,[0,-1,0]);
 const front=best(normals.filter(normal=>dot(normal,up)===0),[0,0,-1]);
 const right=cross(up,front);
 return orientationIndices.get([...right,...front.map(v=>-v),...up].join(","))!;
}

function orientedLayers(axis:number,angles:readonly number[],orientation:number) {
 const matrix=orientations.matrices[orientation];
 const target=matrix.findIndex(row=>row[axis]!==0),sign=matrix[target][axis];
 return {axis:target,angles:(sign>0?[...angles]:[-angles[2],-angles[1],-angles[0]]) as V3};
}

/** Conjugate every supported face, wide, slice or whole-cube turn. */
export function reorientedMove(raw:string,orientation:number):string {
 const move=clean(raw),turn=moveTurns[move];
 if(!turn)throw new Error(`Unknown cube turn: ${raw}`);
 const key=`${orientation}/${move}`,cached=moveCache.get(key);if(cached)return cached;
 const mapped=orientedLayers(turn.axis,turn.angles,orientation);
 const lowercase=/^[rludfb]/.test(move);
 const match=Object.entries(moveTurns).find(([name,candidate])=>
  /^[rludfb]/.test(name)===lowercase&&candidate.axis===mapped.axis&&
  candidate.angles.every((angle,i)=>quarter(angle)===quarter(mapped.angles[i])));
 if(!match)throw new Error(`Cannot reorient cube turn: ${raw}`);
 moveCache.set(key,match[0]);return match[0];
}

function rotationForMove(move:string):Quaternion {
 const turn=moveTurns[move];
 if(!turn||!turn.angles.every(angle=>angle===turn.angles[0]))throw new Error("Cube setup needs an x, y or z rotation");
 const vector:V3=[0,0,0];vector[turn.axis]=turn.angles[0]*Math.PI/180;
 return axisAngle(vector);
}

/** Rotate about the actual cube face axes currently named R/U/F on screen.
 * Keep a camera-space view offset so every later tracked rotation is carried
 * through the same view change, rather than rotating only the sticker state. */
export function rotateCubeView(view:Quaternion,cube:Quaternion,move:string):Quaternion {
 const displayed=multiply(view,cube),basis=orientationQuaternions[displayCubeBasis(displayed)];
 const local=multiply(multiply(inverse(basis),rotationForMove(move)),basis);
 const cameraTurn=multiply(multiply(displayed,local),inverse(displayed));
 const next=multiply(cameraTurn,view),length=Math.hypot(...next);
 return next.map(value=>value/length) as Quaternion;
}

const nearestOrientation=(q:Quaternion)=>orientationQuaternions.reduce((best,candidate,index)=>{
 const score=Math.abs(candidate.reduce((sum,v,i)=>sum+v*q[i],0));
 return score>best.score?{index,score}:best;
},{index:identity,score:-1}).index;

// These words compose in move order: each token acts in the displayed frame.
const rotationWords=(()=>{
 const words=new Map<number,string[]>([[identity,[]]]),queue=[identity];
 for(const basis of queue)for(const move of ["x","x'","x2","y","y'","y2","z","z'","z2"]){
  const next=nearestOrientation(multiply(rotationForMove(move),orientationQuaternions[basis]));
  if(!words.has(next)){words.set(next,[...words.get(basis)!,move]);queue.push(next);}
 }
 return words;
})();

export interface ReorientedMoveStep { frame:number; moves:string[]; rotation:boolean }
/** Both turns and tracked rotations use the same absolute displayed-face
 * assignment as the controls, including the opening frame's camera angle. */
export function reorientedMoveTimeline(rows:readonly ProjectedCubeFrame[],steps:readonly MoveStep[],view:Quaternion):ReorientedMoveStep[] {
 if(!rows.length)return [];
 const byFrame=new Map(steps.map(step=>[step.frame_index,step]));
 const frames=[...new Set([...rows.map(row=>row.frame),...steps.map(step=>step.frame_index)])].sort((a,b)=>a-b);
 const result:ReorientedMoveStep[]=[];
 let previous:number|undefined;
 for(const frame of frames){
  const pose=projectedCubeAtFrame(rows,frame)!;
  const basis=displayCubeBasis(multiply(view,pose.rotation));
  if(previous!=null&&basis!==previous){
   const delta=nearestOrientation(multiply(orientationQuaternions[basis],inverse(orientationQuaternions[previous])));
   const moves=rotationWords.get(delta)!;
   if(moves.length)result.push({frame,moves,rotation:true});
  }
  const moves=byFrame.get(frame)?.state_moves;
  if(moves?.length)result.push({frame,moves:moves.map(move=>reorientedMove(move,basis)),rotation:false});
  previous=basis;
 }
 return result;
}

/** Apply the view to the entire tracked pose. Stickers, signed partial turns
 * and the animation history stay in their original frame during smoothing. */
export function reorientedCubePose(pose:CubePose,view:Quaternion):CubePose {
 if(view.every((v,i)=>v===CAMERA_ORIENTATION[i]))return pose;
 return {...pose,rotation:multiply(view,pose.rotation)};
}
