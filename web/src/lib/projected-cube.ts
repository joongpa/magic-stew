import { stateAtFrame } from "./cube-colors";
import type { ColorTimeline } from "./cube-colors";
import { layerTransition, makeCubePose } from "./cube-motion";
import type { CubePose } from "./cube-motion";
import type { MoveStep } from "./move-search-types";
import { poseIsFlagged, projectedGeometryIndex } from "./reconstruction-view";
import type { SavedVideoResult } from "./video-result-cache";

export interface ProjectedCubeFrame { frame:number; pose:CubePose }
const linear=(t:number)=>t;

function lowerBound(rows:readonly ProjectedCubeFrame[],frame:number) {
 let lo=0,hi=rows.length;
 while(lo<hi){const mid=(lo+hi)>>>1;if(rows[mid].frame<frame)lo=mid+1;else hi=mid;}
 return lo;
}

/** Source-time interpolation is deterministic during playback and scrubbing.
 * The component separately eases each new target in wall-clock time. */
export function projectedCubeAtFrame(rows:readonly ProjectedCubeFrame[],frame:number):CubePose|undefined {
 const index=lowerBound(rows,frame),before=rows[index-1],after=rows[index];
 if(!before)return after?.pose;
 if(!after)return before.pose;
 if(after.frame===frame)return after.pose;
 return layerTransition(before.pose,after.pose).sample((frame-before.frame)/(after.frame-before.frame),linear);
}

/** Rejected fits never become render anchors. Bridge them using accepted
 * orientations and legal layer transitions, including commits in pose gaps. */
export function buildProjectedCubeTimeline(rows:readonly SavedVideoResult[],colors:ColorTimeline,
 options:{fps:number;solved:boolean;steps:readonly MoveStep[];turningStartFrame?:number|null}):ProjectedCubeFrame[] {
 const steps=new Map(options.steps.map(step=>[step.frame_index,step]));
 const accepted:ProjectedCubeFrame[]=[];
 for(const row of rows){
  const pose=row.result.fittedPose,frame=pose?.sourceFrame??Math.round(row.timestamp*options.fps),step=steps.get(frame);
  if(!pose?.pose||poseIsFlagged(pose,step?.pose_flagged))continue;
  const state=stateAtFrame(colors,frame,options.solved);
  const index=projectedGeometryIndex(pose,step?.geometry_option_index,
   options.turningStartFrame!=null&&frame<=options.turningStartFrame);
  const fit=pose.bank[index],cube=state&&fit?makeCubePose(fit,state,step?.orientation_symmetry_index):undefined;
  if(cube)accepted.push({frame,pose:cube});
 }
 accepted.sort((a,b)=>a.frame-b.frame);
 if(!accepted.length)return [];
 const timeline=new Map(accepted.map(row=>[row.frame,row]));
 for(const [index,state] of colors.frames.entries()){
  // A commit with no accepted geometry still gets an aligned target at its
  // own source frame. Keep the initial state before a leading pose gap too.
  if(timeline.has(state.frame)||(index!==0&&!steps.get(state.frame)?.state_moves.length))continue;
  const nearest=projectedCubeAtFrame(accepted,state.frame)!;
  timeline.set(state.frame,{frame:state.frame,pose:{...nearest,angles:[0,0,0],state}});
 }
 return [...timeline.values()].sort((a,b)=>a.frame-b.frame);
}
