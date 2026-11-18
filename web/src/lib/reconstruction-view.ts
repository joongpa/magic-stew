import type { FittedPose } from "./pose-types";
import type { CubeColor } from "./cube-colors";
import { fixedCubeProjection } from "./cube-projection";
/** Meshes belong to the same frame's immutable geometry bank. */
export function projectedGeometryIndex(pose: FittedPose, geometryIndex?: number | null, staticHold = false) {
  let index = geometryIndex ?? 0;
  if (staticHold) index = pose.bank.findIndex(fit => fit.flat_fallback);
  if (index < 0 || index >= pose.bank.length) index = 0;
  return index;
}
export function projectedWireframe(pose: FittedPose, geometryIndex?: number | null, staticHold = false) {
  return pose.wireframes?.[projectedGeometryIndex(pose,geometryIndex,staticHold)] ?? pose.wireframe;
}
export interface ColoredFace { face:number;points:[number,number][];color:CubeColor;depth:number }
export function projectedColoredFaces(pose:FittedPose,colors:CubeColor[]|undefined,geometryIndex?:number|null,staticHold=false):ColoredFace[] {
  if (!colors) return [];
  const fit=pose.bank[projectedGeometryIndex(pose,geometryIndex,staticHold)];
  if (!fit) return [];
  return fixedCubeProjection(fit).map(face=>({...face,color:colors[face.face]}))
    .filter(face=>face.color && Number.isFinite(face.depth))
    .sort((a,b)=>b.depth-a.depth||a.face-b.face);
}
