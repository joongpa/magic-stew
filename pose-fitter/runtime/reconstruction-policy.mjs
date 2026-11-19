// One policy/clock for the app, workers and independent diagnostics.
export const ADMISSION_CAPACITY = 12;
export const SEARCH_CHUNK = 128;
export const POLICY_VERSION = 1;
export const poseWorkerCount = cores => Math.min(4, Math.max(1, (cores || 4) - 1));
export function createSearchConfig({scramble,fps=30,openingHoldSeconds=0,originSeconds=0,startSeconds,endSeconds,motionCostWeight=0.025,diagnostics=true,bidirectional=false,postprocessIdentityLoops=true,postprocessColorTolerance=0.02}) {
  if (!Number.isFinite(fps)||fps<=0||!Number.isFinite(originSeconds)||!Number.isFinite(openingHoldSeconds)||openingHoldSeconds<0)
    throw Error('Invalid reconstruction source clock or opening hold');
  if (!Number.isFinite(motionCostWeight)||motionCostWeight<0)throw Error('Invalid motion cost weight');
  if (!Number.isFinite(postprocessColorTolerance)||postprocessColorTolerance<0||postprocessColorTolerance>1)
    throw Error('Cleanup color tolerance must be between 0 and 1');
  return {scramble:scramble.trim(),fps,max_tps:54,
    turning_start_frame:openingHoldSeconds>0?Math.floor((originSeconds+openingHoldSeconds)*fps):null,
    branch_widths:[3],branch_stage_expansions:10000,state_limit:1_000_000,
    mismatch_window_weight:1,motion_cost_weight:motionCostWeight,unrestricted_sticker_count:3,pause_at_live_frontier:true,
    expand_flagged_frames:true,resolve_orientation_ambiguity:true,max_rotation_tps:20,diagnostics,bidirectional,
    postprocess_identity_loops:postprocessIdentityLoops,postprocess_color_tolerance:postprocessColorTolerance,
    postprocess_pose_cleanup:postprocessIdentityLoops,
    source_origin_seconds:originSeconds,requested_start_seconds:startSeconds,source_end_seconds:endSeconds};
}
export function videoFrameIdentity(timestamp,config) {
  const elapsed=timestamp-(config.source_origin_seconds??0);
  return {sourceFrame:Math.round(timestamp*config.fps),
    sequence:config.fps>30.001?Math.floor(elapsed*30+0.0001):Math.round(elapsed*config.fps)};
}
