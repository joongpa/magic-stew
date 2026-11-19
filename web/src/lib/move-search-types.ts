export type {SearchConfig as MoveSearchConfig} from '../../../pose-fitter/runtime/reconstruction-policy.mjs';
export interface MoveStep {
  pose_flagged?:boolean;
  frame_index:number;geometry_option_index:number|null;state_moves:string[];
  state_score?:number;color_error?:number;queue_priority?:number;
  recent_motion_penalty?:number;required_tps_ratio?:number;
  orientation_symmetry_index?:number;
  cube_rotation_motion?:{rigid_degrees:number;combined_degrees:number};
  cube_rotation_budget_degrees?:number;
}
export interface MoveSearchResult {
  status:"running"|"waiting_for_frames"|"solved"|"already_solved"|"exhausted"|"state_limit";
  frames_received:number;frames_prepared:number;frames_searchable:number;blocked:number;pending:number;
  furthest_admitted_frames?:number;
  search_direction?:"forward"|"bidirectional";
  bidirectional_fallback?:string;
  solved_anchor_frame?:number;join_frame_index?:number|null;
  flagged_frames?:{frame:number;reason:string}[];
  stats:{evaluated_states:number;furthest_frames:number;branch_stage:number;forward_expanded?:number;backward_expanded?:number};
  steps:MoveStep[];solved_frame_index:number|null;ended:boolean;
  timings?:{prepareMs:number;searchMs:number};pipelineVersion?:string;
}
