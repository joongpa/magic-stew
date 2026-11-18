export interface MoveSearchConfig {
  scramble:string;fps:number;max_tps:54;turning_start_frame:number|null;
  branch_widths:number[];branch_stage_expansions:number;state_limit:number;
  mismatch_window_weight:number;unrestricted_sticker_count:number|null;
  source_origin_seconds?:number;source_end_seconds?:number;
  requested_start_seconds?:number;
}
export interface MoveStep {
  frame_index:number;geometry_option_index:number|null;state_moves:string[];
  state_score:number;color_error:number;queue_priority:number;
}
export interface MoveSearchResult {
  status:"running"|"waiting_for_frames"|"solved"|"already_solved"|"exhausted"|"state_limit";
  frames_received:number;frames_prepared:number;frames_searchable:number;blocked:number;pending:number;
  stats:{evaluated_states:number;furthest_frames:number;branch_stage:number};
  steps:MoveStep[];solved_frame_index:number|null;ended:boolean;
  timings?:{prepareMs:number;searchMs:number};pipelineVersion?:string;
}
