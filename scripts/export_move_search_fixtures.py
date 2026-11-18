"""Create same-fit forward oracles using the bidirectional script's CPU scorer.

Only forward exploration runs, with 54/54 TPS and all five-plus-flat banks.
No detector/fitter is timed or re-run. The Rust port receives no scored rows.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import search_independent_keypoint_bidirectional as b
from scripts import independent_keypoint_pose_geometry as f
from scripts.benchmark_cpu_pose_port import hydrate
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS
r=b.frame_search.replay.load_renderer()

def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [clean(v) for v in value]
 if isinstance(value,np.ndarray):return clean(value.tolist())
 if isinstance(value,np.generic):return clean(value.item())
 if isinstance(value,float) and not np.isfinite(value):return None
 return value
def step(value):
 return {k:clean(value.get(k)) for k in ('frame_index','geometry_option_index','previous_geometry_index',
  'state_moves','state_score','color_error','color_match_count','color_correct_count','color_confusion_count',
  'color_forbidden_count','angular_travel_degrees','rotation_budget_degrees','queue_priority','sibling_rank',
  'window_mismatch_rate','window_mismatch_penalty')}
def build_records(items):
 records={};frames=[]
 for seq,item in enumerate(items):
  bank=item['expected'];obs=item['observations'];params=bank[0]['params'] if bank else [0.,0.,0.,0.,0.,1000.]
  base,_=hydrate(dict(camera=dict(rvec=params[:3],tvec=params[3:6],k=r.default_camera_matrix((item['width'],item['height'])).tolist(),dist=[0.]*4),observations=obs,image_size=[item['width'],item['height']],retained=[],prior=None,neighbors=[],static_mode=False))
  base=replace(base,sticker_color_names=tuple(item['colors']))
  options=[]
  for fit in bank:
   native=f._materialize(r,base,fit['axis'],fit['params']);assert native is not None
   native.score=fit['score'];native.flat_fallback=fit['flat_fallback'];native.nearest=np.asarray(fit['nearest']);native.valid=np.asarray(fit['valid'])
   native.option.update(fit['metrics']);options.append(f.projection_option(r,native))
  records[item['frame']]=dict(frame_index=item['frame'],has_pose=bool(bank),search_pose=dict(sticker_type='joint',rvec=params[:3],tvec=params[3:6],
   camera_matrix=base.camera_matrix.tolist(),dist_coeffs=[0.]*4,image_size=[item['width'],item['height']],sticker_mask_centers=obs['stickers'],sticker_mask_center_uncertainties=obs['uncertainties'],sticker_color_names=item['colors']),
   search=dict(status='prepared_full_search' if bank else 'unavailable_geometry',geometry=dict(joint_refined=True,options=options)))
  frames.append(dict(sequence=seq,frame=item['frame'],bank=bank,colors=item['colors']))
 return records,frames

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',nargs='+',default=list(REFERENCE_SOLUTIONS));p.add_argument('--states',type=int,default=10000)
 p.add_argument('--stage-expansions',type=int);p.add_argument('--window-start',type=int);p.add_argument('--frame-limit',type=int);p.add_argument('--widths',type=int,nargs='+',default=[3]);p.add_argument('--output-dir',type=Path,default=ROOT/'pose-fitter/fixtures/move-search');a=p.parse_args()
 for name in a.cases:
  data=json.loads((ROOT/f'pose-fitter/fixtures/full-pipeline/{name}.json').read_text());saved=json.loads((ROOT/REFERENCE_SOLUTIONS[name]).read_text());scramble=saved['config']['initial_scramble']
  items=[v for v in data['frames'] if a.window_start is None or v['frame']>=a.window_start]
  if a.frame_limit:items=items[:a.frame_limit]
  records,frames=build_records(items)
  colors=b.replay.initial_sticker_colors(r,r.parse_scramble_moves(scramble))
  scorer=b.replay.PreparedReplayScorer(r,records,'cpu',parallel_layers=True,**b.widening_scorer_options(r,colors,data['turning_start_frame'],r.DEFAULT_POSE_MODEL))
  indices=tuple(records);scorer.configure_rotation_budget(indices,data['fps'],54.,54.)
  started=time.perf_counter()
  for frame in indices:scorer.prepare(frame)
  scorer.finalize_rotation_budgets();prepare_seconds=time.perf_counter()-started
  config=dict(scramble=scramble,fps=data['fps'],max_tps=54.,turning_start_frame=data['turning_start_frame'],branch_widths=a.widths,
   branch_stage_expansions=a.stage_expansions or max(1,sum(i>(data['turning_start_frame'] or -1) for i in indices))*100,state_limit=a.states,mismatch_window_weight=1.,unrestricted_sticker_count=3)
  probes=[];original=scorer.rotation_choices;calls=[0];live_calls=[0]
  def choices(frame,state,cue=None):
   values=original(frame,state,cue);calls[0]+=1
   if scorer.hold_reason(frame) is None:live_calls[0]+=1
   if (scorer.hold_reason(frame) is None and live_calls[0]<=24) or calls[0]%1000==0:
    probes.append(dict(sequence=indices.index(frame),state=list(state),cue=cue,choices=[step(dict(c.payload,state_score=c.score)) for c in values]))
   return values
  scorer.rotation_choices=choices
  started=time.perf_counter()
  result=b.frame_search.frame_full_solve(scorer.key(r.CubeGeometryState.identity()),indices,scorer,scorer.solved_predicate(colors),scorer.key,
   serialize_state=scorer.key,deserialize_state=scorer.state_from_key,max_explored_states=a.states,ignore_queue_geometry=True,
   mismatch_window_weight=1.,branch_widths=a.widths,branch_stage_expansions=config['branch_stage_expansions'],unrestricted_sticker_count=3)
  seconds=time.perf_counter()-started
  stats={k:result[k] for k in ('evaluated_states','attempted_transitions','dominated_pruned','reopened_states','improved_pending','deferred_promoted') if k in result}
  out=dict(name=name,config=config,frames=frames,probes=probes,expected=dict(status=result['status'],solved_frame_index=result['solved_frame_index'],steps=[step(v) for v in result['steps']],stats=stats),
   python_prepare_seconds=prepare_seconds,python_search_seconds=seconds,scope=__doc__)
  a.output_dir.mkdir(parents=True,exist_ok=True);path=a.output_dir/f'{name}.json';path.write_text(json.dumps(clean(out),separators=(',',':'),allow_nan=False)+'\n')
  print(name,result['status'],stats,'prepare',round(prepare_seconds,3),'search',round(seconds,3),flush=True)
if __name__=='__main__':main()
