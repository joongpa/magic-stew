"""Validate a streamed Rust path against independent Python forward transitions."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.export_move_search_fixtures import build_records,b,r

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',default='matty');p.add_argument('--result',type=Path);p.add_argument('--browser-recording',type=Path);a=p.parse_args()
 if a.browser_recording:
  data=json.loads(a.browser_recording.read_text());session=data['moveSearches'][-1];streamed=dict(result=session['result']);fixture=dict(config=session['config'])
  rows={row['result']['fittedPose']['sourceFrame']:row['result'] for row in data['frames'] if row['sourceId']==session['sourceId'] and row['result'].get('fittedPose')}
  items=[]
  for step in streamed['result']['steps']:
   frame=step['frame_index'];value=rows[frame];points=[d for d in value['detections'] if d['classId']<3];stickers=[d for d in value['detections'] if d['classId']>=3]
   items.append(dict(frame=frame,expected=value['fittedPose']['bank'],width=value['transform']['sourceWidth'],height=value['transform']['sourceHeight'],
    colors=[r.CLASS_NAMES[d['classId']] for d in stickers],observations=dict(points=[[p['x'],p['y']] for p in points],classes=[p['classId'] for p in points],stickers=[[p['x'],p['y']] for p in stickers],uncertainties=[1]*len(stickers))))
 else:
  fixture=json.loads((ROOT/f'pose-fitter/fixtures/move-search/{a.case}.json').read_text())
  streamed=json.loads((a.result or ROOT/f'pose-fitter/results/{a.case}-move-search-stream.json').read_text())
  originals={v['frame']:v for v in json.loads((ROOT/f'pose-fitter/fixtures/full-pipeline/{a.case}.json').read_text())['frames']}
  items=[dict(originals[v['frame']],expected=v['bank'],colors=v['colors']) for v in fixture['frames']]
 records,_=build_records(items);config=fixture['config'];colors=b.replay.initial_sticker_colors(r,r.parse_scramble_moves(config['scramble']))
 scorer=b.replay.PreparedReplayScorer(r,records,'cpu',parallel_layers=True,**b.widening_scorer_options(r,colors,config['turning_start_frame'],r.DEFAULT_POSE_MODEL))
 scorer.configure_rotation_budget(tuple(records),config['fps'],54.,54.)
 started=time.perf_counter();state=scorer.key(r.CubeGeometryState.identity());cue=None;window=();checked=0
 for step in streamed['result']['steps']:
  frame=step['frame_index'];options=scorer.rotation_choices(frame,state,cue);held=scorer.hold_reason(frame) is not None
  matching=[c for c in options if list(c.payload['state_moves'])==step['state_moves']
   and (held or c.payload['geometry_option_index']==step['geometry_option_index'])
   and math.isclose(c.score,step['state_score'],rel_tol=1e-12,abs_tol=1e-12)]
  assert matching,(frame,step)
  choice=matching[0];payload=dict(choice.payload)
  window=b.frame_search.update_mismatch_window(window,payload,config['mismatch_window_weight'])
  priority=.5*(payload.get('color_error') or 0.)+payload['window_mismatch_penalty']
  assert math.isclose(priority,step['queue_priority'],abs_tol=1e-12),(frame,'priority')
  state=choice.state;cue=scorer.rotation_context(payload,cue);checked+=1
 solved=scorer.solved_predicate(colors)(state)
 if streamed['result']['status'] in ('solved','already_solved'):assert solved
 report=dict(case=a.case,status=streamed['result']['status'],checked_frames=checked,solved=solved,
  expanded_states=streamed['result']['stats']['evaluated_states'],python_validation_seconds=time.perf_counter()-started,
  scope='Every streamed edge, geometry index, state score and queue priority checked against the Python forward scorer; no future-frame traversal')
 path=ROOT/f'pose-fitter/benchmarks/{a.case}{"-browser" if a.browser_recording else ""}-move-search-stream.json';path.write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
