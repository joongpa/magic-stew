"""Replay identical five-plus-flat pose banks through Python, native Rust and WASM."""
from pathlib import Path
import argparse,json,subprocess,math
ROOT=Path(__file__).resolve().parents[1]
def close(a,b):
 if isinstance(a,(int,float)) and isinstance(b,(int,float)):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
 return a==b
def compare(fixture,actual):
 issues=[];expected=fixture['expected'];result=actual['result']
 for field in ('status','solved_frame_index'):
  if expected[field]!=result[field]:issues.append(f'{field}: {expected[field]} != {result[field]}')
 for key,value in expected['stats'].items():
  if value!=result['stats'][key]:issues.append(f'{key}: {value} != {result["stats"][key]}')
 fields=('geometry_option_index','previous_geometry_index','state_moves','state_score','color_error','color_match_count',
  'color_correct_count','color_confusion_count','color_forbidden_count','angular_travel_degrees','rotation_budget_degrees')
 for i,(probe,output) in enumerate(zip(fixture['probes'],actual['probes'])):
  a,b=probe['choices'],output['choices']
  if len(a)!=len(b):issues.append(f'probe {i}: {len(a)} != {len(b)} choices');continue
  for j,(x,y) in enumerate(zip(a,b)):
   for key in fields:
    if x[key] is not None and not close(x[key],y[key]):issues.append(f'probe {i}/{j} {key}: {x[key]} != {y[key]}');break
 if len(fixture['probes'])!=len(actual['probes']):issues.append('probe count')
 if expected['status'] in ('solved','already_solved'):
  if len(expected['steps'])!=len(result['steps']):issues.append('path length')
  for i,(x,y) in enumerate(zip(expected['steps'],result['steps'])):
   for key in (*fields,'queue_priority','sibling_rank','window_mismatch_rate','window_mismatch_penalty'):
    if x[key] is not None and not close(x[key],y[key]):issues.append(f'path {i} {key}: {x[key]} != {y[key]}');break
 return issues
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',nargs='+',default=['matty','nahm','xuanyi','tommy','yiheng']);p.add_argument('--reuse',action='store_true');p.add_argument('--wasm',action='store_true');a=p.parse_args()
 report=[]
 for name in a.cases:
  source=ROOT/f'pose-fitter/fixtures/move-search/{name}.json';fixture=json.loads(source.read_text());out=ROOT/f'pose-fitter/results/{name}-move-search{"-wasm" if a.wasm else ""}.json'
  if not a.reuse:
   command=['node',str(ROOT/'pose-fitter/tools/run-move-search.mjs')] if a.wasm else [str(ROOT/'pose-fitter/target/release/move-search-benchmark')]
   subprocess.run([*command,str(source),str(out)],check=True,cwd=ROOT)
  actual=json.loads(out.read_text());issues=compare(fixture,actual)
  row=dict(name=name,runtime='wasm' if a.wasm else 'native',frames=len(fixture['frames']),status=actual['result']['status'],
   evaluated_states=actual['result']['stats']['evaluated_states'],candidate_probes=sum(len(p['choices']) for p in fixture['probes']),
   python_prepare_seconds=fixture['python_prepare_seconds'],python_search_seconds=fixture['python_search_seconds'],
   rust_prepare_seconds=actual['prepare_seconds'],rust_search_seconds=actual['search_seconds'],issues=issues)
  report.append(row);print(name,row['status'],len(issues),'issues',issues[:4],flush=True)
 path=ROOT/f'pose-fitter/benchmarks/move-search-{"wasm" if a.wasm else "native"}.json';path.write_text(json.dumps(report,indent=2)+'\n')
 if any(r['issues'] for r in report):raise SystemExit(1)
if __name__=='__main__':main()
