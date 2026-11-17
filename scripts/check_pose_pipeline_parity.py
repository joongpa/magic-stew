"""Run standalone WASM pipeline diagnostics and check documented CPU parity.

The web app, detector models, and video decode are outside this tool. Accepted
exceptions are explicit saved outputs; they cannot silently authorize new
assignment patterns or new differing frames.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pose_assignment_comparison import compare_sticker_assignments

def signature(g,fit,reference):
    r=cv2.Rodrigues(np.asarray(fit['params'][:3]))[0];sym=np.asarray(g['symmetries'])
    which=np.einsum('nij,ij->n',r@sym.transpose(0,2,1),reference).argmax()
    centers=np.asarray(g['faces']).mean(axis=1)
    permutation=np.linalg.norm((centers@sym[which].T)[:,None]-centers[None],axis=2).argmin(axis=1)
    return tuple(np.where(fit['valid'],permutation[np.asarray(fit['nearest'])],-1).tolist())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',default=['matty','nahm','xuanyi','tommy','yiheng'])
    p.add_argument('--input-dir',type=Path,default=ROOT/'pose-fitter/fixtures/full-pipeline')
    p.add_argument('--output-dir',type=Path,default=ROOT/'pose-fitter/results/validated-pipeline')
    p.add_argument('--reuse',action='store_true')
    p.add_argument('--capture-stages',action='store_true')
    p.add_argument('--report',type=Path,default=ROOT/'pose-fitter/benchmarks/full-pipeline-parity.json')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    approved=json.loads((ROOT/'pose-fitter/benchmarks/accepted-pipeline-differences.json').read_text())
    accepted={(r['name'],r['frame']):r for r in approved['cases']}
    report=dict(scope=__doc__,passed=True,acceptance_gate_scope='Rank-0 sticker assignments, recovery decisions, and five-unique-plus-flat policy. Complete ordered-bank equality is reported separately and is not claimed.',
        acceptance_manifest_sha256=hashlib.sha256(json.dumps(approved,sort_keys=True).encode()).hexdigest(),cases=[])
    for name in a.cases:
        path=a.input_dir/f'{name}.json';out=a.output_dir/f'{name}.json'
        if not a.reuse:
            cmd=['node',str(ROOT/'pose-fitter/tools/run-pipeline.mjs'),str(path),str(out)]
            if a.capture_stages:cmd.append('--capture-stages')
            subprocess.run(cmd,check=True,cwd=ROOT)
        expected=json.loads(path.read_text());actual=json.loads(out.read_text());g=expected['geometry']
        if actual['diagnostics']['fixture_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('Stale replay: fixture hash changed')
        if len(expected['frames'])!=len(actual['frames']):raise ValueError('Frame count changed')
        case=dict(name=name,frames=len(actual['frames']),solve_frames=sum(r['frame']>=(expected.get('turning_start_frame')or 0) for r in expected['frames']),
            strict_winner_differing_frames=0,accepted_differing_frames=0,new_differing_frames=0,
            recovery_decision_differences=[],bank_policy_failures=[],ordered_bank_differing_frames=0,unordered_bank_differing_frames=0,
            differing_frames=[],timing=actual['diagnostics'],reference_pixel_scale=expected['pixel_scale_source'])
        for e,b in zip(expected['frames'],actual['frames']):
            assert e['frame']==b['frame'];frame=e['frame'];ref=e['expected'];bank=b['bank']
            if e['expected_depth_recovery']!=b['depth_recovery']:case['recovery_decision_differences'].append(frame)
            if not ref and not bank:continue
            if not ref or not bank:
                case['new_differing_frames']+=1;case['differing_frames'].append(dict(frame=frame,reason='missing fit'));continue
            comparison=compare_sticker_assignments(g,ref[0],bank[0])
            if comparison['sticker_assignment_changes']:
                case['strict_winner_differing_frames']+=1
                approved=accepted.get((name,frame))
                allowed=bool(approved and compare_sticker_assignments(g,approved['accepted_wasm'],bank[0])['sticker_assignment_changes']==0)
                case['accepted_differing_frames']+=int(allowed);case['new_differing_frames']+=int(not allowed)
                case['differing_frames'].append(dict(frame=frame,accepted=allowed,**comparison))
            if len(ref)!=len(bank) or any(compare_sticker_assignments(g,x,y)['sticker_assignment_changes'] for x,y in zip(ref,bank)):
                case['ordered_bank_differing_frames']+=1
            ordinary=[f for f in bank if not f['flat_fallback']];flat=[f for f in bank if f['flat_fallback']]
            reference=cv2.Rodrigues(np.asarray(bank[0]['params'][:3]))[0]
            keys=[signature(g,f,reference) for f in ordinary]
            common=cv2.Rodrigues(np.asarray(ref[0]['params'][:3]))[0]
            expected_keys={signature(g,f,common) for f in ref if not f['flat_fallback']}
            actual_keys={signature(g,f,common) for f in ordinary}
            case['unordered_bank_differing_frames']+=int(expected_keys!=actual_keys)
            if len(ordinary)>5 or len(keys)!=len(set(keys)) or len(flat)!=1 or any(flat[0]['angles']) or not bank[-1]['flat_fallback']:
                case['bank_policy_failures'].append(frame)
        case['passed']=not(case['new_differing_frames']or case['recovery_decision_differences']or case['bank_policy_failures'])
        report['passed']&=case['passed'];report['cases'].append(case)
        print(name,'frames',case['frames'],'strict differences',case['strict_winner_differing_frames'],'accepted',case['accepted_differing_frames'],'new',case['new_differing_frames'],'passed',case['passed'],flush=True)
    report['total_frames']=sum(c['frames']for c in report['cases'])
    report['total_strict_differences']=sum(c['strict_winner_differing_frames']for c in report['cases'])
    report['total_new_differences']=sum(c['new_differing_frames']for c in report['cases'])
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
