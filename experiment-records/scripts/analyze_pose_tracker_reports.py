"""Re-evaluate recorded banks, pairing full reprojection and reference agreement.

Reference agreement and low residual must occur on the SAME candidate. Saved
reports that separately measure the best-scored residual and best-of-bank
agreement do not establish that conjunction. All work here is evaluation-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_physics_pose_tracker import saved_candidate, final_selected_fit
from scripts import independent_keypoint_pose_geometry as geometry
from scripts import render_yolo11n_independent_keypoints as renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False, parents=True)
    reports = [json.loads(path.read_text()) for path in args.reports]
    output = dict(scope='Post-update evaluation against fitted final-selected references, not independent ground truth.',
        criterion='Same candidate: full robust reprojection <=0.035 cube diameters AND reference projection RMS <=0.03.',
        reports=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=[])
                 for path in args.reports])
    for name in [c['name'] for c in reports[0]['cases']]:
        cases = [next(c for c in report['cases'] if c['name'] == name) for report in reports]
        assert len({c['source_log'] for c in cases}) == 1
        assert len({c['source_solution'] for c in cases}) == 1
        solution_path = Path(cases[0]['source_solution'])
        if not solution_path.is_absolute():
            solution_path = ROOT/solution_path
        solution = json.loads(solution_path.read_text())
        steps = {step['frame_index']:step for step in solution['steps']}
        rows = [{r['frame']:r for r in c['rollout_rows']} for c in cases]
        evaluated = [[] for _ in reports]
        needed = set.union(*(set(r) for r in rows))
        for line in Path(cases[0]['source_log']).open():
            record = json.loads(line)
            frame = record['frame_index']
            if frame not in needed:
                continue
            record['_reference_step'] = steps[frame]
            base = saved_candidate(record)
            target = final_selected_fit(record)
            diameter = renderer.projected_cube_diameter(target.candidate)
            for index, bank in enumerate(rows):
                if frame not in bank:
                    continue
                original = bank[frame]
                errors, residuals = [], []
                for stored in original['pose_bank']:
                    angles = np.asarray(stored['angles'])[[0, 2]]
                    fit = geometry._materialize(renderer, base, stored['axis'],
                        np.r_[np.asarray(stored['rvec']).ravel(), np.asarray(stored['tvec']).ravel(), angles])
                    errors.append(float(np.sqrt(np.mean(np.sum((fit.candidate.assignment_projected_points-
                        target.candidate.assignment_projected_points)**2, axis=1)))/diameter))
                    residuals.append(renderer.pose_reprojection_fit_metric(fit.candidate))
                # Reconstructed bank must reproduce the original evaluation.
                np.testing.assert_allclose(min(errors), original['reference_best_projected_rms_fraction'], atol=1e-10)
                np.testing.assert_allclose(residuals[0], original['reprojection_fit'], atol=1e-10)
                paired = [error <= .03 and residual <= .035 for error, residual in zip(errors, residuals, strict=True)]
                evaluated[index].append(dict(frame=frame, bank_same_candidate_success=any(paired),
                    best_scored_same_candidate_success=paired[0], projection_errors=errors, full_reprojection_errors=residuals,
                    reference_full_reprojection=renderer.pose_reprojection_fit_metric(target.candidate)))
        for index, measured in enumerate(evaluated):
            assert len(measured) == len(rows[index])
            summary = dict(name=name, frames=len(measured),
                bank_same_candidate_success_frames=sum(row['bank_same_candidate_success'] for row in measured),
                best_scored_same_candidate_success_frames=sum(row['best_scored_same_candidate_success'] for row in measured),
                reference_low_reprojection_frames=sum(row['reference_full_reprojection'] <= .035 for row in measured),
                rows=measured)
            output['reports'][index]['cases'].append(summary)
            print(json.dumps(dict(report=str(args.reports[index]), **{k:v for k,v in summary.items() if k != 'rows'})), flush=True)
    (args.output_dir/'comparison.json').write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    import torch
    torch.set_num_threads(2)
    main()
