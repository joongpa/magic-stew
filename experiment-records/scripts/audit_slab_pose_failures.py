"""Diagnose the failed slab tracker; reference-based interventions are NOT tracking.

Compare complete wireframes, not just projections assigned to observed points.
Hold reference translation or inject a reference static pose only as diagnostic
ablations. No output from this script is a new causal tracking benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import render_yolo11n_independent_keypoints as m
from scripts import independent_keypoint_pose_geometry as geometry
from scripts.benchmark_physics_pose_tracker import saved_candidate, final_selected_fit
from scripts.independent_keypoint_slab_search import SlabScorer, StaticSlabPose, fit_articulation
from scripts.pose_wireframe_metrics import center_pixels, wire_mask, wire_error


def fit_from_stored(base, stored, translation=None):
    return geometry._materialize(m, base, stored['axis'], np.r_[np.asarray(stored['rvec']).ravel(),
        np.asarray(stored['tvec']).ravel() if translation is None else np.asarray(translation).ravel(),
        np.asarray(stored['angles'])[[0, 2]]])


def summarize(values):
    if not values:
        return None
    return dict(median=float(np.median(values)), p90=float(np.percentile(values, 90)))


def audit_case(case, rows, max_interventions):
    steps = {s['frame_index']:s for s in json.loads(Path(case['source_solution']).read_text())['steps']}
    rows = {row['frame']:row for row in rows}
    measured, interventions = [], []
    candidates = []
    for line in Path(case['source_log']).open():
        record = json.loads(line)
        frame = record['frame_index']
        if frame not in rows or frame not in steps:
            continue
        record['_reference_step'] = steps[frame]
        target = final_selected_fit(record)
        reference = target.candidate
        base = saved_candidate(record)
        stored = rows[frame]['pose_bank'][0]
        fit = fit_from_stored(base, stored)
        diameter = m.projected_cube_diameter(reference)
        center = center_pixels(reference, reference.tvec)
        shift = float(np.linalg.norm(center_pixels(reference, fit.candidate.tvec)-center)/diameter)
        depth = float(abs(fit.candidate.tvec.ravel()[2]/reference.tvec.ravel()[2]-1.))
        correct_core = shift <= .02 and depth <= .02
        mask = wire_mask(reference, center, diameter)
        error = wire_error(fit.candidate, reference, mask)
        translation_only = fit_from_stored(base, stored, translation=reference.tvec)
        corrected_error = wire_error(translation_only.candidate, reference, mask)
        value = dict(frame=frame, core_center_error_fraction=shift, depth_error_fraction=depth,
            correct_core_by_2percent_gate=correct_core, wire_p90_fraction=error,
            full_reprojection=m.pose_reprojection_fit_metric(fit.candidate),
            reference_full_reprojection=m.pose_reprojection_fit_metric(reference),
            wire_p90_with_reference_translation=corrected_error)
        if correct_core and error > .02:
            bank = [error]+[wire_error(fit_from_stored(base, s).candidate, reference, mask)
                           for s in rows[frame]['pose_bank'][1:]]
            value['best_bank_wire_p90_fraction'] = min(bank)
            candidates.append((record, base, target, value))
        measured.append(value)
    # Spread reference-only diagnostic interventions across the failed frames.
    indices = np.unique(np.linspace(0, len(candidates)-1, min(max_interventions, len(candidates)), dtype=int)) if candidates else []
    for index in indices:
        record, base, target, value = candidates[index]
        reference = target.candidate
        x = np.r_[reference.rvec.ravel(), reference.tvec.ravel()]
        selected = SlabScorer(m, base).score([x], [0])[0]
        known_axis = StaticSlabPose(x.copy(), target.axis, 1, 0., 8, 3, 0)
        def evaluate_anchor(proposal):
            fits = fit_articulation(m, base, proposal, iterations=3)
            winner = min(fits, key=lambda f:f.score)
            return dict(axis=proposal.axis, anchor=proposal.layer,
                reprojection=m.pose_reprojection_fit_metric(winner.candidate),
                wire_p90=wire_error(winner.candidate, reference))
        locked = evaluate_anchor(selected)
        correct_axis = evaluate_anchor(known_axis)
        interventions.append(dict(frame=value['frame'], original=value,
            oracle_static_pose_selected_slab=locked, oracle_static_pose_reference_axis=correct_axis,
            reference_axis=target.axis, reference_angles=list(target.angles), selected_static_score=selected.score))
    good = [v for v in measured if v['correct_core_by_2percent_gate']]
    bad = [v for v in good if v['wire_p90_fraction'] > .02]
    return dict(name=case['name'], frames=len(measured), correct_core_frames=len(good),
        correct_core_but_wire_error_over_2percent_frames=len(bad),
        correct_core_wire_error=summarize([v['wire_p90_fraction'] for v in good]),
        correct_core_bad_wire_with_low_reprojection_frames=sum(v['full_reprojection'] <= .035 for v in bad),
        correct_core_bad_wire_still_bad_after_translation_repair_frames=sum(v['wire_p90_with_reference_translation'] > .02 for v in bad),
        correct_core_bad_wire_has_good_alternative_in_bank_frames=sum(v['best_bank_wire_p90_fraction'] <= .02 for v in bad),
        selected_axis_intervention_success_frames=sum(v['oracle_static_pose_selected_slab']['wire_p90'] <= .02 and
            v['oracle_static_pose_selected_slab']['reprojection'] <= .035 for v in interventions),
        reference_axis_intervention_success_frames=sum(v['oracle_static_pose_reference_axis']['wire_p90'] <= .02 and
            v['oracle_static_pose_reference_axis']['reprojection'] <= .035 for v in interventions),
        interventions=interventions, rows=measured)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--report', type=Path)
    source.add_argument('--render-manifest', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--interventions-per-case', type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = []
    if args.report:
        cases = [(case, case['rollout_rows']) for case in json.loads(args.report.read_text())['cases']]
    else:
        for case in json.loads(args.render_manifest.read_text())['cases']:
            rows = [json.loads(line) for line in (args.render_manifest.parent/case['name']/'tracking.jsonl').open()]
            cases.append((dict(name=case['name'], source_log=case['observations'], source_solution=case['reference_solution']), rows))
    result = dict(scope='Failure analysis only. Reference interventions use current reference poses and are NOT causal predictions.',
        wire_metric='Symmetric p90 raster wireframe distance at 256px, normalized by reference cube diameter; geometry only, no face-color identities.',
        core_gate='Projected center error <=2% of cube diameter and relative depth error <=2%.', cases=[])
    for case, rows in cases:
        audited = audit_case(case, rows, args.interventions_per_case)
        result['cases'].append(audited)
        (args.output_dir/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in audited.items() if k not in ('interventions', 'rows')}), flush=True)


if __name__ == '__main__':
    import torch
    torch.set_num_threads(2)
    main()
