#!/usr/bin/env python3
"""Compare saved winner, reference-constrained paths, and retained search states.

Reference timing is inferred retrospectively by minimum summed frame cost.
This diagnostic never alters or seeds the production search.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import pickle
import sqlite3
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "score_analysis_search", ROOT / "scripts/search_independent_keypoint_frame.py")
search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = search
SPEC.loader.exec_module(search)


def details(step, branch):
    error = step.get("color_error")
    total = step.get("color_match_count", 0)
    accuracy = 1. - error if error is not None and total else None
    return dict(branch=branch, frame_index=step['frame_index'],
        geometry=step.get('score_components', {}).get('geometry'),
        color_error=error, color_accuracy=accuracy,
        correct=round(accuracy*total) if accuracy is not None else None, observed=total,
        combined=step['state_score'], keypoint_error=step.get('keypoint_mean_error'),
        sticker_position_penalty=step.get('heatmap_mean_error'),
        keypoint_inliers=step.get('keypoint_inlier_count'),
        keypoint_layer_inliers=step.get('keypoint_layer_inlier_count'),
        keypoint_p90_error=step.get('keypoint_p90_error'),
        keypoint_zero_fallback=(step.get('keypoint_mean_error') == 0. and
                               step.get('keypoint_inlier_count') == 0),
        moves=' '.join(step.get('state_moves', [])), trial_id=step.get('trial_id'),
        reference_progress=step.get('reference_progress'),
        color_pass=accuracy >= 2./3. - 1e-12 if accuracy is not None else None)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(name, rows, steps, reference_quarters=None):
    observed = [r for r in rows if r['geometry'] is not None and r['color_accuracy'] is not None]
    return dict(branch=name, frames=len(observed),
        mean_geometry=float(np.mean([r['geometry'] for r in observed])),
        mean_color_accuracy=float(np.mean([r['color_accuracy'] for r in observed])),
        pooled_color_accuracy=sum(r['correct'] for r in observed)/sum(r['observed'] for r in observed),
        mean_combined=float(np.mean([r['combined'] for r in observed])),
        max_combined=max(r['combined'] for r in observed),
        minimum_color_accuracy=min(r['color_accuracy'] for r in observed),
        color_pruned_frames=[r['frame_index'] for r in observed if not r['color_pass']],
        move_count=sum(len(s.get('state_moves', [])) for s in steps),
        keypoint_zero_fallback_frames=[r['frame_index'] for r in rows if r['keypoint_zero_fallback']],
        solved_frame_index=next((s['frame_index'] for s in steps
                                if reference_quarters is not None and
                                s.get('reference_progress') == reference_quarters), None))


def local_candidate_rows(m, scorer, branches, gap):
    rows = []
    initial = scorer.key(m.CubeGeometryState.identity())
    for name in ('found', 'reference_gated', 'reference_ungated'):
        if name not in branches:
            continue
        state = initial
        update = (None, None, 0)
        for step in branches[name]:
            frame = step['frame_index']
            choices, _ = scorer.for_update_context(frame, state, update, gap)
            selected_state = scorer.apply_moves(state, tuple(step['state_moves']))
            for rank, choice in enumerate(choices, 1):
                row = details(dict(choice.payload, frame_index=frame), 'local_candidate')
                row.update(parent_branch=name, local_rank=rank,
                    selected_trial=choice.payload.get('trial_id') == step.get('trial_id'),
                    selected_resulting_state=choice.state == selected_state)
                rows.append(row)
            state = selected_state
            chosen = search.replay.Transition(state, step['state_score'], (), step,
                                              m.state_update_layer_direction(step))
            update = search.advance_update(update, chosen, frame, gap)
            assert update is not None
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path, help='JSON with scramble and reference sequence.')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run = args.result if args.result.is_dir() else args.result.parent
    result = json.loads((run/'result.json').read_text())
    if result['status'] != 'solved':
        parser.error('A solved result is required')
    out = args.output_dir or run/'score-analysis'
    out.mkdir(parents=True, exist_ok=True)
    reference = json.loads(args.reference.read_text())
    config = result['config']
    if reference['scramble'] != config['initial_scramble']:
        parser.error('Reference scramble differs from the run')
    import torch
    torch.set_num_threads(2)
    m = search.replay.load_renderer()
    records = {r['frame_index']: r for r in
               (json.loads(line) for line in Path(config['log']).read_text().splitlines() if line.strip())}
    colors = search.replay.initial_sticker_colors(m, config['initial_scramble'])
    kwargs = dict(initial_colors=colors, turning_start_frame=config['turning_start_frame'],
                  balance_geometry_color=True, cache_size=8192)
    gated = search.replay.PreparedReplayScorer(m, records, 'cpu', minimum_color_accuracy=2./3., **kwargs)
    ungated = search.replay.PreparedReplayScorer(m, records, 'cpu', minimum_color_accuracy=None, **kwargs)
    # Share immutable frame projections; evaluator caches remain independent.
    ungated.compiled_frames = gated.compiled_frames
    positions = gated.positions
    indices = {p: i for i,p in enumerate(positions)}
    permutations = []
    for symmetry in m.CUBE_SYMMETRY_MATRICES:
        sources = dict(m._geometry_symmetry_permutation(tuple(int(v) for v in symmetry.ravel())))
        permutations.append([indices[sources[p]] for p in positions])
    permutations = np.asarray(permutations)
    codes = {c: i for i,c in enumerate(sorted(set(m.CubeState.solved().colors.values())))}
    translation = bytes(codes[colors[label]] for label in sorted(gated.identity_indices)) + bytes(202)

    def canonical_colors(encoded):
        return min(row.tobytes() for row in encoded[permutations])

    @lru_cache(maxsize=131072)
    def canonical(key):
        return canonical_colors(np.frombuffer(key.translate(translation), dtype=np.uint8))

    def cube_key(state):
        return canonical_colors(np.asarray([codes[state.color_at(p)] for p in positions], dtype=np.uint8))

    cube = m.CubeState.from_scramble(reference['scramble'])
    reference_states = {cube_key(cube): {0}}
    quarter = 0
    for raw in reference['reference'].split():
        move = raw.replace("2'", '2')
        if move[0] in 'xyz':
            cube.apply_move(move)
        elif move.endswith('2'):
            base = move[:-1]
            for alternative in (base, base+"'"):
                midway = m.CubeState(dict(cube.colors))
                midway.apply_move(alternative)
                reference_states.setdefault(cube_key(midway), set()).add(quarter+1)
            cube.apply_move(move)
            quarter += 2
        else:
            cube.apply_move(move)
            quarter += 1
        reference_states.setdefault(cube_key(cube), set()).add(quarter)
    assert cube.is_solved()
    initial = gated.key(m.CubeGeometryState.identity())
    turning_start = config['turning_start_frame']
    start = max(config['start_frame'], turning_start+1 if turning_start is not None else config['start_frame'])
    end = result['solved_frame_index']
    gap = config['min_state_update_gap_frames']
    fronts = {name: {(initial, (None, None, 0), 0): (0., None)}
              for name in ('reference_gated', 'reference_ungated')}
    reference_frames = []
    for frame in range(start, end+1):
        gated.prepare(frame)
        for name, scorer in (('reference_gated', gated), ('reference_ungated', ungated)):
            prior = fronts[name]
            following = {}
            for (state, update, progress), (cost, parent) in prior.items():
                choices, _ = scorer.for_update_context(frame, state, update, gap)
                for choice in choices:
                    updated = search.advance_update(update, choice, frame, gap)
                    if updated is None:
                        continue
                    moved = len(choice.payload['state_moves'])
                    target = progress + moved
                    matches = reference_states.get(canonical(scorer.key(choice.state)), set())
                    if target not in matches:
                        continue
                    key = (scorer.key(choice.state), search.canonical_update(updated, frame+1, gap), target)
                    candidate_cost = cost + choice.score
                    previous = following.get(key)
                    if previous is None or candidate_cost < previous[0]:
                        step = dict(choice.payload, frame_index=frame, reference_progress=target)
                        following[key] = (candidate_cost, (parent, step))
            fronts[name] = following
            reference_frames.append(dict(branch=name, frame_index=frame, prior_contexts=len(prior),
                surviving_contexts=len(following), maximum_progress=max((k[2] for k in following), default=None)))
        if (frame-start) % 10 == 0:
            print(f'Reference alignment frame {frame}/{end}: ' + ', '.join(
                f'{name}={len(front)}' for name,front in fronts.items()), flush=True)
    branches = {'found': [s for s in result['steps'] if start <= s['frame_index'] <= end]}
    reference_feasible = {}
    for name, front in fronts.items():
        solved = [value for key,value in front.items() if key[2] == quarter]
        reference_feasible[name] = bool(solved)
        if not solved:
            continue
        _, node = min(solved, key=lambda x:x[0])
        steps = []
        while node is not None:
            node, step = node
            steps.append(step)
        branches[name] = list(reversed(steps))

    print('Extracting retained search contexts', flush=True)
    connection = sqlite3.connect(f'file:{(run/"search.sqlite3").resolve()}?mode=ro', uri=True)
    paths = {index: (parent, pickle.loads(zlib.decompress(blob)))
             for index,parent,blob in connection.execute('SELECT id,parent,step FROM paths')}
    found_states = {}
    state = initial
    for step in result['steps']:
        state = gated.apply_moves(state, tuple(step.get('state_moves', [])))
        found_states[step['frame_index']] = state
    found_final = canonical(state)
    context_rows = []
    representatives = []
    by_frame = {}
    for key, depth, pending, priority, payload in connection.execute(
            'SELECT key,depth,pending,priority,payload FROM contexts WHERE depth BETWEEN ? AND ?',
            (start-config['start_frame']+1, end-config['start_frame']+1)):
        encoded, _update, path_id, _recent = pickle.loads(zlib.decompress(payload))
        step = paths[path_id][1]
        frame = step['frame_index']
        assert abs(priority-step['state_score']) < 1e-9
        if encoded == found_states[frame]:
            continue
        row = details(step, 'other_retained_context')
        row.update(path_id=path_id, pending=bool(pending), depth=depth)
        context_rows.append(row)
        if row['geometry'] is not None and row['color_accuracy'] is not None:
            by_frame.setdefault(frame, []).append(row)
        if frame == end and pending and canonical(encoded) != found_final:
            representatives.append((priority, path_id, encoded))
    representatives.sort(key=lambda item:(item[0], item[1]))
    chosen_colors = set()
    alternative_metadata = []
    for priority, path_id, encoded in representatives:
        color_state = canonical(encoded)
        if color_state in chosen_colors:
            continue
        chosen_colors.add(color_state)
        name = f'queued_alternative_{len(chosen_colors)}'
        steps = []
        cursor = path_id
        while cursor:
            cursor, step = paths[cursor]
            if start <= step['frame_index'] <= end:
                steps.append(step)
        branches[name] = list(reversed(steps))
        alternative_metadata.append(dict(branch=name, path_id=path_id, endpoint_frame=end,
            latest_score=priority, solved=gated.solved_predicate(colors)(encoded),
            selection='lowest latest-frame cost among pending, distinct unsolved color states at the winner endpoint'))
        if len(chosen_colors) == 3:
            break
    connection.close()
    distributions = []
    for frame in range(start, end+1):
        entries = by_frame.get(frame, [])
        row = dict(frame_index=frame, other_contexts=len(entries))
        for field in ('geometry', 'color_accuracy', 'combined'):
            values = [v[field] for v in entries]
            for label,q in [('min',0.), ('p10',.1), ('median',.5), ('p90',.9), ('max',1.)]:
                row[f'{field}_{label}'] = float(np.quantile(values,q)) if values else None
        distributions.append(row)
    rows = [details(step, name) for name,steps in branches.items() for step in steps]
    summaries = [summarize(name, [r for r in rows if r['branch']==name], steps, quarter)
                 for name,steps in branches.items()]
    for summary in summaries:
        if summary['branch']=='found':
            summary['solved_frame_index']=end
    local_rows = local_candidate_rows(m, gated, branches, gap)
    write_csv(out/'branch_frame_scores.csv', rows)
    write_csv(out/'other_context_scores.csv', context_rows)
    write_csv(out/'other_context_distributions.csv', distributions)
    write_csv(out/'reference_alignment_diagnostics.csv', reference_frames)
    write_csv(out/'branch_summary.csv', summaries)
    write_csv(out/'local_candidate_scores.csv', local_rows)
    report = dict(run=str(run.resolve()), frames=[start,end], zero_indexed=True,
        roi_padding=config['observation_input'].get('roi_padding'),
        reference_quarter_turns=quarter, reference_feasible=reference_feasible,
        reference_timing='inferred retrospectively: minimum total frame cost, monotonic supplied sequence, same move spacing and orientation rescue; not ground-truth frame annotations',
        score='0.5 * normalized_geometry + 0.5 * (1 - color_accuracy)',
        geometry='0.5*clip(keypoint_error/0.05,0,1) + 0.5*clip(sticker_position_penalty/0.05,0,1)',
        geometry_caveat='The scorer can replace an invalid/nonfinite keypoint metric with zero when sticker positions support a layer. Zero keypoint cost with zero reported keypoint inliers is fallback, not evidence of a perfect landmark fit.',
        original_search_priority='latest frame only; reported path means are diagnostics, not queue priority',
        other_context_count=len(context_rows),
        other_context_scope='retained best context per frame/physical-identity state/update history at the moment search stopped, excluding winner states; includes pending and expanded contexts, not discarded color-pruned trials',
        alternatives=alternative_metadata, summaries=summaries,
        branches=branches, distributions=distributions)
    (out/'score_analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('frames','reference_feasible','other_context_count','summaries')},indent=2))


if __name__ == '__main__':
    main()
