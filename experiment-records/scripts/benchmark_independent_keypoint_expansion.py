"""Capture real expansions, then time identical work and verify every candidate.

The workload fixes direction, frame, state and residual context. Queue scheduling
and wall-clock time limits therefore cannot change the work between versions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

if __package__:
    from . import search_independent_keypoint_bidirectional as search
else:
    import search_independent_keypoint_bidirectional as search


def ordered(expand, node, depth):
    return sorted(expand(node, depth), key=lambda edge: (
        edge.step['state_score'], search.frame_search.unsupported_state_update(edge.step), edge.rank[-1]))


def fingerprint(edges):
    rows = [dict(state=edge.state.hex(), cue=edge.cue, rank=edge.rank,
                 step=dict(edge.step)) for edge in edges]
    encoded = json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def admitted(expand, node, depth, width, threshold):
    candidates = expand(node, depth)
    if isinstance(candidates, search.replay.fast.ExpansionBatch):
        edges, _ = candidates.select(width, threshold)
    else:
        edges = sorted(candidates, key=lambda edge: (
            edge.step['state_score'], search.frame_search.unsupported_state_update(edge.step), edge.rank[-1]))
        unrestricted = search.frame_search.unrestricted_sticker_branch([edge.step for edge in edges], threshold)
        moves_seen, states_seen, selected = set(), set(), []
        previous, admission_rank = None, 0
        for edge in edges:
            moves = search.frame_search.move_sequence_key(edge.step)
            if moves in moves_seen:
                continue
            key = edge.state, edge.cue
            if key in states_seen:
                continue
            moves_seen.add(moves); states_seen.add(key)
            order = edge.step['state_score'], search.frame_search.unsupported_state_update(edge.step), edge.rank[-1]
            if order != previous:
                admission_rank, previous = len(states_seen), order
            if not unrestricted and admission_rank > width:
                break
            selected.append(edge)
        edges = selected
    # Include the real admission work: concrete state and exportable diagnostics.
    return [(edge.state, edge.cue, dict(edge.step)) for edge in edges]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--workload', type=Path)
    parser.add_argument('--capture-expansions', type=int, default=2000)
    parser.add_argument('--sample-stride', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--baseline-source', type=Path)
    parser.add_argument('--compare-admission', type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    if args.baseline_source:
        mixin = sys.modules[search.replay.RotationBudgetScorerMixin.__module__]
        for module, filename in ((mixin, 'independent_keypoint_rotation_budget.py'),
                                 (search.replay, 'search_independent_keypoint_full_solve.py'),
                                 (search, 'search_independent_keypoint_bidirectional.py')):
            code = (args.baseline_source/filename).read_text()
            exec(compile(code, module.__file__, 'exec'), module.__dict__)
    cases = [] if args.workload is None else json.loads(args.workload.read_text())['cases']
    measurements = []
    admission_hashes = []
    original_join = search.widening_join

    if args.workload is None:
        def capture(forward, backward, *positional, **kwargs):
            calls = 0
            def wrap(expand, direction):
                def measured(node, depth):
                    nonlocal calls
                    calls += 1
                    edges = ordered(expand, node, depth)
                    if calls % args.sample_stride == 0:
                        cases.append(dict(direction=direction, depth=depth,
                            state=node.state.hex(), cue=node.cue, count=len(edges),
                            fingerprint=fingerprint(edges)))
                    return edges
                return measured
            return original_join(wrap(forward, 0), wrap(backward, 1), *positional, **kwargs)
        search.widening_join = capture
    else:
        original_search = search.bidirectional_search
        def fixed_search(scorer, *positional, **kwargs):
            def replay(forward, backward, *_positional, **_kwargs):
                # Check all fields in a separate pass, outside the timed loop.
                for case in cases:
                    node = search.Link(bytes.fromhex(case['state']), case['cue'])
                    edges = ordered((forward, backward)[case['direction']], node, case['depth'])
                    assert len(edges) == case['count']
                    assert fingerprint(edges) == case['fingerprint'], case
                    selected = admitted((forward, backward)[case['direction']], node, case['depth'],
                                        max(config['branch_widths']), config.get('unrestricted_sticker_count'))
                    serialized = json.dumps([(state.hex(), cue, step) for state, cue, step in selected],
                                            sort_keys=True, separators=(',', ':'), allow_nan=True)
                    admission_hashes.append(hashlib.sha256(serialized.encode()).hexdigest())
                if args.compare_admission:
                    expected = json.loads(args.compare_admission.read_text())['admission_fingerprints']
                    assert admission_hashes == expected, 'Admitted candidate records changed'
                for _ in range(args.repeats):
                    scorer.cache.clear()
                    if hasattr(scorer, 'row_cache'):
                        scorer.row_cache.clear()
                    start = time.perf_counter()
                    candidates = 0
                    for case in cases:
                        node = search.Link(bytes.fromhex(case['state']), case['cue'])
                        edges = admitted((forward, backward)[case['direction']], node, case['depth'],
                                         max(config['branch_widths']), config.get('unrestricted_sticker_count'))
                        candidates += len(edges)
                    elapsed = time.perf_counter() - start
                    measurements.append(dict(seconds=elapsed, expansions=len(cases),
                        candidates=candidates, expansions_per_second=len(cases)/elapsed))
                return None
            search.widening_join = replay
            kwargs['time_limit'] = float('inf')
            return original_search(scorer, *positional, **kwargs)
        search.bidirectional_search = fixed_search
    sys.argv = ['benchmark', '--log', config['log'], '--initial-scramble', config['initial_scramble'],
        '--end-frame', str(config['solved_anchor_frame']), '--branch-widths',
        *map(str, config['branch_widths']), '--max-tps', str(config['max_tps']),
        '--max-double-tps', str(config['max_double_tps']), '--search-seconds', '300',
        '--max-explored-states', str(args.capture_expansions),
        '--output-dir', str(args.output_dir/'search')]
    if config.get('unrestricted_sticker_count') is not None:
        sys.argv += ['--unrestricted-sticker-count', str(config['unrestricted_sticker_count'])]
    search.main()
    if args.workload is None:
        report = dict(config=str(args.config.resolve()), cases=cases)
        (args.output_dir/'workload.json').write_text(json.dumps(report)+'\n')
    else:
        report = dict(workload=str(args.workload.resolve()), candidate_equivalence=True,
            admission_fingerprints=admission_hashes,
            measurements=measurements,
            median_expansions_per_second=statistics.median(r['expansions_per_second'] for r in measurements))
        (args.output_dir/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases', 'admission_fingerprints')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
