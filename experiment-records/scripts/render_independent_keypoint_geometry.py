#!/usr/bin/env python3
"""Refit cached landmark/sticker observations and render pose geometry only."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import render_yolo11n_independent_keypoints as renderer
from scripts.independent_keypoint_pose_geometry import process_pose_geometry
from scripts.render_independent_keypoint_search import candidate_for_step
from scripts.search_independent_keypoint_full_solve import initial_sticker_colors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--render-refined', action='store_true',
        help='Render an existing joint-fit log without fitting the poses again.')
    parser.add_argument('--joint-depth-fallback', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--geometry-sweep-device', choices=('cpu', 'mps', 'cuda'), default='cpu',
        help='Device for exhaustive sweeps, independent of the refinement device.')
    parser.add_argument('--geometry-refinement-device', choices=('cpu', 'mps'), default='cpu',
        help='Experimental batched Metal refinement; final scoring remains on CPU.')
    args = parser.parse_args()
    metadata = json.loads(args.log.with_suffix('.meta.json').read_text())
    source = args.source or Path(metadata['source'])
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
    colors = initial_sticker_colors(renderer, metadata['initial_scramble'])
    identity = renderer.CubeGeometryState.identity()
    cache = []
    for record in records:
        candidate = candidate_for_step(renderer, record, None, identity, colors)
        if args.render_refined and candidate is not None:
            bank = (record.get('search') or {}).get('geometry') or {}
            if not bank.get('joint_refined'):
                raise ValueError('--render-refined requires a joint-fit observation log')
            option = bank['options'][0]
            rotations = tuple(renderer.CubeLayerRotation(**r) for r in option['layer_rotations'])
            candidate = replace(candidate, rvec=np.asarray(option['pose_rvec']),
                tvec=np.asarray(option['pose_tvec']), layer_rotations=rotations,
                sticker_layer_rotations=rotations, joint_geometry_candidates=tuple(bank['options']),
                turn_search_diagnostics=record['search'])
        # These are previously thresholded object detections in full-frame
        # coordinates. Do not synthesize confidence scores or threshold again.
        cache.append(SimpleNamespace(frame_index=record['frame_index'],
            pose_candidate=candidate, roi_xyxy=None, roi_source='cached observations',
            pose_ready=True, local_points=np.empty((0, 2)), point_confidences=np.empty(0),
            class_ids=np.empty(0, int), box_confidences=np.empty(0)))
    config = renderer.build_parser().parse_args([
        '--source', str(source), '--initial-scramble', metadata['initial_scramble'],
        '--no-display', '--wireframe-only', '--no-show-fps',
    ])
    config.turning_start_frame = metadata.get('turning_start_frame')
    config.joint_depth_fallback = args.joint_depth_fallback
    config.geometry_sweep_device = args.geometry_sweep_device
    config.geometry_refinement_device = args.geometry_refinement_device
    fps = metadata.get('source_fps', metadata['fps'])
    last = 0.

    def progress(stage, completed, total):
        nonlocal last
        now = perf_counter()
        if completed == total or now-last >= 5:
            print(f'{stage}: {completed}/{total}', flush=True)
            last = now

    started = perf_counter()
    if not args.render_refined:
        process_pose_geometry(renderer, cache, config, colors, fps, progress)
    fit_seconds = perf_counter()-started
    with (output/'observations.jsonl').open('w') as stream:
        for record in cache:
            stream.write(json.dumps(renderer.turn_candidate_log_record(
                record.pose_candidate, record.frame_index), separators=(',', ':'))+'\n')

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f'Could not open {source}')
    width, height = (int(capture.get(prop)) for prop in
                     (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    video = output/'matty-wireframe.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
        '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    by_frame = {record.frame_index: record for record in cache}
    rendered_count = 0
    previews = {380, 418, 419, 420, 421, 422, 424, 425, 426, 469, 573, 574, 575, 576, 577, cache[-1].frame_index}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            record = by_frame.get(rendered_count)
            if record is None:
                raise ValueError(f'No cached observations at frame {rendered_count}')
            drawn, _, _ = renderer.render_frame(frame, None, None, None, config,
                cached_final_only=True, neural_data=record)
            encoder.stdin.write(drawn.tobytes())
            if rendered_count in previews:
                cv2.imwrite(str(output/f'frame{rendered_count}-wireframe.jpg'), drawn)
            rendered_count += 1
    finally:
        capture.release()
        encoder.stdin.close()
        returncode = encoder.wait()
    if returncode:
        raise RuntimeError(f'Video encoder exited {returncode}')
    if rendered_count != len(cache):
        raise ValueError(f'Rendered {rendered_count} frames for {len(cache)} observations')
    renderer.add_quicktime_frame_counter(video)
    metadata.update(fresh_inference=False, cached_detections_used=True,
        input_observations=str(args.log.resolve()), joint_pose_geometry=True,
        move_search_used=False, committed_moves=0, wireframe_only=True,
        orientation_center_color_weight=renderer.POSE_CONTINUITY_COLOR_WEIGHT,
        refinement_seconds=fit_seconds, rendered_frame_count=rendered_count,
        joint_outlier_policy='targeted_sparse_depth_recovery' if args.joint_depth_fallback else 'baseline',
        temporal_pose_seeds='previous_and_next_frame',
        geometry_shortlist_policy='exhaustive_sweep_and_refined_pose_families',
        reference_solution_used=False, video=str(video.resolve()))
    if not args.render_refined:
        metadata['geometry_sweep_device'] = args.geometry_sweep_device
        metadata['geometry_refinement_device'] = args.geometry_refinement_device
    if args.render_refined:
        metadata['refinement_seconds'] = json.loads(
            args.log.with_suffix('.meta.json').read_text()).get('refinement_seconds')
    (output/'observations.meta.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(video=str(video.resolve()), frames=rendered_count,
                         refinement_seconds=fit_seconds)), flush=True)


if __name__ == '__main__':
    import torch

    torch.set_num_threads(2)
    main()
