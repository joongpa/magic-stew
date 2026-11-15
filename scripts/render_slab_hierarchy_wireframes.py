"""Render full solve windows: final reference versus causal slab-first tracker."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS
from scripts.benchmark_physics_pose_tracker import saved_candidate, bootstrap_fits, final_selected_fit
from scripts.independent_keypoint_slab_search import SlabHierarchyTracker
from scripts import independent_keypoint_pose_geometry as geometry
from scripts import render_yolo11n_independent_keypoints as renderer


def label(image, value, position, *, scale=.65, color=(240, 240, 240)):
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def crop_square(image, bounds, size=640):
    x, y, side = bounds
    patch = np.zeros((side, side, 3), np.uint8)
    height, width = image.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x+side), min(height, y+side)
    if x1 > x0 and y1 > y0:
        patch[y0-y:y1-y, x0-x:x1-x] = image[y0:y1, x0:x1]
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR)


def display_crop(records, steps):
    """Reference-derived framing only; never supplied to the pose tracker."""
    centers, diameters = [], []
    for record in records:
        step = steps.get(record['frame_index'])
        if step is None or saved_candidate(record) is None:
            continue
        payload = record.get('search_pose') or record['selected_pose']
        option = record['search']['geometry']['options'][step.get('geometry_option_index', 0)]
        tvec = np.asarray(step.get('pose_tvec', option['pose_tvec']), float)
        camera = np.asarray(payload['camera_matrix'], float)
        point = cv2.projectPoints(np.zeros((1, 3)), np.zeros(3), tvec, camera,
                                 np.asarray(payload['dist_coeffs']))[0].reshape(2)
        centers.append(point)
        diameter = renderer.projected_cube_diameter(saved_candidate(record), tvec=tvec)
        diameters.append(diameter)
    centers, diameters = np.asarray(centers), np.asarray(diameters)
    lower = np.percentile(centers-1.05*diameters[:, None], 1, axis=0)
    upper = np.percentile(centers+1.05*diameters[:, None], 99, axis=0)
    middle = (lower+upper)/2
    side = max(64, int(np.ceil(np.max(upper-lower))))
    return int(round(middle[0]-side/2)), int(round(middle[1]-side/2)), side


def draw_panel(frame, candidate, bounds, color):
    drawn = frame.copy()
    if candidate is not None:
        renderer.draw_projected_cube_pose(drawn, candidate, candidate.rvec, candidate.tvec,
            color, 2, outline=True, uniform_color=True)
    return crop_square(drawn, bounds)


def render_case(name, args):
    solution = json.loads((ROOT/REFERENCE_SOLUTIONS[name]).read_text())
    log = Path(solution['config']['log'])
    metadata = json.loads(log.with_suffix('.meta.json').read_text())
    source = Path(metadata['source'])
    source_fps = metadata.get('source_fps', metadata['fps'])
    start = metadata.get('turning_start_frame') or 0
    end = solution.get('solved_frame_index') or max(s['frame_index'] for s in solution['steps'])
    saved_case = None
    if args.report:
        saved_report = json.loads(args.report.read_text())
        saved_case = next(c for c in saved_report['cases'] if c['name'] == name)
        start, end = saved_case['bootstrap_frame'], saved_case['rollout_rows'][-1]['frame']
    saved_rows = {row['frame']:row for row in saved_case['rollout_rows']} if saved_case else {}
    steps = {s['frame_index']:s for s in solution['steps']}
    records = []
    for line in log.open():
        record = json.loads(line)
        if start <= record['frame_index'] <= end:
            records.append(record)
    by_frame = {r['frame_index']:r for r in records}
    bootstrap = None
    for record in records:
        if record['frame_index'] not in steps:
            continue
        if saved_candidate(record) is None:
            continue
        tagged = dict(record, _reference_step=steps[record['frame_index']])
        fits = bootstrap_fits(tagged)
        if len(geometry.diverse_shortlist(renderer, fits)) >= 5:
            bootstrap = record['frame_index']
            break
    if bootstrap is None:
        raise ValueError(f'{name}: no five-geometry bootstrap in solve window')
    all_axes = bool(saved_report['parameters'].get('slab_all_axes', False)) if saved_case else args.slab_all_axes
    tracker = SlabHierarchyTracker(renderer, beam=args.static_beam, iterations=args.iterations, all_axes=all_axes)
    tracker.initialize(fits, bootstrap/source_fps)
    bounds = display_crop(records, steps)
    stride = max(1, int(np.ceil(source_fps/args.max_update_fps-.01)))
    frames = list(range(bootstrap, end+1, stride))
    if saved_case:
        frames = [bootstrap, *sorted(saved_rows)]
    if frames[-1] != end:
        frames.append(end)
    directory = args.output_dir/name
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/f'{name}-wireframe-comparison.mp4'
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f'Cannot open {source}')
    capture.set(cv2.CAP_PROP_POS_FRAMES, bootstrap)
    source_position = bootstrap-1
    encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', str(source_fps/stride), '-i', '-',
        '-an', '-vf', 'fps=30', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-threads', '2',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    samples, previews = [], []
    report_every = perf_counter()
    preview_indices = {0, len(frames)//3, 2*len(frames)//3, len(frames)-1}
    previous = tracker.branches[0].fit.candidate
    try:
        with (directory/'tracking.jsonl').open('w') as stream:
            for index, frame_index in enumerate(frames):
                frame = None
                while source_position < frame_index:
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f'{name}: cannot decode source frame {frame_index}')
                    source_position += 1
                record = by_frame.get(frame_index)
                observations = saved_candidate(record) if record else None
                fit_ms = 0.
                status = 'bootstrap' if index == 0 else 'missing observations - held'
                accepted = False
                fits = [branch.fit for branch in tracker.branches]
                if index and saved_case:
                    row = saved_rows[frame_index]
                    fits = [geometry._materialize(renderer, observations, fit['axis'],
                        np.r_[np.asarray(fit['rvec']).ravel(), np.asarray(fit['tvec']).ravel(),
                              np.asarray(fit['angles'])[[0, 2]]]) for fit in row['pose_bank']]
                    previous = fits[0].candidate
                    accepted = row['motion_prediction']['final_full_fit_accepted']
                    fit_ms = row['milliseconds']
                    status = 'full fit passed' if accepted else 'full fit FAILED'
                    np.testing.assert_allclose(renderer.pose_reprojection_fit_metric(previous), row['reprojection_fit'], atol=1e-10)
                elif index and observations is not None and len(observations.assignment_image_points) >= 3 and len(observations.sticker_mask_centers):
                    started = perf_counter()
                    fits = tracker.update(observations, frame_index/source_fps)
                    fit_ms = 1000.*(perf_counter()-started)
                    previous = fits[0].candidate
                    accepted = tracker.last_prediction_diagnostics['final_full_fit_accepted']
                    status = 'full fit passed' if accepted else 'full fit FAILED'
                candidate = previous
                error = renderer.pose_reprojection_fit_metric(candidate)
                reference = None
                if record and frame_index in steps and observations is not None:
                    reference = final_selected_fit(dict(record, _reference_step=steps[frame_index])).candidate
                output = np.zeros((720, 1280, 3), np.uint8)
                reference_color = (255, 220, 60)
                fit_color = (90, 240, 90) if accepted else ((220, 220, 220) if index == 0 else (0, 165, 255))
                output[40:680, :640] = draw_panel(frame, reference, bounds, reference_color)
                output[40:680, 640:] = draw_panel(frame, candidate, bounds, fit_color)
                mode = 'BENCHMARK WINDOW' if saved_case else 'FULL SOLVE'
                label(output, f'{name.upper()}  |  {mode}  |  frame {frame_index}  |  {frame_index/source_fps:.2f}s', (18, 28), scale=.7)
                label(output, 'Final selected reference' if reference is not None else 'Reference unavailable this frame', (18, 705), color=reference_color)
                label(output, f'Slab-first: {status} | error {error:.3f}', (655, 705), scale=.6, color=fit_color)
                cv2.line(output, (640, 40), (640, 680), (150, 150, 150), 1)
                encoder.stdin.write(output.tobytes())
                sample = dict(frame=frame_index, status=status, full_fit_passed=bool(accepted), reprojection_error=error,
                    fitting_ms=fit_ms, bank_size=len(fits), reference_available=reference is not None,
                    pose_bank=[dict(axis=f.axis, angles=list(f.angles), rvec=f.candidate.rvec.tolist(),
                                    tvec=f.candidate.tvec.tolist(), score=f.score) for f in fits])
                samples.append(sample)
                stream.write(json.dumps(sample)+'\n')
                if index in preview_indices:
                    preview = directory/f'frame-{frame_index}.jpg'
                    cv2.imwrite(str(preview), output)
                    previews.append(preview)
                if perf_counter()-report_every >= 8.:
                    print(f'{name}: rendered {index+1}/{len(frames)}', flush=True)
                    report_every = perf_counter()
    finally:
        capture.release()
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f'{name}: encoder exited {code}')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate:format=duration', '-of', 'json', str(path)]))
    if int(probe['streams'][0]['nb_read_frames']) < len(frames)-2:
        raise ValueError(f'{name}: encoded frame count is unexpectedly short')
    result = dict(name=name, source=str(source), observations=str(log), reference_solution=REFERENCE_SOLUTIONS[name],
        video=str(path.resolve()), bootstrap_frame=bootstrap, requested_solve_start=metadata.get('turning_start_frame') or 0,
        clip_start_frame=bootstrap, clip_end_frame=end, solved_frame=solution.get('solved_frame_index'),
        render_scope='saved benchmark window' if saved_case else 'full solve window',
        source_fps=source_fps, input_render_fps=source_fps/stride, encoded_fps=30, rendered_samples=len(samples),
        source_frame_indices=frames, crop=list(bounds), crop_uses_reference_for_display_only=True,
        reference_used_by_tracker='Initial bootstrap only; subsequent references only drawn after update.',
        initial_reference_bootstrap=True, post_solve_frames_included=False,
        all_articulation_axes_searched=all_axes, static_beam=args.static_beam, fitting_iterations=args.iterations,
        replayed_report=str(args.report.resolve()) if args.report else None,
        solve_end_frame=solution.get('solved_frame_index'),
        min_retained_geometries=min(s['bank_size'] for s in samples), full_fit_passed_frames=sum(s['full_fit_passed'] for s in samples),
        preview_images=[str(p.resolve()) for p in previews], mean_fitting_ms=float(np.mean([s['fitting_ms'] for s in samples if s['fitting_ms']])),
        codec_probe=probe)
    (directory/'metadata.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=list(REFERENCE_SOLUTIONS), default=list(REFERENCE_SOLUTIONS))
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--static-beam', type=int, default=8)
    parser.add_argument('--slab-all-axes', action='store_true')
    parser.add_argument('--max-update-fps', type=float, default=30.)
    parser.add_argument('--resume', action='store_true', help='Reuse completed videos with metadata in the existing output directory')
    parser.add_argument('--report', type=Path, help='Replay exact recorded benchmark windows instead of fitting full solve windows')
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=args.resume, parents=True)
    results = []
    for name in args.cases:
        metadata = args.output_dir/name/'metadata.json'
        if args.resume and metadata.exists():
            result = json.loads(metadata.read_text())
            expected_axes = (json.loads(args.report.read_text())['parameters'].get('slab_all_axes', False)
                             if args.report else args.slab_all_axes)
            if bool(result.get('all_articulation_axes_searched', False)) != expected_axes:
                raise ValueError('Cannot reuse a video rendered with a different axis-search mode')
            subprocess.run(['ffmpeg', '-v', 'error', '-i', result['video'], '-f', 'null', '-'], check=True)
            results.append(result)
            print(f'{name}: reused completed video', flush=True)
        else:
            results.append(render_case(name, args))
        (args.output_dir/'manifest.json').write_text(json.dumps(dict(cases=results,
            tracker_source_sha256=hashlib.sha256((ROOT/'scripts/independent_keypoint_slab_search.py').read_bytes()).hexdigest()), indent=2)+'\n')
    listing = args.output_dir/'concat.txt'
    listing.write_text(''.join("file '"+r['video'].replace("'", "'\\''")+"'\n" for r in results))
    overview = args.output_dir/'all-five-wireframes.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'concat', '-safe', '0', '-i', str(listing),
        '-c', 'copy', '-movflags', '+faststart', str(overview)], check=True)
    tiles = [cv2.resize(cv2.imread(r['preview_images'][1]), (640, 360)) for r in results]
    cv2.imwrite(str(args.output_dir/'contact-sheet.jpg'), np.vstack(tiles))
    print(f'Overview: {overview.resolve()}', flush=True)


if __name__ == '__main__':
    import torch
    torch.set_num_threads(2)
    main()
