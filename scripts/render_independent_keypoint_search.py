#!/usr/bin/env python3
"""Render a saved full-search solution using its cached observations, without inference."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_search_replay", ROOT / "scripts/search_independent_keypoint_full_solve.py")
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


def static_assignments(m, candidate):
    """Recover class-aware display correspondences from the cached observations."""
    objects, classes = m.visible_cube_lattice_points(candidate.rvec, candidate.tvec)
    if not len(objects) or not len(candidate.assignment_image_points):
        return candidate
    projected, _ = cv2.projectPoints(objects, candidate.rvec, candidate.tvec,
                                    candidate.camera_matrix, candidate.dist_coeffs)
    projected = projected.reshape(-1, 2)
    _, rows, columns = m.observed_keypoint_reprojection_nearest_assignment(
        candidate.assignment_image_points, projected, candidate.image_size,
        observed_class_ids=candidate.assignment_observed_class_ids, projected_class_ids=classes)
    if len(rows) != len(candidate.assignment_image_points):
        return candidate
    return replace(candidate, assignment_object_points=objects[columns],
                   assignment_projected_points=projected[columns],
                   assignment_projected_class_ids=classes[columns])


def candidate_for_step(m, record, step, state, colors, *, after_solution=False):
    """Apply the saved trial to the pre-move physical state, never choosing a new move."""
    if record.get("has_pose") is False:
        return None
    pose = record.get("search_pose") or {}
    if pose.get("rvec") is None:
        pose = record.get("selected_pose") or {}
    if pose.get("rvec") is None:
        return None
    centers = {colors[i.label]: i.label for i in m.canonical_sticker_identities()
               if i.category == "center"}
    observed = np.asarray(pose.get("assignment_image_points") or [], dtype=float).reshape(-1, 2)
    candidate = m.StickerPnPCandidate(
        sticker_type=pose["sticker_type"], sticker_average_confidence=1.,
        image_points=np.empty((0, 2)), object_points=np.empty((0, 3)),
        projected_sticker_points=np.empty((0, 2)), reprojection_error=float("nan"),
        camera_matrix=np.asarray(pose["camera_matrix"]), dist_coeffs=np.asarray(pose["dist_coeffs"]),
        rvec=np.asarray(pose["rvec"]), tvec=np.asarray(pose["tvec"]),
        image_size=tuple(pose["image_size"]), assignment_image_points=observed,
        assignment_observed_class_ids=np.asarray(pose.get("assignment_observed_class_ids")
                                               or [-1] * len(observed)),
        sticker_mask_centers=tuple(np.asarray(p) for p in pose.get("sticker_mask_centers", [])),
        sticker_mask_center_uncertainties=tuple(pose.get("sticker_mask_center_uncertainties", [])),
        sticker_color_names=tuple(pose.get("sticker_color_names", [])),
        cube_geometry_state=state,
        turn_fit_search_rvec=np.asarray(pose["rvec"]).copy(),
        turn_fit_search_tvec=np.asarray(pose["tvec"]).copy(),
        orientation_axis_center_labels=tuple(centers[c] for c in m.ORIENTATION_AXIS_CENTER_COLORS),
    )
    candidate = static_assignments(m, candidate)
    if step and step.get('bridge_inferred'):
        rotations = ((m.CubeLayerRotation(step['layer_axis'], step['layer_side'],
                      step['color_offset'], step['layer_count']),) if step['state_moves'] else ())
        candidate = replace(candidate, turn_state_moves=tuple(step['state_moves']),
                            sticker_layer_rotations=rotations, turn_color_unreliable=True)
    elif step and "geometry_option_index" in step:
        # Stored rotations/moves already use physical coordinates after rescue.
        # The camera must receive the same symmetry before drawing that trial.
        trial = dict(step, layer_rotations=tuple(m.CubeLayerRotation(**r)
                                                for r in step["layer_rotations"]),
                     keypoint_error=step.get("keypoint_mean_error", float("inf")),
                     sticker_error=step.get("heatmap_mean_error", float("inf")),
                     combined_mean_error=step["state_score"])
        candidate = m._candidate_with_turn_trial(candidate, candidate, trial)
        base = static_assignments(m, replace(candidate, layer_rotations=(),
                                             sticker_layer_rotations=None))
        if candidate.layer_rotations and base.assignment_object_points is not None:
            # Materialize only the saved partial angle for diagnostic lines.
            # These display assignments do not alter its pose, state, or score.
            rotation = candidate.layer_rotations[0]
            options = m._score_turn_configurations_on_gpu(
                base, m.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * m.projected_cube_diameter(base),
                "cpu", angle_values=np.asarray([rotation.angle_degrees]),
                enforce_layer_observation_gate=False, enforce_validity_gate=False)
            chosen = next((o for o in options or [] if
                           (o["layer_axis"], o["layer_side"], o["layer_count"]) ==
                           (rotation.axis, rotation.side, rotation.layer_count)
                           and o["layer_rotations"] == candidate.layer_rotations), None)
            if chosen is not None:
                candidate = replace(candidate,
                    assignment_object_points=chosen["assignment_object_points"],
                    assignment_projected_points=chosen["assignment_projected_points"],
                    assignment_projected_class_ids=chosen["assignment_projected_class_ids"])
        else:
            candidate = replace(candidate, assignment_object_points=base.assignment_object_points,
                                assignment_projected_points=base.assignment_projected_points,
                                assignment_projected_class_ids=base.assignment_projected_class_ids)
    else:
        candidate = replace(candidate, turn_state_moves=(), turn_color_offset_degrees=0.)
        if after_solution:
            # The winning path ends at the goal. Freeze its state and use only
            # direct orientation matching for the remaining source frames.
            candidate = static_assignments(m, m.match_orientation_to_colors(candidate, colors))
    candidate = replace(candidate, orientation_anchor=m.orientation_anchor_for_candidate(candidate))
    candidate = replace(candidate, turn_color_matches=m.turn_color_match_matrix(
        candidate, expected_color_by_label=colors))
    if candidate.assignment_projected_points is not None:
        distances = np.linalg.norm(observed - candidate.assignment_projected_points, axis=1)
        candidate = replace(candidate, overall_reprojection_error=(float(distances.mean())
                                                                  if len(distances) else float("inf")))
    return candidate


def render_result(m, records, result, source, *, output_path=None, display=False,
                  show_keypoints=True, show_orientation_axes=True, show_virtual_cube=True,
                  turn_log=None):
    """Stream the source in order, replaying the winning path and its solved tail."""
    if result.get("status") not in {"solved", "already_solved"}:
        raise ValueError("Rendering requires a solved result with an accepted path")
    source = Path(source)
    if output_path is not None and Path(output_path).resolve() == source.resolve():
        raise ValueError("Output video must differ from the source video")
    if turn_log is not None and Path(turn_log).exists():
        raise ValueError("Use a fresh rendered turn log path")
    steps = {s["frame_index"]: s for s in result.get("steps", [])}
    if len(steps) != len(result.get("steps", [])):
        raise ValueError("Solution contains duplicate frame indices")
    config = result["config"]
    colors = replay.initial_sticker_colors(m, config["initial_scramble"])
    state = m.CubeGeometryState.identity()
    final_state = state
    for index in sorted(steps):
        final_state = final_state.apply_moves(tuple(steps[index].get("state_moves", ())))
    if not final_state.is_solved(colors):
        raise ValueError("Saved path does not solve the supplied scramble")
    accumulator = m.MoveAccumulator()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open source video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or config["fps"]
    writer = None
    frame_index = 0
    goal = result.get("solved_frame_index")
    if turn_log is not None:
        Path(turn_log).parent.mkdir(parents=True, exist_ok=True)
    completed = True
    last_reliable_candidate = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            record = records.get(frame_index)
            if record is None:
                raise ValueError(f"Missing cached observations for source frame {frame_index}")
            step = steps.get(frame_index)
            state_before = state
            bridged = bool(step and step.get('bridge_inferred'))
            tail = result["status"] == "already_solved" or goal is not None and frame_index > goal
            candidate = candidate_for_step(m, record, step, state, colors, after_solution=tail)
            if candidate is None and step and step.get("state_moves") and not bridged:
                raise ValueError(f"Move at frame {frame_index} has no cached pose")
            image = frame.copy()
            if candidate is not None and show_keypoints:
                points = np.concatenate((candidate.assignment_image_points,
                                         np.asarray(candidate.sticker_mask_centers).reshape(-1, 2)))
                classes = np.concatenate((candidate.assignment_observed_class_ids,
                    np.asarray([m.CLASS_NAMES.index(c) for c in candidate.sticker_color_names])))
                ones = np.ones(len(points))
                image, _ = m.draw_pose(image, points, ones, classes, ones, 0., {}, False, False)
            moves = tuple(step.get("state_moves", ())) if step else ()
            if bridged:
                accumulator.observe_unobserved_moves(moves, sticker_color_names=colors)
            else:
                accumulator.observe(candidate, sticker_color_names=colors)
            if candidate is not None and not bridged:
                image = m.draw_sticker_pnp_candidate(image, candidate, False,
                                                    show_orientation_axes)
                last_reliable_candidate = candidate
            if show_virtual_cube:
                virtual = candidate
                if bridged and virtual is None and last_reliable_candidate is not None:
                    rotations = ((m.CubeLayerRotation(step['layer_axis'], step['layer_side'],
                                  step['color_offset'], step['layer_count']),) if moves else ())
                    virtual = replace(last_reliable_candidate, cube_geometry_state=state,
                        layer_rotations=(), sticker_layer_rotations=rotations,
                        sticker_mask_centers=(), sticker_color_names=(), turn_color_matches=None)
                image = m.draw_classified_cube_diagnostic_panel(
                    image, virtual, colors,
                    rvec=last_reliable_candidate.rvec if bridged and last_reliable_candidate else None,
                    show_sticker_identities=False,
                    show_orientation_axes=show_orientation_axes)
            state = state.apply_moves(moves)
            if result["status"] == "already_solved" or goal is not None and frame_index >= goal:
                accumulator.solved_frame_index = goal if goal is not None else 0
            image = m.draw_move_accumulator_overlay(image, accumulator)
            if bridged:
                label = f"frame {frame_index} | unreliable observations bridged | inferred {' '.join(moves) or 'hold'}"
            elif step and "color_error" in step:
                total = step["color_match_count"]
                correct = step.get('color_correct_count', round((1.-step['color_error'])*total))
                label = f"frame {frame_index} | score {step['state_score']:.3f} | colors {correct}/{total}"
                if 'color_confusion_count' in step:
                    label += f" | allowed confusions {step['color_confusion_count']}"
            else:
                phase = ("solved state held" if tail else
                         (step or {}).get("observation_status", "outside search interval"))
                label = f"frame {frame_index} | {phase}"
            if step and 'sticker_mismatch_count' in step:
                label += f" | mismatches {step['sticker_mismatch_count']}/54"
            if step and 'sticker_history_penalty' in step:
                label += f" | history {step['sticker_history_penalty']:.3f}"
                priority = step.get('queue_priority', step.get('inherited_queue_priority'))
                if priority is not None:
                    label += f" | queue {priority:.3f}"
            if (step or {}).get('soft_window_penalty'):
                label += f" | old fork +{step['soft_window_penalty']:.2f}"
            if (step or {}).get('local_frame_rank') is not None:
                label += f" | local rank {step['local_frame_rank']}/{step['local_candidate_count']}"
                if step.get('outside_top_k_penalty'):
                    label += f" | outside top K +{step['outside_top_k_penalty']:.2f}"
            (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
            cv2.rectangle(image, (4, 46-height), (min(image.shape[1]-1, width+12), 50+baseline),
                          (0, 0, 0), -1)
            cv2.putText(image, label, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            if turn_log is not None:
                history_info = {key:step[key] for key in ('sticker_mismatch_count',
                    'sticker_history_qualified_count','sticker_mismatched_mask_hex',
                    'sticker_history_penalty','queue_priority','inherited_queue_priority',
                    'soft_window_penalty','soft_window_penalty_until_frame',
                    'supported_move_count','soft_window_protected_frame', 'color_scoring',
                    'color_correct_count','color_confusion_count','color_forbidden_count',
                    'color_confusion_cost', 'local_frame_rank', 'local_candidate_count',
                    'outside_top_k_penalty', 'layer_activity', 'previous_layer_activity',
                    'activity_evidence_status') if step and key in step}
                forced_static = (step or {}).get("observation_status") == "turning_start_frame"
                search_status = ('unreliable_frame_bridge' if bridged else
                                 "forced_static" if forced_static else
                                 "full_search_winning_path" if step else
                                 "solved_hold" if tail else "outside_search_interval")
                if candidate is None and bridged:
                    row = dict(record, selected_committed_moves=list(moves),
                        cube_geometry_state_before=state_before.to_json(),
                        search=dict(status=search_status, frame_quality=step['frame_quality'],
                                    geometry_score_used=False, color_score_used=False, **history_info))
                    with Path(turn_log).open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                else:
                    m.append_turn_candidate_log(Path(turn_log), replace(candidate,
                    turn_selected_trial_id=(step or {}).get("trial_id"),
                    turn_candidate_diagnostics=(dict(step),) if step and "trial_id" in step else (),
                    turn_search_diagnostics=dict(status=search_status,
                        turning_start_frame=config.get("turning_start_frame"),
                        frame_quality=(step or {}).get('frame_quality'),
                        geometry_score_used=not bridged, color_score_used=not bridged, **history_info),
                    turn_fit_evaluated=bool(step and "trial_id" in step),
                    ) if candidate is not None else None, frame_index)
            if output_path is not None:
                if writer is None:
                    writer = m.create_video_writer(Path(output_path), image, fps)
                writer.write(image)
            frame_index += 1
            if display:
                cv2.imshow("Full-search solution", image)
                if cv2.waitKey(max(1, round(1000. / fps))) & 0xff in (ord("q"), 27):
                    completed = False
                    break
            elif frame_index % 25 == 0:
                m.print_headless_progress("rendering winning path", frame_index, len(records))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyWindow("Full-search solution")
        else:
            print()
    if completed and any(index >= frame_index for index in steps):
        raise ValueError("Source video ends before the saved solution path")
    if writer is not None:
        m.add_quicktime_frame_counter(Path(output_path))
        print(f"Saved full-search solution: {output_path}", flush=True)
    return dict(source=str(source.resolve()), output=str(Path(output_path).resolve()) if output_path else None,
                frames=frame_index, fps=fps, completed=completed, path="saved_full_search_solution",
                model_inference_used=False, move_search_used=False, solved_frame_index=goal,
                virtual_cube_panel=show_virtual_cube,
                committed_moves=list(accumulator.moves))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="Saved result.json or run directory.")
    parser.add_argument("--source", type=Path, help="Override the original video path.")
    parser.add_argument("--log", type=Path, help="Override the observation log path.")
    parser.add_argument("--save-final-video", type=Path)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--show-keypoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-orientation-axes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-virtual-cube", action=argparse.BooleanOptionalAction, default=True,
                        help="Show the colored virtual cube beside the source video (default: on).")
    parser.add_argument("--turn-candidate-log", type=Path)
    args = parser.parse_args()
    if not args.save_final_video and not args.display:
        parser.error("Specify --save-final-video or --display")
    result_path = args.result / "result.json" if args.result.is_dir() else args.result
    result = json.loads(result_path.read_text())
    metadata = result["config"].get("observation_input", {})
    source = args.source or metadata.get("source")
    if not source:
        parser.error("The cached observations lack a source video; specify --source")
    log = args.log or Path(result["config"]["log"])
    records = {r["frame_index"]: r for r in
               (json.loads(line) for line in log.read_text().splitlines() if line.strip())}
    import torch
    torch.set_num_threads(2)
    try:
        report = render_result(replay.load_renderer(), records, result, source,
            output_path=args.save_final_video, display=args.display, show_keypoints=args.show_keypoints,
            show_orientation_axes=args.show_orientation_axes, show_virtual_cube=args.show_virtual_cube,
            turn_log=args.turn_candidate_log)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if args.save_final_video:
        args.save_final_video.with_suffix(".render.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
