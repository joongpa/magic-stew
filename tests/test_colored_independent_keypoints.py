"""Categorical landmark/color pipeline regressions, with real cube projection."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys

import cv2
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "colored_independent_renderer",
    Path(__file__).resolve().parents[1] / "scripts/render_yolo11n_independent_keypoints.py",
)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


def cube():
    return m.StickerPnPCandidate(
        sticker_type="corner", sticker_average_confidence=.9,
        image_points=np.zeros((4, 2)), object_points=m.canonical_sticker_object_points("corner"),
        projected_sticker_points=np.zeros((4, 2)),
        camera_matrix=m.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1)), rvec=np.array([[.35], [-.4], [.12]]),
        tvec=np.array([[0.], [0.], [230.]]), reprojection_error=0.,
        image_size=(256, 256),
    )


def initial_colors(scramble="R U F L2 D B'"):
    state = m.CubeState.from_scramble(scramble)
    return {i.label: state.color_at((i.cubie_index, i.axis, i.side))
            for i in m.canonical_sticker_identities()}


def observed_cube(candidate=None, expected=None):
    candidate = cube() if candidate is None else candidate
    expected = initial_colors() if expected is None else expected
    faces = m.projected_sticker_faces(candidate, candidate.rvec, candidate.tvec)
    return m.attach_color_keypoints(candidate, np.array([p.mean(axis=0) for _, _, p in faces]),
                                   tuple(expected[i.label] for i, _, _ in faces))


def test_nearest_landmarks_are_separate_from_color_keypoints():
    points = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.], [100., 100.]])
    matches = m.sticker_keypoint_matches_for_colors(
        points, np.ones(5), np.array([0, 1, 2, 1, 0]), np.array([[5., 5.]]),
    )
    four = [match for match in matches if len(match.keypoint_indices) == 4]
    assert len(four) == 1
    assert set(four[0].keypoint_indices) == {0, 1, 2, 3}
    assert four[0].sticker_type == "corner"
    assert any(len(match.keypoint_indices) == 3 for match in matches)


def test_nearest_three_survive_invalid_four_and_keep_pnp_fallback(monkeypatch):
    points = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 20.]])
    matches = m.sticker_keypoint_matches_for_colors(
        points, np.ones(4), np.array([0, 1, 2, 0]), np.array([[7., 3.]]),
    )
    assert matches and all(len(match.keypoint_indices) == 3 for match in matches)
    called = []
    def solve(*args):
        called.append(args[2])
        return ()
    monkeypatch.setattr(m, "solve_sticker_pnp_candidates", solve)
    m.best_sticker_pnp_candidates(points, np.array([0, 1, 2, 0]), matches, (256, 256), 1.2,
                                 prepare_mask_geometry=False)
    assert called and len(called[0].keypoint_indices) == 3


def test_parallelogram_priority_is_preserved():
    square = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
    distorted = square.copy()
    distorted[2] += [8, 4]
    good = m.StickerKeypointMatch(0, (0, 1, 2, 3), "corner", .5, m.sticker_shape_error(square, range(4)))
    bad = m.StickerKeypointMatch(1, (0, 1, 2, 3), "corner", .99, m.sticker_shape_error(distorted, range(4)))
    assert m.sticker_match_priority_key(good) < m.sticker_match_priority_key(bad)


def test_prediction_metadata_remaps_color_ids():
    result = SimpleNamespace(
        keypoints=SimpleNamespace(xy=np.array([[[2., 3.]], [[4., 5.]]]), conf=np.ones((2, 1))),
        boxes=SimpleNamespace(cls=np.array([0, 1]), conf=np.ones(2)), names={0: "red", 1: "corner"},
    )
    _, _, classes, _ = m.pose_predictions(result)
    assert classes.tolist() == [m.CLASS_NAMES.index("red"), 0]


def test_color_thresholds_are_applied_before_candidates_and_scoring():
    valid = m.usable_pose_points(
        np.array([[1., 1.], [2., 2.], [np.nan, 0.], [0., 0.]]),
        np.array([1., 1., 1., 0.]), np.array([3, 6, 0, 0]),
        np.array([.5, .1, 1., 1.]), .05, m.DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES,
    )
    assert valid.tolist() == [True, False, False, False]


@pytest.mark.parametrize(("missing", "fps", "stride"), [
    (0, 60., 1), (0, 24., 1), (1, 60., 1), (1, 24., 1), (1, 60., 5), (1, 24., 2),
])
def test_cached_trajectory_recovers_missing_pose_from_current_detections(
    monkeypatch, missing, fps, stride,
):
    # Exercise recovery when the single-frame proposal stage has no result.
    monkeypatch.setattr(m, "_mixed_rigid_baseline", lambda *_: None)
    refit = m._refit_rigid_trajectory_pose
    angle_sigmas = []

    def capture_prior(obs, camera, center, rotation, seeds, sigma, angle_sigma, *rest):
        angle_sigmas.append(angle_sigma)
        return refit(obs, camera, center, rotation, seeds, sigma, angle_sigma, *rest)

    monkeypatch.setattr(m, "_refit_rigid_trajectory_pose", capture_prior)
    args = m.build_parser().parse_args(["--source", "clip.mp4", "--initial-scramble", ""])
    records = []
    positions = (0., 0., 0.) if missing == 0 else (-5., 0., 5.)
    for index, tx in enumerate(positions):
        pose = replace(cube(), tvec=np.array([[tx], [1.0], [230.0]]))
        objects, classes = m.visible_cube_lattice_points(pose.rvec, pose.tvec)
        landmarks = cv2.projectPoints(objects, pose.rvec, pose.tvec, pose.camera_matrix,
                                      pose.dist_coeffs)[0].reshape(-1, 2)
        stickers = [quad.mean(axis=0) for _, _, quad in
                    m.projected_sticker_faces(pose, pose.rvec, pose.tvec)]
        points = np.vstack((landmarks, stickers))
        confidences = np.ones(len(points))
        records.append(m.NeuralFrameData(
            100 + index * stride, np.zeros((256, 256, 3), dtype=np.uint8), (0, 0, 256, 256), "tracker", None,
            points, confidences, np.r_[classes, np.full(len(stickers), 4)], confidences,
            pose_candidate=None if index == missing else pose, pose_ready=True,
        ))
    anchors = [record for index, record in enumerate(records) if index != missing]
    endpoints = [record.pose_candidate for record in anchors]

    assert m.smooth_cached_rigid_poses(records, args, fps, {}) == 1
    seconds = stride / fps
    assert angle_sigmas == pytest.approx([(.17 + (1 + missing) * seconds) * np.sqrt(60 * seconds)])
    result = records[missing].pose_candidate
    np.testing.assert_allclose(result.tvec.ravel(), [0, 1, 230], atol=.5)
    assert result.layer_rotations == ()
    assert result.pose_continuity_diagnostics["anchor_frames"] == [record.frame_index for record in anchors]
    assert all(record.pose_candidate is pose for record, pose in zip(anchors, endpoints, strict=True))


def test_render_confidence_override_retains_other_sticker_thresholds():
    args = m.build_parser().parse_args([
        "--source", "clip.mp4", "--initial-scramble", "", "--pose-class-conf", "red=.8",
    ])
    assert args.pose_class_conf["red"] == .8
    assert args.pose_class_conf["yellow"] == .9


def test_rigid_fit_counts_duplicate_sticker_detections_as_one_area():
    pose = cube()
    _, _, quad = m.projected_sticker_faces(pose, pose.rvec, pose.tvec)[0]
    duplicate = np.tile(quad.mean(axis=0), (2, 1))
    observations = m.RigidPoseObservations(np.empty((0, 2)), np.empty(0, dtype=int),
                                           duplicate, ("red", "red"), 1.0)
    fit = m._match_rigid_pose(observations, pose.rvec, pose.tvec, pose)
    assert fit.sticker_count == 1


def test_color_matrix_is_exact_class_equality():
    expected = initial_colors()
    candidate = observed_cube(expected=expected)
    matrix = m.turn_color_match_matrix(candidate, expected_color_by_label=expected)
    assert matrix.shape == (54, len(candidate.sticker_color_names))
    assert set(np.unique(matrix)) <= {0., 1.}
    assert m.turn_color_match_matrix(candidate) is None


@pytest.mark.parametrize("device_name", ["cpu", "mps"])
def test_cpu_and_batched_gpu_score_duplicates_background_and_distinct_states(device_name):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    expected = initial_colors()
    candidate = observed_cube(expected=expected)
    # Two projected stickers are observed, one with a wrong and a correct
    # duplicate. A third color is far outside the cube and must be ignored.
    centers = np.array([candidate.sticker_mask_centers[0], candidate.sticker_mask_centers[0],
                        candidate.sticker_mask_centers[1], [-10000., -10000.]])
    correct = candidate.sticker_color_names[0]
    wrong = next(color for color in m.GLOBAL_COLOR_NAMES if color != correct)
    second_wrong = next(color for color in m.GLOBAL_COLOR_NAMES if color != candidate.sticker_color_names[1])
    candidate = m.attach_color_keypoints(candidate, centers, (wrong, correct, second_wrong, "red"))
    matrix = m.turn_color_match_matrix(candidate, expected_color_by_label=expected)
    cpu, total = m.sticker_mask_color_turn_fit_metrics(
        candidate, candidate.rvec, candidate.tvec, (), .001, matrix,
    )
    assert total == 2
    assert cpu == pytest.approx(.5)
    option = dict(is_baseline=True, layer_axis=0, layer_side=1, layer_count=1,
                  layer_rotations=(), label="static", combined_mean_error=0.,
                  assignment_projected_points=None, assignment_object_points=None,
                  assignment_projected_class_ids=None)
    states = (m.CubeGeometryState.identity(), m.CubeGeometryState.identity().apply_move("R"))
    batches = m._score_turn_color_offsets_for_states_on_gpu(
        candidate, (option,), .001, torch.device(device_name), matrix, states, color_offsets=(0.,),
    )
    assert batches is not None
    assert batches[0][0]["color_error"] == pytest.approx(cpu)
    assert batches[0][0]["color_match_count"] == total
    for state, batch in zip(states, batches, strict=True):
        c = replace(candidate, cube_geometry_state=state)
        error, count = m.sticker_mask_color_turn_fit_metrics(c, c.rvec, c.tvec, (), .001, matrix)
        assert batch[0]["color_error"] == pytest.approx(error, abs=1e-6)
        assert batch[0]["color_match_count"] == count


def test_no_containing_stickers_is_invalid_but_absent_colors_are_neutral():
    candidate = m.attach_color_keypoints(cube(), np.array([[-10000., -10000.]]), ("red",))
    matrix = m.turn_color_match_matrix(candidate, expected_color_by_label=initial_colors())
    assert m.sticker_mask_color_turn_fit_metrics(candidate, candidate.rvec, candidate.tvec, (), 10, matrix) == (float("inf"), 0)
    assert m.sticker_mask_color_turn_fit_metrics(candidate, candidate.rvec, candidate.tvec, (), 10, None) == (0., 0)


def test_orientation_uses_all_colors_even_without_centers_and_ignores_continuity():
    expected = initial_colors()
    candidate = observed_cube(expected=expected)
    faces = m.projected_sticker_faces(candidate, candidate.rvec, candidate.tvec)
    keep = [index for index, (identity, _, _) in enumerate(faces) if identity.category != "center"]
    candidate = m.attach_color_keypoints(candidate,
        np.array(candidate.sticker_mask_centers)[keep], tuple(candidate.sticker_color_names[i] for i in keep))
    rotated = m.transform_candidate_by_symmetry(candidate, m.CUBE_SYMMETRY_MATRICES[8])
    recovered = m.match_orientation_to_colors(rotated, expected)
    observations = m.projected_color_observations(recovered)
    assert observations
    assert all(expected[label] in colors for label, colors in observations.items())
    np.testing.assert_allclose(cv2.Rodrigues(recovered.rvec)[0], cv2.Rodrigues(candidate.rvec)[0], atol=1e-6)
    assert recovered.pose_continuity_diagnostics is None


def test_inference_calls_only_pose_model():
    calls = []
    class Pose:
        def predict(self, **kwargs):
            calls.append(kwargs)
            return ["pose-result"]
    args = m.build_parser().parse_args(["--source", "image.png", "--initial-scramble", ""])
    result = m.predict_pose_result(Pose(), np.zeros((32, 32, 3)), args)
    assert result == "pose-result"
    assert len(calls) == 1
    assert args.model == m.ROOT / "runs/pose/runs/26m-pose/weights/best.pt"


def test_diagonal_center_error_beats_compactness_and_confidence():
    compact = np.array([[0., 0.], [10., 0.], [13., 9.], [3., 9.]])
    stretched = np.array([[100., 0.], [150., 0.], [150., 3.], [100., 3.]])
    points = np.vstack((compact, stretched))
    matches = m.sticker_keypoint_matches_for_colors(
        points, np.array([.6] * 4 + [.99] * 4), np.array([0, 1, 2, 1] * 2),
        np.array([compact.mean(axis=0), stretched.mean(axis=0) + [2., 0.]]),
    )
    four = [match for match in matches if len(match.keypoint_indices) == 4]
    assert len(four) == 2
    assert min(four, key=m.sticker_match_priority_key).mask_index == 0
    assert [match.center_error for match in sorted(four, key=lambda match: match.mask_index)] == [0., 2.]
    assert m.sticker_shape_error(compact, range(4)) < m.sticker_shape_error(stretched, range(4))


def test_center_error_is_rms_of_both_diagonals_in_pixels_without_size_normalization():
    quad = np.array([[0., 0.], [10., 0.], [12., 10.], [0., 10.]])
    center = np.array([5., 5.])
    match, = m.sticker_center_alignment_matches(quad, np.array([0, 1, 2, 1]), np.ones(4),
                                               (0, 1, 2, 3), center, 0)
    scaled, = m.sticker_center_alignment_matches(quad * 10, np.array([0, 1, 2, 1]), np.ones(4),
                                                (0, 1, 2, 3), center * 10, 0)
    assert match.center_error == pytest.approx(np.sqrt(.5))
    assert scaled.center_error == pytest.approx(10 * match.center_error)


def test_nearest_invalid_cyclic_group_is_skipped_before_shortlisting():
    points = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.], [-4., 11.]])
    matches = m.sticker_keypoint_matches_for_colors(points, np.ones(5), np.array([0, 1, 1, 2, 1]),
                                                   np.array([[4., 4.]]))
    four = [match for match in matches if len(match.keypoint_indices) == 4]
    assert len(four) == 1
    assert set(four[0].keypoint_indices) == {0, 1, 3, 4}
    assert four[0].object_index_sequences


def test_three_point_completion_uses_color_center_instead_of_largest_gap(monkeypatch):
    points = np.array([[0., 0.], [30., 0.], [10., 10.]])
    matches = m.sticker_center_alignment_matches(points, np.full(3, 2), np.ones(3),
                                                (0, 1, 2), np.array([5., 5.]), 0)
    match, = matches
    assert match.center_error == 0.
    assert all((sequence[0] - sequence[2]) % 4 == 2 for sequence in match.object_index_sequences)
    measured = []
    def solve(object_points, image_points, *_args, **_kwargs):
        measured.append((object_points, image_points))
        return False, (), (), None
    monkeypatch.setattr(m.cv2, "solvePnPGeneric", solve)
    monkeypatch.setattr(m.cv2, "solveP3P", lambda *args, **kwargs: solve(*args, **kwargs)[:3])
    monkeypatch.setattr(m, "sticker_correspondence_object_indices", lambda *_: pytest.fail("completion was discarded"))
    m.solve_sticker_pnp_candidates(points, np.full(3, 2), match, (256, 256), 1.2)
    assert measured
    assert all(len(obj) == len(img) == 3 and np.array_equal(img, points) for obj, img in measured)


def test_three_point_roots_union_solvers_and_deduplicate_camera_transforms(monkeypatch):
    zero = np.zeros((3, 1))
    translation = np.array([[0.], [0.], [200.]])
    alternative = np.array([[.4], [.2], [.1]])
    other_translation = translation + [[1.], [2.], [3.]]
    monkeypatch.setattr(m.cv2, "solvePnPGeneric", lambda *_a, **_kw: (True, (zero,), (translation,), None))
    calls = []
    def perspective(*_a, flags):
        calls.append(flags)
        return True, (np.array([[2*np.pi], [0.], [0.]]), alternative), (translation + 1e-6, other_translation)
    monkeypatch.setattr(m.cv2, "solveP3P", perspective)
    roots = m.three_point_sticker_pose_roots(np.zeros((3, 3)), np.zeros((3, 2)), np.eye(3), np.zeros((4, 1)))
    assert calls == [cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_AP3P]
    assert len(roots) == 2
    assert np.array_equal(roots[0][1], translation)
    assert np.array_equal(roots[1][0], alternative)


@pytest.mark.parametrize("failed_solver", [cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_AP3P])
def test_one_three_point_solver_failure_does_not_discard_other_roots(monkeypatch, failed_solver):
    def solve(*_args, flags):
        if flags == failed_solver:
            raise cv2.error("degenerate solver branch")
        result = (True, (np.zeros((3, 1)),), (np.array([[0.], [0.], [200.]]),))
        return (*result, None) if flags == cv2.SOLVEPNP_SQPNP else result
    monkeypatch.setattr(m.cv2, "solvePnPGeneric", solve)
    monkeypatch.setattr(m.cv2, "solveP3P", solve)
    roots = m.three_point_sticker_pose_roots(np.zeros((3, 3)), np.zeros((3, 2)), np.eye(3), np.zeros((4, 1)))
    assert len(roots) == 1


def test_frame_579_retains_and_selects_correct_three_point_root_with_all_observations():
    fixture = json.loads((Path(__file__).parent / "fixtures/nahm_frame_579_three_point_pose.json").read_text())
    points = np.array(fixture["landmarks"], dtype=np.float32)
    classes = np.array(fixture["landmark_classes"], dtype=np.int32)
    colors = np.array(fixture["color_points"], dtype=np.float32)
    matches = m.sticker_keypoint_matches_for_colors(points, np.ones(len(points)), classes, colors)
    candidates = m.best_sticker_pnp_candidates(points, classes, matches, tuple(fixture["image_size"]),
                                              fixture["focal_scale"], scoring_points=points, prepare_mask_geometry=False)
    expected_seed = points[fixture["selected_landmark_indices"]]
    same_seed = [candidate for candidate in candidates if candidate.sticker_mask_index == fixture["selected_sticker"]
                 and len(candidate.image_points) == 3 and np.allclose(candidate.image_points, expected_seed)]
    assert len(same_seed) == 2
    turning_seed = min(same_seed, key=lambda candidate: float(np.linalg.norm(
        candidate.assignment_image_points - candidate.assignment_projected_points, axis=1,
    ).mean()))
    # A failed all-landmark proposal must preserve this valid mid-turn seed.
    assert m.refine_static_pose_from_all_observations(turning_seed) is turning_seed
    finals = tuple(m.solve_pnp_from_low_error_assignments(
        m.attach_color_keypoints(candidate, colors, tuple(fixture["color_names"])), recover_layer_turns=False,
    ) for candidate in candidates)
    previous = SimpleNamespace(tvec=np.array(fixture["previous_tvec"]))
    selected = m.select_pose_hypothesis(finals, None, (previous,))
    assert selected.sticker_mask_index == fixture["selected_sticker"]
    assert np.allclose(selected.image_points, expected_seed)
    assert len(selected.assignment_image_points) == len(points) == 23
    distances = np.linalg.norm(selected.assignment_image_points - selected.assignment_projected_points, axis=1)
    assert 20. < float(distances.mean()) < 23.
    # Outliers are allowed for fitting and remain present in the scored frame.
    assert np.any(distances > m.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * m.projected_cube_diameter(selected))
    assert m.pose_reprojection_fit_metric(selected) < .1
    assert np.linalg.norm(selected.tvec - previous.tvec) < 3.


def xuanyi_176_fixture():
    return json.loads((Path(__file__).parent / "fixtures/xuanyi_frame_176_static_pose.json").read_text())


def test_off_axis_cube_keeps_visible_top_face_when_normal_z_is_positive():
    fixture = xuanyi_176_fixture()
    rvec, tvec = np.array(fixture["reference_rvec"]), np.array(fixture["reference_tvec"])
    rotation = cv2.Rodrigues(rvec)[0]
    assert (rotation @ np.array([0., -1., 0.]))[2] > 0.
    visible, classes = m.visible_cube_lattice_points(rvec, tvec)
    expected = np.array(fixture["reference_objects"])
    # Actual rays see the front and top, even though optical-axis facing does not.
    assert len(visible) == 28
    for point, cls in zip(expected, fixture["landmark_classes"], strict=True):
        found = np.flatnonzero(np.all(np.isclose(visible, point, atol=1e-4), axis=1))
        assert len(found) == 1 and classes[found[0]] == cls


def test_frame_176_retained_front_stickers_recover_all_observations_without_expanding_shortlist():
    fixture = xuanyi_176_fixture()
    points = np.array(fixture["landmarks"], dtype=np.float32)
    classes = np.array(fixture["landmark_classes"], dtype=np.int32)
    colors = np.array(fixture["color_points"], dtype=np.float32)
    matches = m.sticker_keypoint_matches_for_colors(points, np.ones(len(points)), classes, colors)
    candidates = m.best_sticker_pnp_candidates(points, classes, matches, tuple(fixture["image_size"]),
                                              fixture["focal_scale"], scoring_points=points, prepare_mask_geometry=False)
    snapshots = [(c.rvec.copy(), c.tvec.copy(), c.assignment_projected_points.copy()) for c in candidates]
    finals = tuple(m.solve_pnp_from_low_error_assignments(
        m.attach_color_keypoints(candidate, colors, tuple(fixture["color_names"])), recover_layer_turns=False,
    ) for candidate in candidates)
    selected = m.select_pose_hypothesis(finals, None, (SimpleNamespace(tvec=np.array(fixture["previous_tvec"])),))
    # The top-sticker seed was fifth; this is recovered from the existing front seeds.
    assert selected.sticker_mask_index in (4, 9)
    assert len(selected.assignment_image_points) == len(points) == 24
    residuals = np.linalg.norm(selected.assignment_image_points - selected.assignment_projected_points, axis=1)
    assert float(residuals.mean()) < 1.2
    assert m.pose_reprojection_fit_metric(selected) < .02
    assert np.all(residuals <= m.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * m.projected_cube_diameter(selected))
    fresh = cv2.projectPoints(selected.assignment_object_points, selected.rvec, selected.tvec,
                             selected.camera_matrix, selected.dist_coeffs)[0].reshape(-1, 2)
    assert np.allclose(fresh, selected.assignment_projected_points, atol=1e-4)
    for candidate, (rvec, tvec, projected) in zip(candidates, snapshots, strict=True):
        assert np.array_equal(candidate.rvec, rvec)
        assert np.array_equal(candidate.tvec, tvec)
        assert np.array_equal(candidate.assignment_projected_points, projected)


def test_batched_off_axis_visibility_agrees_with_cpu_for_static_cube():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    fixture = xuanyi_176_fixture()
    points = np.array(fixture["landmarks"], dtype=np.float32)
    classes = np.array(fixture["landmark_classes"], dtype=np.int32)
    rvec, tvec = np.array(fixture["reference_rvec"]), np.array(fixture["reference_tvec"])
    camera = m.default_camera_matrix(tuple(fixture["image_size"]), fixture["focal_scale"])
    distortion = np.zeros((4, 1))
    anchor_indices = np.array([4, 5, 9, 1])
    anchors = m.canonical_sticker_object_points("center")
    _, rows, columns, projected, objects, projected_classes = m.evaluate_pose_hypothesis(
        rvec, tvec, points, classes, tuple(fixture["image_size"]), camera, distortion,
        fixed_observed_indices=anchor_indices, fixed_object_points=anchors,
    )
    c = m.StickerPnPCandidate(
        sticker_type="center", sticker_average_confidence=1., image_points=points[anchor_indices], object_points=anchors,
        projected_sticker_points=cv2.projectPoints(anchors, rvec, tvec, camera, distortion)[0].reshape(-1, 2),
        camera_matrix=camera, dist_coeffs=distortion, rvec=rvec, tvec=tvec, reprojection_error=0.,
        image_size=tuple(fixture["image_size"]), assignment_image_points=points[rows],
        assignment_projected_points=projected[columns], assignment_object_points=objects[columns],
        assignment_observed_class_ids=classes[rows], assignment_projected_class_ids=projected_classes[columns],
        assignment_fixed_mask=np.isin(rows, anchor_indices),
    )
    limit = m.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * m.projected_cube_diameter(c)
    trials = m._score_turn_configurations_on_gpu(c, limit, torch.device("cpu"), angle_values=np.array([0.]))
    # Some layer baselines have no observations because the bottom is cropped.
    baseline = [trial for trial in trials if trial["is_baseline"] and trial["keypoint_layer_inlier_count"] > 0]
    expected = np.linalg.norm(c.assignment_image_points - c.assignment_projected_points, axis=1).mean()
    assert baseline
    for trial in baseline:
        assert trial["keypoint_inlier_count"] == 24
        assert trial["keypoint_error"] * m.projected_cube_diameter(c) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("count", [3, 4])
def test_other_sticker_inside_real_or_completed_quad_rejects_seed(count):
    points = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])[:count]
    classes = np.array([0, 1, 2, 1])[:count]
    center = np.array([5., 5.])
    assert m.sticker_center_alignment_matches(points, classes, np.ones(count),
                                              tuple(range(count)), center, 0)
    # This point is outside the measured triangle but inside its completion.
    assert not m.sticker_center_alignment_matches(points, classes, np.ones(count),
                                                  tuple(range(count)), center, 0, np.array([[1., 8.]]))
    assert m.sticker_center_alignment_matches(points, classes, np.ones(count),
                                              tuple(range(count)), center, 0, np.array([[-1., 8.]]))


def test_pair_pruning_skips_invalid_combinations_before_full_validation(monkeypatch):
    points = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
    called = []
    original = m.sticker_center_alignment_matches
    def validate(*args):
        called.append((len(args[3]), args[5]))
        return original(*args)
    monkeypatch.setattr(m, "sticker_center_alignment_matches", validate)
    matches = m.sticker_keypoint_matches_for_colors(points, np.ones(4), np.array([0, 1, 2, 1]),
                                                   np.array([[5., 5.], [2., 4.]]))
    assert not any(len(match.keypoint_indices) == 4 for match in matches)
    assert (4, 0) not in called


def test_pair_pruning_agrees_with_convex_triangle_containment():
    rng = np.random.default_rng(42)
    points = rng.normal(size=(12, 2)) * 20
    others = rng.normal(size=(8, 2)) * 10
    center = np.array([2., 3.])
    blocked = m.sticker_blocked_landmark_pairs(points, center, others)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            triangle = np.asarray([center, points[i], points[j]], dtype=np.float32)
            expected = any(cv2.pointPolygonTest(triangle, tuple(point), False) >= 0 for point in others)
            assert blocked[i, j] == expected


def test_color_shortlist_ranks_three_and_four_points_by_center_and_deduplicates_groups(monkeypatch):
    matches = (
        m.StickerKeypointMatch(0, (0, 1, 2), "corner", .8, center_error=0.),
        m.StickerKeypointMatch(1, (0, 1, 2), "corner", .8, center_error=.1),
        m.StickerKeypointMatch(2, (3, 4, 5, 6), "corner", .4, center_error=1.),
        m.StickerKeypointMatch(3, (7, 8, 9), "edge", .9, center_error=1.),
        m.StickerKeypointMatch(4, (10, 11, 12, 13), "center", .99, center_error=2.),
    )
    called = []
    def solve(_points, _classes, match, *_args):
        called.append(match.mask_index)
        return (SimpleNamespace(sticker_average_confidence=match.average_confidence),)
    monkeypatch.setattr(m, "solve_sticker_pnp_candidates", solve)
    monkeypatch.setattr(m, "pose_hypothesis_visual_score", lambda _: 0.)
    m.best_sticker_pnp_candidates(np.zeros((14, 2)), np.zeros(14), matches, (256, 256), 1.2,
                                 prepare_mask_geometry=False)
    # Duplicate detections/correspondences consume one slot, and four wins the tie.
    assert called == [0, 2, 3]


def test_color_shortlist_skips_failed_pnp_groups(monkeypatch):
    matches = tuple(m.StickerKeypointMatch(i, (3*i, 3*i+1, 3*i+2), "center", .9, center_error=float(i))
                    for i in range(5))
    called = []
    def solve(_points, _classes, match, *_args):
        called.append(match.mask_index)
        return () if match.mask_index == 0 else (SimpleNamespace(sticker_average_confidence=.9),)
    monkeypatch.setattr(m, "solve_sticker_pnp_candidates", solve)
    monkeypatch.setattr(m, "pose_hypothesis_visual_score", lambda _: 0.)
    result = m.best_sticker_pnp_candidates(np.zeros((15, 2)), np.zeros(15), matches, (256, 256), 1.2,
                                          prepare_mask_geometry=False)
    assert len(result) == 3
    assert called == [0, 1, 2, 3]


def test_rhombus_score_is_invariant_to_image_scale_position_and_rotation():
    quad = np.array([[0., 0.], [10., 0.], [12., 9.], [3., 8.]])
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = 50 * quad @ rotation.T + np.array([812., -307.])
    assert m.sticker_shape_error(quad, range(4)) == pytest.approx(
        m.sticker_shape_error(transformed, range(4)), abs=1e-12)
    assert m.sticker_shape_error(quad, range(4)) == pytest.approx(
        m.sticker_shape_error(quad[::-1], range(4)), abs=1e-12)


@pytest.mark.parametrize("quad", [
    [[0., 0.], [1., 0.], [2., 0.], [3., 0.]],
    [[0., 0.], [1., 0.], [1., 0.], [0., 1.]],
    [[0., 0.], [4., 0.], [1., 1.], [0., 4.]],
    [[0., 0.], [1., 1.], [0., 1.], [1., 0.]],
    [[0., 0.], [1., 0.], [float("nan"), 1.], [0., 1.]],
])
def test_rhombus_score_rejects_degenerate_concave_crossed_and_nonfinite_groups(quad):
    assert m.sticker_shape_error(np.array(quad), range(4)) == float("inf")
