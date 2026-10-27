import importlib.util
from itertools import permutations
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "render_yolo11n_independent_keypoints.py"
)
SPEC = importlib.util.spec_from_file_location(
    "render_yolo11n_independent_keypoints",
    SCRIPT_PATH,
)
render_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = render_script
SPEC.loader.exec_module(render_script)


def test_sticker_type_patterns_and_three_point_fallbacks():
    classify = render_script.sticker_type_from_keypoint_classes

    assert classify([0, 1, 1, 2]) == "corner"
    assert classify([1, 1, 2, 2]) == "edge"
    assert classify([2, 2, 2, 2]) == "center"
    assert classify([0, 1, 2]) == "corner"
    assert classify([1, 1, 2]) == "edge"
    assert classify([1, 2, 2]) == "edge"
    assert classify([2, 2, 2]) == "center"
    assert classify([0, 2, 2]) == "unknown"


def test_cube_symmetries_are_the_24_proper_rotations():
    matrices = render_script.CUBE_SYMMETRY_MATRICES

    assert len(matrices) == 24
    assert len({tuple(matrix.reshape(-1)) for matrix in matrices}) == 24
    assert all(np.isclose(np.linalg.det(matrix), 1.0) for matrix in matrices)
    np.testing.assert_array_equal(matrices[0], np.eye(3))


def test_symmetry_transformation_preserves_projected_cube_geometry():
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    symmetry = render_script.CUBE_SYMMETRY_MATRICES[7]
    points = render_script.CUBE_LATTICE_OBJECT_POINTS_MM
    transformed_points = render_script.transform_object_points_by_symmetry(
        points,
        symmetry,
    )
    transformed_rvec = render_script.transform_rvec_by_symmetry(rvec, symmetry)
    original_projection, _ = cv2.projectPoints(
        points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    transformed_projection, _ = cv2.projectPoints(
        transformed_points,
        transformed_rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    np.testing.assert_allclose(
        transformed_projection,
        original_projection,
        atol=1e-4,
    )


def test_symmetry_tracker_selects_equivalent_pose_near_previous_frame():
    object_points = render_script.canonical_sticker_object_points("corner")
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = render_script.solve_sticker_pnp(
        image_points.reshape(-1, 2),
        np.asarray([0, 1, 2, 1], dtype=np.int32),
        match,
        (256, 256),
        1.2,
    )
    assert candidate is not None
    target_rvec = render_script.transform_rvec_by_symmetry(
        candidate.rvec,
        render_script.CUBE_SYMMETRY_MATRICES[9],
    )
    tracker = render_script.CubeSymmetryTracker()
    tracker.observe(target_rvec)

    canonical = render_script.canonicalize_pose_symmetry(candidate, tracker)

    assert render_script.rotation_symmetry_distance(canonical.rvec, target_rvec) < 1e-6


def test_static_refinement_skips_layer_turn_recovery(monkeypatch):
    object_points = render_script.canonical_sticker_object_points("corner")
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    image_points, _ = cv2.projectPoints(
        object_points,
        np.zeros((3, 1), dtype=np.float64),
        np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
    )
    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = render_script.solve_sticker_pnp(
        image_points.reshape(-1, 2),
        np.asarray([0, 1, 2, 1], dtype=np.int32),
        match,
        (256, 256),
        1.2,
    )
    assert candidate is not None

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("static refinement must not recover layer turns")

    monkeypatch.setattr(
        render_script,
        "recover_misaligned_cube_layers",
        fail_if_called,
    )
    refined = render_script.solve_pnp_from_low_error_assignments(candidate)

    assert refined.layer_rotations == ()
    assert refined.turn_candidate_scores == ()
    assert refined.final_solve_image_points is not None
    assert len(refined.final_solve_image_points) >= 3


def test_project_cube_orientation_axes_follow_camera_projection():
    candidate = SimpleNamespace(
        camera_matrix=render_script.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    projected = render_script.project_cube_orientation_axes(
        candidate,
        np.zeros((3, 1), dtype=np.float64),
        np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
        axis_length=24.0,
    )

    assert projected is not None
    assert projected.shape == (4, 2)
    assert projected[1, 0] > projected[0, 0]  # +X points right
    assert projected[2, 1] > projected[0, 1]  # +Y points down in the image
    np.testing.assert_allclose(projected[3], projected[0], atol=1e-5)


def test_canonical_sticker_identities_cover_54_stickers_in_intuitive_groups():
    identities = render_script.canonical_sticker_identities()

    assert len(identities) == 54
    assert len({identity.label for identity in identities}) == 54
    assert {identity.category for identity in identities} == {"center", "edge", "corner"}
    assert sum(identity.category == "center" for identity in identities) == 6
    assert sum(identity.category == "edge" for identity in identities) == 24
    assert sum(identity.category == "corner" for identity in identities) == 24
    assert [identity.label for identity in identities[:6]] == [
        "C01 U", "C02 D", "C03 F", "C04 B", "C05 R", "C06 L",
    ]


def test_center_color_classifier_chooses_a_legal_standard_scheme():
    trace = render_script.StickerColorTrace.create()
    expected = {
        "U": "white",
        "D": "yellow",
        "F": "green",
        "B": "blue",
        "R": "red",
        "L": "orange",
    }
    labels_by_face = {
        identity.face: identity.label
        for identity in trace.identities
        if identity.category == "center"
    }
    observations = {
        labels_by_face[face]: np.asarray(render_script.PURE_COLOR_RGB[color], dtype=np.uint8)
        for face, color in expected.items()
    }
    trace.observe(0, observations)
    trace.observe(1, observations)

    classified = render_script.classify_center_color_names(trace)

    assert {
        next(identity.face for identity in trace.identities if identity.label == label): color
        for label, color in classified.items()
    } == expected
    assert len(render_script.legal_center_color_assignments()) == 24


def test_oklab_conversion_round_trips_rgb_values():
    rgb = np.asarray(tuple(render_script.PURE_COLOR_RGB.values()), dtype=np.float64)

    np.testing.assert_allclose(
        render_script.oklab_to_rgb(render_script.rgb_to_oklab(rgb)),
        rgb,
        atol=0.01,
    )


def test_piecewise_profiles_compare_trajectories_in_oklab():
    trace = render_script.StickerColorTrace.create()
    center_labels = {
        identity.face: identity.label
        for identity in trace.identities
        if identity.category == "center"
    }
    red_values = np.asarray(
        [[64, 8, 8], [128, 16, 16], [220, 40, 40], [255, 110, 110]],
        dtype=np.uint8,
    )
    blue_values = np.asarray(
        [[8, 8, 64], [16, 16, 128], [40, 40, 220], [110, 110, 255]],
        dtype=np.uint8,
    )
    for frame_index, (red, blue) in enumerate(zip(red_values, blue_values, strict=True)):
        trace.observe(
            frame_index,
            {
                center_labels["R"]: red,
                center_labels["B"]: blue,
            },
        )

    red_profile = render_script.fit_sticker_color_profile(trace, center_labels["R"])
    blue_profile = render_script.fit_sticker_color_profile(trace, center_labels["B"])

    assert red_profile is not None
    assert blue_profile is not None
    assert render_script.sticker_color_profile_distance(red_profile, red_profile) < 1e-8
    assert render_script.sticker_color_profile_distance(red_profile, blue_profile) > 0.02


def test_sticker_assignment_rejects_opposite_colors_on_one_edge():
    trace = render_script.StickerColorTrace.create()
    center_labels = {
        identity.face: identity.label
        for identity in trace.identities
        if identity.category == "center"
    }
    centers = {
        face: np.asarray(render_script.PURE_COLOR_RGB[color], dtype=np.uint8)
        for face, color in {
            "U": "white",
            "D": "yellow",
            "F": "green",
            "B": "blue",
            "R": "red",
            "L": "orange",
        }.items()
    }
    edge_identities = [
        identity
        for identity in trace.identities
        if identity.category == "edge" and identity.cubie_label == "UF"
    ]
    observations = {
        center_labels[face]: value for face, value in centers.items()
    }
    observations[edge_identities[0].label] = np.asarray(
        render_script.PURE_COLOR_RGB["red"], dtype=np.uint8
    )
    observations[edge_identities[1].label] = np.asarray(
        render_script.PURE_COLOR_RGB["orange"], dtype=np.uint8
    )
    trace.observe(0, observations)

    center_colors = render_script.classify_center_color_names(trace)
    classified = render_script.classify_sticker_color_names(trace, center_colors)
    assigned_edge_colors = tuple(classified[label] for label in observations if label.startswith("E"))

    assert len(assigned_edge_colors) == 2
    assert render_script._legal_cubie_color_tuple(assigned_edge_colors)
    assert frozenset(assigned_edge_colors) not in render_script.OPPOSITE_COLOR_PAIRS


def test_sticker_assignment_does_not_reuse_the_same_edge_piece():
    trace = render_script.StickerColorTrace.create()
    center_labels = {
        identity.face: identity.label
        for identity in trace.identities
        if identity.category == "center"
    }
    center_colors = {
        "U": "white",
        "D": "yellow",
        "F": "green",
        "B": "blue",
        "R": "red",
        "L": "orange",
    }
    observations = {
        center_labels[face]: np.asarray(render_script.PURE_COLOR_RGB[color], dtype=np.uint8)
        for face, color in center_colors.items()
    }
    edge_groups = {}
    for identity in trace.identities:
        if identity.category == "edge":
            edge_groups.setdefault(identity.cubie_label, []).append(identity)
    for group_name in ("UF", "UR"):
        for identity in edge_groups[group_name]:
            observations[identity.label] = np.asarray(
                render_script.PURE_COLOR_RGB["red" if identity.face == "U" else "orange"],
                dtype=np.uint8,
            )
    trace.observe(0, observations)

    classified = render_script.classify_sticker_color_names(
        trace,
        render_script.classify_center_color_names(trace),
    )
    assigned_pairs = []
    for group_name in ("UF", "UR"):
        assigned_pairs.append(
            frozenset(classified[identity.label] for identity in edge_groups[group_name])
        )

    assert assigned_pairs[0] != assigned_pairs[1]
    assert len(render_script.legal_edge_color_combinations()) == 12
    assert len(render_script.legal_corner_color_combinations()) == 8


def test_corner_assignment_enforces_proper_color_chirality():
    identities = [
        identity
        for identity in render_script.canonical_sticker_identities()
        if identity.category == "corner" and identity.cubie_label == "UBL"
    ]
    identities.sort(key=lambda identity: {"U": 0, "L": 1, "B": 2}[identity.face])

    # On the UBL corner, white-orange-blue is the clockwise standard order.
    assert render_script.corner_color_assignment_is_chiral(
        identities,
        ("white", "orange", "blue"),
    )
    assert not render_script.corner_color_assignment_is_chiral(
        identities,
        ("white", "blue", "orange"),
    )

    # Exactly half of the permutations of a legal color triple are proper
    # rotations; the other half are reflected/impossible corner assignments.
    assignments = [
        render_script.corner_color_assignment_is_chiral(
            identities,
            permutation,
        )
        for permutation in permutations(("white", "orange", "blue"))
    ]
    assert sum(assignments) == 3


def test_corner_chirality_uses_rotated_center_color_scheme():
    identities = [
        identity
        for identity in render_script.canonical_sticker_identities()
        if identity.category == "corner" and identity.cubie_label == "UBL"
    ]
    identities.sort(key=lambda identity: {"U": 0, "L": 1, "B": 2}[identity.face])

    # This is a valid cube rotation.  Chirality must follow the centers rather
    # than assuming that a particular object face is always a fixed color.
    rotated_face_colors = render_script.legal_center_color_assignments()[1]
    rotated_color_face = {
        color: face
        for face, color in rotated_face_colors.items()
    }
    ordered_colors = tuple(rotated_face_colors[identity.face] for identity in identities)
    assert render_script.corner_color_assignment_is_chiral(
        identities,
        ordered_colors,
        rotated_color_face,
    )


def test_projected_center_color_overlay_only_draws_visible_classified_centers():
    candidate = SimpleNamespace(
        layer_rotations=(),
        camera_matrix=render_script.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    drawn = render_script.draw_projected_center_colors(
        image,
        candidate,
        {"C02 D": "yellow", "C01 U": "white"},
        np.zeros((3, 1), dtype=np.float64),
        np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
    )

    assert drawn == 1
    assert np.any(image[:, :, 1] > 0)


def test_classified_cube_diagnostic_panel_keeps_cube_stationary():
    common = {
        "layer_rotations": (),
        "rvec": np.zeros((3, 1), dtype=np.float64),
        "camera_matrix": render_script.default_camera_matrix((256, 256), 1.2),
        "dist_coeffs": np.zeros((4, 1), dtype=np.float64),
    }
    first_candidate = SimpleNamespace(
        **common,
        tvec=np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
    )
    second_candidate = SimpleNamespace(
        **common,
        tvec=np.asarray([[40.0], [-25.0], [320.0]], dtype=np.float64),
    )
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    colors = {"C01 U": "white", "C02 D": "yellow"}

    first = render_script.draw_classified_cube_diagnostic_panel(
        image,
        first_candidate,
        colors,
    )
    second = render_script.draw_classified_cube_diagnostic_panel(
        image,
        second_candidate,
        colors,
    )

    assert first.shape == (256, 512, 3)
    np.testing.assert_array_equal(first[:, 256:], second[:, 256:])


def test_projected_sticker_faces_match_cube_pose_geometry_after_symmetry():
    symmetry = render_script.CUBE_SYMMETRY_MATRICES[7]
    candidate = SimpleNamespace(
        layer_rotations=(),
        object_symmetry=symmetry,
        camera_matrix=render_script.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    rvec = render_script.transform_rvec_by_symmetry(
        np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64),
        symmetry,
    )
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)

    sticker_faces = render_script.projected_sticker_faces(candidate, rvec, tvec)
    cube_faces = render_script.projected_cubelet_faces(candidate, rvec, tvec)

    assert len(sticker_faces) == len(cube_faces)
    for (_identity, sticker_depth, sticker_polygon), (
        cube_depth,
        cube_polygon,
        _anchor,
    ) in zip(sticker_faces, cube_faces, strict=True):
        np.testing.assert_allclose(sticker_depth, cube_depth)
        np.testing.assert_allclose(sticker_polygon, cube_polygon)


def test_sticker_identity_overlay_draws_only_visible_stickers():
    candidate = SimpleNamespace(
        layer_rotations=(),
        camera_matrix=render_script.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    drawn = render_script.draw_projected_sticker_identity_labels(
        image,
        candidate,
        np.zeros((3, 1), dtype=np.float64),
        np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
    )

    assert drawn == 9
    assert np.any(image)


def test_sticker_color_sampling_uses_projected_mask_and_cube_sticker_intersection():
    image = np.full((256, 256, 3), (60, 80, 100), dtype=np.uint8)
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    candidate = SimpleNamespace(
        layer_rotations=(),
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
    )
    sticker_mask = np.ones((256, 256), dtype=bool)
    cube_mask = np.ones((256, 256), dtype=bool)
    predicted = ([sticker_mask], {0: "sticker"}, np.asarray([0], dtype=np.int32), cube_mask)

    observations = render_script.sample_projected_sticker_colors(
        image,
        (0, 0, 256, 256),
        predicted,
        candidate,
        max_samples_per_sticker=32,
        min_samples_per_sticker=4,
    )

    assert observations
    assert set(observations) == {
        "C02 D",
        "E09 DF/D",
        "E11 DR/D",
        "E13 DB/D",
        "E15 DL/D",
        "K13 DFR/D",
        "K16 DFL/D",
        "K19 DBL/D",
        "K22 DBR/D",
    }
    assert all(np.array_equal(value, np.asarray((100, 80, 60), dtype=np.uint8)) for value in observations.values())

    empty_cube = np.zeros_like(cube_mask)
    empty_predicted = (predicted[0], predicted[1], predicted[2], empty_cube)
    assert render_script.sample_projected_sticker_colors(
        image,
        (0, 0, 256, 256),
        empty_predicted,
        candidate,
    ) == {}


def test_sticker_rgb_html_plot_has_selectors_for_all_stickers(tmp_path):
    pytest.importorskip("plotly.graph_objects")
    trace = render_script.StickerColorTrace.create()
    trace.observe(
        7,
        {trace.identities[0].label: np.asarray((10, 20, 30), dtype=np.uint8)},
    )
    output_path = tmp_path / "sticker_rgb.html"

    render_script.save_sticker_rgb_html_plot(
        trace,
        output_path,
        title="Sticker RGB",
    )

    html = output_path.read_text(encoding="utf-8")
    assert "All stickers" in html
    assert "Centers" in html
    assert "Edges" in html
    assert "Corners" in html
    assert "fitted regression" in html
    assert '"mode":"lines"' in html
    assert '"groupclick":"toggleitem"' in html
    assert all(
        identity.label.replace("/", "\\u002f") in html
        for identity in trace.identities
    )


def test_sticker_keypoint_matches_returns_a_consistent_circular_order():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    points = np.asarray(
        [
            (8, 8),
            (23, 8),
            (23, 23),
            (8, 23),
        ],
        dtype=np.float32,
    )
    point_confidences = np.ones(4, dtype=np.float32)
    class_ids = np.asarray([0, 1, 2, 1], dtype=np.int32)
    box_confidences = np.ones(4, dtype=np.float32)

    indices, sticker_type = render_script.sticker_keypoint_matches(
        mask,
        points,
        point_confidences,
        class_ids,
        box_confidences,
        point_threshold=0.05,
        class_confidences=dict(render_script.DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES),
    )

    assert indices == [0, 1, 2, 3]
    assert sticker_type == "corner"


def test_sticker_selection_prefers_parallelogram_like_geometry():
    points = np.asarray(
        [
            (8.0, 8.0),
            (24.0, 9.0),
            (23.0, 24.0),
            (7.0, 23.0),
        ],
        dtype=np.float32,
    )
    distorted = points.copy()
    distorted[2] += np.asarray((8.0, 7.0))

    good_error = render_script.sticker_shape_error(points, [0, 1, 2, 3])
    bad_error = render_script.sticker_shape_error(distorted, [0, 1, 2, 3])
    good_match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.60,
        shape_error=good_error,
    )
    bad_match = render_script.StickerKeypointMatch(
        mask_index=1,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.99,
        shape_error=bad_error,
    )

    assert good_error < bad_error
    assert render_script.sticker_match_priority_key(good_match) < render_script.sticker_match_priority_key(bad_match)


def test_render_sticker_masks_uses_inferred_sticker_type_colors():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    local_mask = np.zeros((32, 32), dtype=bool)
    local_mask[8:24, 8:24] = True
    predicted = ([local_mask], {0: "sticker"}, np.asarray([0], dtype=np.int32))
    pose = (
        np.asarray([(8, 8), (23, 8), (23, 23), (8, 23)], dtype=np.float32),
        np.ones(4, dtype=np.float32),
        np.asarray([0, 1, 2, 1], dtype=np.int32),
        np.ones(4, dtype=np.float32),
    )

    output = render_script.render_sticker_masks(
        image,
        (0, 0, 32, 32),
        predicted,
        opacity=1.0,
        border_thickness=0,
        pose_predictions=pose,
        point_threshold=0.05,
        class_confidences=dict(render_script.DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES),
    )

    assert np.all(output[12, 12] == render_script.STICKER_TYPE_COLORS_BGR["corner"])


def test_sticker_mask_erosion_parser_and_roi_masks():
    assert render_script.build_parser().parse_args(["--source", "frame.png"]).sticker_mask_erosion == 0
    args = render_script.build_parser().parse_args(
        ["--source", "frame.png", "--sticker-mask-erosion", "1"]
    )
    assert args.sticker_mask_erosion == 1

    local_mask = np.zeros((9, 9), dtype=bool)
    local_mask[2:7, 2:7] = True
    predicted = (
        [local_mask],
        {0: "sticker"},
        np.asarray([0], dtype=np.int32),
        np.ones((9, 9), dtype=bool),
    )
    masks = render_script.sticker_segmentation_masks_in_roi(
        predicted,
        (9, 9),
        sticker_mask_erosion=1,
    )

    assert len(masks) == 1
    assert int(masks[0].sum()) == 9


def test_sticker_mask_parser_preserves_the_model_cube_mask():
    cube_mask = np.zeros((16, 16), dtype=np.float32)
    cube_mask[2:14, 2:14] = 1.0
    sticker_mask = np.zeros((16, 16), dtype=np.float32)
    sticker_mask[5:11, 5:11] = 1.0
    result = SimpleNamespace(
        masks=SimpleNamespace(data=np.asarray([cube_mask, sticker_mask])),
        boxes=SimpleNamespace(cls=np.asarray([0, 1], dtype=np.int32)),
        names={0: "cube", 1: "sticker"},
    )

    parsed = render_script.sticker_masks_from_result(result)

    assert parsed is not None
    sticker_masks, _names, sticker_classes, parsed_cube_mask = parsed
    assert len(sticker_masks) == 1
    np.testing.assert_array_equal(sticker_classes, np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(parsed_cube_mask, cube_mask.astype(bool))


def test_solve_sticker_pnp_uses_canonical_corner_correspondence():
    object_points = render_script.canonical_sticker_object_points("corner")
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = render_script.solve_sticker_pnp(
        image_points.reshape(-1, 2),
        np.asarray([0, 1, 2, 1], dtype=np.int32),
        match,
        (256, 256),
        1.2,
    )

    assert candidate is not None
    assert candidate.reprojection_error < 1e-3
    assert candidate.tvec[2, 0] > 0.0
    assert candidate.assignment_image_points is not None
    assert candidate.assignment_projected_points is not None
    assert len(candidate.assignment_image_points) == 4
    assert len(candidate.assignment_projected_points) == 4


def test_projected_cube_diameter_uses_the_initial_pose_scale():
    image_size = (256, 256)
    camera_matrix = render_script.default_camera_matrix(image_size, 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    candidate = SimpleNamespace(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvec=rvec,
        tvec=tvec,
    )

    projected_cube, _ = cv2.projectPoints(
        render_script.CUBE_CORNER_POINTS_MM,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    _center, radius = cv2.minEnclosingCircle(
        projected_cube.reshape(-1, 2).astype(np.float32)
    )

    assert render_script.projected_cube_diameter(candidate) == pytest.approx(
        2.0 * radius
    )


def test_cube_layer_rotation_changes_only_the_selected_face_layer():
    points = render_script.CUBE_CORNER_POINTS_MM
    rotation = render_script.CubeLayerRotation(axis=2, side=1, angle_degrees=90.0)
    rotated = render_script.apply_cube_layer_rotations(points, (rotation,))
    mask = render_script.cube_layer_mask(points, axis=2, side=1)

    np.testing.assert_allclose(rotated[~mask], points[~mask])
    assert not np.allclose(rotated[mask], points[mask])
    np.testing.assert_allclose(
        rotated[mask][:, 2],
        np.full(int(mask.sum()), render_script.CUBE_HALF_SIZE_MM),
    )


def test_single_layer_mask_leaves_the_shared_seam_stationary():
    points = render_script.CUBE_LATTICE_OBJECT_POINTS_MM
    seam_coordinate = render_script.CUBE_HALF_SIZE_MM - render_script.CUBE_LAYER_SIZE_MM

    positive_mask = render_script.cube_layer_mask(points, axis=2, side=1)
    negative_mask = render_script.cube_layer_mask(points, axis=2, side=-1)

    assert not np.any(positive_mask[np.isclose(points[:, 2], seam_coordinate)])
    assert not np.any(negative_mask[np.isclose(points[:, 2], -seam_coordinate)])


def test_orientation_anchor_uses_stationary_complement_of_turn_layer():
    candidate = SimpleNamespace(
        layer_rotations=(
            render_script.CubeLayerRotation(axis=2, side=1, angle_degrees=30.0),
        )
    )

    anchor = render_script.orientation_anchor_for_candidate(candidate)

    assert anchor == render_script.OrientationAnchor(axis=2, layers=(0, 1))


def test_orientation_anchor_is_whole_cube_without_a_turn():
    candidate = SimpleNamespace(layer_rotations=())

    assert render_script.orientation_anchor_for_candidate(candidate) == (
        render_script.all_cube_orientation_anchor()
    )


def test_projected_cube_faces_mark_only_anchor_layers():
    image_size = (256, 256)
    candidate = SimpleNamespace(
        layer_rotations=(),
        orientation_anchor=render_script.OrientationAnchor(axis=2, layers=(0, 1)),
        camera_matrix=render_script.default_camera_matrix(image_size, 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)

    faces = render_script.projected_cubelet_faces(candidate, rvec, tvec)

    assert faces
    assert any(not is_anchor for _depth, _face, is_anchor in faces)
    assert any(is_anchor for _depth, _face, is_anchor in faces)


def test_temporal_pose_filter_refits_from_stationary_layers():
    image_size = (256, 256)
    camera_matrix = render_script.default_camera_matrix(image_size, 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    rotation = render_script.CubeLayerRotation(axis=2, side=1, angle_degrees=30.0)
    assignment_object_points = render_script.apply_cube_layer_rotations(
        render_script.CUBE_LATTICE_OBJECT_POINTS_MM,
        (rotation,),
    )
    assignment_image_points, _ = cv2.projectPoints(
        assignment_object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    sticker_object_points = render_script.canonical_sticker_object_points("corner")
    sticker_image_points, _ = cv2.projectPoints(
        sticker_object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    candidate = render_script.StickerPnPCandidate(
        sticker_type="corner",
        sticker_average_confidence=0.9,
        image_points=sticker_image_points.reshape(-1, 2),
        object_points=sticker_object_points,
        projected_sticker_points=sticker_image_points.reshape(-1, 2),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        reprojection_error=0.0,
        assignment_image_points=assignment_image_points.reshape(-1, 2),
        assignment_projected_points=assignment_image_points.reshape(-1, 2),
        assignment_object_points=assignment_object_points,
        layer_rotations=(rotation,),
    )

    filtered = render_script.apply_temporal_pose_filter(
        candidate,
        render_script.PoseMotionFilter(),
    )

    assert filtered.orientation_anchor == render_script.OrientationAnchor(
        axis=2,
        layers=(0, 1),
    )
    assert filtered.motion_diagnostics is not None
    assert filtered.raw_rvec is not None
    assert filtered.assignment_projected_points is not None


def test_rank_misaligned_layer_faces_prefers_fewest_explained_observations():
    points = render_script.CUBE_LATTICE_OBJECT_POINTS_MM
    distances = np.zeros(len(points), dtype=np.float32)
    positive_z = render_script.cube_layer_mask(points, axis=2, side=1)
    distances[positive_z] = 12.0

    ranked = render_script.rank_misaligned_layer_faces(
        points,
        points,
        distances,
        fixed_mask=np.zeros(len(points), dtype=bool),
        assignment_error_limit=4.0,
        max_face_candidates=2,
    )

    assert ranked[0][:2] == (2, 1)
    assert ranked[0][2] == 0.0
    assert len(ranked) == 2


def test_transformed_visibility_rejects_occluded_lattice_points():
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64)
    back_center_index = render_script.CUBE_LATTICE_COORDS.index((1, 1, 3))

    unturned_points = render_script.CUBE_LATTICE_OBJECT_POINTS_MM
    unturned_hidden = render_script.transformed_surface_occlusion_mask(
        unturned_points,
        rvec,
        tvec,
        (),
    )
    assert unturned_hidden[back_center_index]

    rvec = np.asarray([[0.5], [0.3], [0.0]], dtype=np.float64)
    rotation = render_script.CubeLayerRotation(axis=0, side=-1, angle_degrees=30.0)
    turned_points = render_script.apply_cube_layer_rotations(
        unturned_points,
        (rotation,),
    )
    turned_hidden = render_script.transformed_surface_occlusion_mask(
        turned_points,
        rvec,
        tvec,
        (rotation,),
    )
    visible_points, _class_ids = render_script.visible_transformed_cube_lattice_points(
        rvec,
        tvec,
        (rotation,),
    )

    transformed_normals = render_script.CUBE_LATTICE_BOUNDARY_NORMALS.copy()
    layer_mask = render_script.cube_layer_mask(
        unturned_points,
        rotation.axis,
        rotation.side,
        rotation.layer_count,
    )
    layer_rvec = np.zeros((3, 1), dtype=np.float64)
    layer_rvec[rotation.axis, 0] = rotation.side * np.deg2rad(
        rotation.angle_degrees
    )
    layer_matrix, _ = cv2.Rodrigues(layer_rvec)
    transformed_normals[layer_mask] = np.einsum(
        "ij,nfj->nfi",
        layer_matrix,
        transformed_normals[layer_mask],
    )
    camera_matrix, _ = cv2.Rodrigues(rvec)
    front_facing = np.any(
        render_script.CUBE_LATTICE_BOUNDARY_FACE_MASK
        & (
            np.einsum(
                "ij,nfj->nfi",
                camera_matrix,
                transformed_normals,
            )[..., 2]
            < 0.0
        ),
        axis=1,
    )

    assert np.any(turned_hidden & front_facing)
    assert len(visible_points) < len(turned_points)


def test_recover_misaligned_layer_updates_assignment_geometry():
    image_size = (256, 256)
    camera_matrix = render_script.default_camera_matrix(image_size, 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    sticker_object_points = render_script.canonical_sticker_object_points("corner")
    sticker_image_points, _ = cv2.projectPoints(
        sticker_object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    sticker_image_points = sticker_image_points.reshape(-1, 2)

    visible_objects, visible_classes = render_script.visible_cube_lattice_points(
        rvec,
        tvec,
    )
    layer_choice = None
    for axis in range(3):
        for side in (-1, 1):
            mask = render_script.cube_layer_mask(visible_objects, axis, side)
            if int(mask.sum()) >= 3 and not np.any(
                np.all(
                    np.isclose(
                        visible_objects[mask][:, None, :],
                        sticker_object_points[None, :, :],
                        atol=1e-4,
                    ),
                    axis=2,
                )
            ):
                layer_choice = (axis, side, mask)
                break
        if layer_choice is not None:
            break
    assert layer_choice is not None
    axis, side, layer_mask = layer_choice
    rotation = render_script.CubeLayerRotation(axis, side, 45.0)
    rotated_objects = render_script.apply_cube_layer_rotations(
        visible_objects,
        (rotation,),
    )
    layer_indices = np.flatnonzero(layer_mask)
    source_index = int(
        layer_indices[
            np.argmax(
                np.linalg.norm(
                    rotated_objects[layer_indices] - visible_objects[layer_indices],
                    axis=1,
                )
            )
        ]
    )
    extra_point, _ = cv2.projectPoints(
        rotated_objects[source_index : source_index + 1],
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    all_points = np.concatenate((sticker_image_points, extra_point.reshape(-1, 2)))
    all_classes = np.concatenate(
        (
            np.asarray([0, 1, 2, 1], dtype=np.int32),
            np.asarray([visible_classes[source_index]], dtype=np.int32),
        )
    )
    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = render_script.solve_sticker_pnp(
        all_points,
        all_classes,
        match,
        image_size,
        1.2,
        scoring_points=all_points,
    )
    assert candidate is not None
    limit = (
        render_script.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION
        * render_script.projected_cube_diameter(candidate)
    )

    recovered = render_script.recover_misaligned_cube_layers(candidate, limit)
    exhaustive = render_script.recover_misaligned_cube_layers(
        candidate,
        limit,
        max_face_candidates=6,
    )
    shortlisted = render_script.recover_misaligned_cube_layers(
        candidate,
        limit,
        max_face_candidates=2,
    )
    logged = render_script.recover_misaligned_cube_layers(
        candidate,
        limit,
        max_face_candidates=6,
        record_all_turn_candidates=True,
    )

    assert recovered.layer_rotations
    assert recovered.assignment_projected_points is not None
    recovered_distances = np.linalg.norm(
        recovered.assignment_image_points
        - recovered.assignment_projected_points,
        axis=1,
    )
    assert np.min(recovered_distances) <= limit
    assert shortlisted.layer_rotations == exhaustive.layer_rotations
    np.testing.assert_allclose(
        shortlisted.assignment_projected_points,
        exhaustive.assignment_projected_points,
    )
    assert len(logged.turn_candidate_scores) > render_script.PNP_TURN_DEBUG_CANDIDATE_COUNT
    assert len(logged.turn_candidate_debug) == len(logged.turn_candidate_scores)


def test_three_point_edge_uses_image_gap_to_preserve_cyclic_correspondence():
    object_points = render_script.canonical_sticker_object_points("edge")
    # p0, p1, p2 are observed; p3 is the missing corner.  The p2 -> p0
    # segment is therefore the long cyclic gap in the ordered image points.
    image_points = np.asarray(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        dtype=np.float32,
    )

    sequences = render_script.sticker_correspondence_object_indices(
        "edge",
        [1, 1, 2],
        image_points,
    )

    assert sequences == ((0, 1, 2),)
    np.testing.assert_allclose(object_points[list(sequences[0])], object_points[:3])


def test_three_point_pnp_uses_segmentation_to_choose_the_matching_pose_branch():
    object_points = render_script.canonical_sticker_object_points("corner")
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    full_image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    full_image_points = full_image_points.reshape(-1, 2).astype(np.float32)

    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = render_script.solve_sticker_pnp(
        full_image_points[[2, 3, 0]],
        np.asarray([2, 1, 0], dtype=np.int32),
        match,
        (256, 256),
        1.2,
        sticker_contour=full_image_points,
    )

    assert candidate is not None
    np.testing.assert_allclose(
        candidate.projected_sticker_points,
        full_image_points[[2, 3, 0]],
        atol=1e-3,
    )
    np.testing.assert_allclose(candidate.tvec, tvec, atol=1e-2)


def test_best_sticker_prefers_four_points_over_higher_confidence_three_points(monkeypatch):
    matches = (
        render_script.StickerKeypointMatch(0, (0, 1, 2), "corner", 0.999),
        render_script.StickerKeypointMatch(1, (0, 1, 2, 3), "edge", 0.900),
    )
    selected = []

    def fake_solve(_points, _classes, match, _image_size, _focal_scale, *_args):
        selected.append(match)
        return SimpleNamespace(
            match=match,
            overall_reprojection_error=1.0,
            reprojection_error=0.0,
            sticker_average_confidence=match.average_confidence,
            assignment_image_points=None,
            assignment_projected_points=None,
            assignment_object_points=None,
        )

    monkeypatch.setattr(render_script, "solve_sticker_pnp", fake_solve)
    result = render_script.best_sticker_pnp_candidate(
        np.zeros((4, 2), dtype=np.float32),
        np.zeros((4,), dtype=np.int32),
        matches,
        (128, 128),
        1.2,
    )

    assert result.match is matches[1]
    assert selected == [matches[1]]


def test_best_sticker_tries_all_four_point_stickers_and_chooses_lowest_overall_error(
    monkeypatch,
):
    matches = (
        render_script.StickerKeypointMatch(0, (0, 1, 2, 3), "corner", 0.99),
        render_script.StickerKeypointMatch(1, (4, 5, 6, 7), "edge", 0.90),
        render_script.StickerKeypointMatch(2, (8, 9, 10, 11), "center", 0.80),
        render_script.StickerKeypointMatch(3, (12, 13, 14, 15), "corner", 0.70),
        render_script.StickerKeypointMatch(4, (8, 9, 10), "center", 1.00),
    )
    selected = []

    def fake_solve(_points, _classes, match, _image_size, _focal_scale, *_args):
        selected.append(match.mask_index)
        return SimpleNamespace(
            match=match,
            overall_reprojection_error={0: 12.0, 1: 4.0, 2: 8.0}[match.mask_index],
            reprojection_error=0.0,
            sticker_average_confidence=match.average_confidence,
            assignment_image_points=None,
            assignment_projected_points=None,
            assignment_object_points=None,
        )

    monkeypatch.setattr(render_script, "solve_sticker_pnp", fake_solve)
    result = render_script.best_sticker_pnp_candidate(
        np.zeros((16, 2), dtype=np.float32),
        np.zeros((16,), dtype=np.int32),
        matches,
        (128, 128),
        1.2,
    )

    assert result.match is matches[1]
    assert selected == [0, 1, 2]


def test_best_sticker_candidates_retains_ranked_alternatives(monkeypatch):
    matches = tuple(
        render_script.StickerKeypointMatch(
            index,
            (index * 4, index * 4 + 1, index * 4 + 2, index * 4 + 3),
            "corner",
            0.9,
        )
        for index in range(3)
    )

    def fake_solve(_points, _classes, match, _image_size, _focal_scale, *_args):
        return SimpleNamespace(
            match=match,
            overall_reprojection_error=float(3 - match.mask_index),
            reprojection_error=0.0,
            sticker_average_confidence=match.average_confidence,
        )

    monkeypatch.setattr(render_script, "solve_sticker_pnp", fake_solve)
    candidates = render_script.best_sticker_pnp_candidates(
        np.zeros((12, 2), dtype=np.float32),
        np.zeros((12,), dtype=np.int32),
        matches,
        (128, 128),
        1.2,
    )

    assert [candidate.match.mask_index for candidate in candidates] == [2, 1, 0]


def test_pose_hypothesis_buffer_keeps_current_set_separate_from_history():
    buffer = render_script.PoseHypothesisBuffer(max_frames=2)
    first = SimpleNamespace(rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)))
    second = SimpleNamespace(rvec=np.ones((3, 1)), tvec=np.ones((3, 1)))
    third = SimpleNamespace(rvec=np.full((3, 1), 2.0), tvec=np.full((3, 1), 2.0))

    buffer.append((first,), 10)
    buffer.append((second,), 11)
    assert buffer.previous_candidates() == (first,)
    buffer.append((third,), 12)

    assert [frame.frame_index for frame in buffer.frames] == [11, 12]
    assert buffer.previous_candidates() == (second,)


def test_pose_hypothesis_buffer_exposes_only_best_candidate_for_fast_graph():
    buffer = render_script.PoseHypothesisBuffer(max_frames=2)
    first = SimpleNamespace()
    alternate = SimpleNamespace()
    second = SimpleNamespace()
    buffer.append((first, alternate), 10)
    buffer.append((second,), 11)

    fast_frames = buffer.best_candidate_frames()

    assert [frame.candidates for frame in fast_frames] == [(first,), (second,)]


def test_hypothesis_selection_prefers_continuity_only_for_close_visual_scores():
    def candidate(error, angle):
        return SimpleNamespace(
            overall_reprojection_error=error,
            reprojection_error=0.0,
            rvec=np.asarray([[0.0], [0.0], [angle]], dtype=np.float64),
            tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
        )

    near = candidate(10.0, 0.01)
    far = candidate(10.5, 1.0)
    selected = render_script.select_pose_hypothesis(
        (far, near),
        (near.rvec, near.tvec),
    )

    assert selected is near


def test_hypothesis_selection_keeps_much_better_visual_candidate_on_turn():
    good_turn = SimpleNamespace(
        overall_reprojection_error=2.0,
        reprojection_error=0.0,
        rvec=np.asarray([[0.0], [0.0], [1.0]], dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    stale = SimpleNamespace(
        overall_reprojection_error=8.0,
        reprojection_error=0.0,
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )

    selected = render_script.select_pose_hypothesis(
        (good_turn, stale),
        (stale.rvec, stale.tvec),
    )

    assert selected is good_turn


def test_pose_graph_prefers_path_with_bounded_derivatives():
    def candidate(position, error=0.0):
        return SimpleNamespace(
            overall_reprojection_error=error,
            reprojection_error=0.0,
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.asarray([[position], [0.0], [200.0]], dtype=np.float64),
        )

    frames = (
        render_script.PoseHypothesisFrame(0, (candidate(0.0),)),
        render_script.PoseHypothesisFrame(
            1,
            (candidate(1.0), candidate(50.0)),
        ),
        render_script.PoseHypothesisFrame(2, (candidate(2.0),)),
    )
    selection = render_script.select_pose_graph_path(
        frames,
        render_script.PoseMotionFilterConfig(
            dt_seconds=1.0,
            max_acceleration_units_s2=5.0,
            max_jerk_units_s3=5.0,
            max_angular_acceleration_rad_s2=5.0,
            max_angular_jerk_rad_s3=5.0,
        ),
    )

    assert selection is not None
    assert selection.path[1].tvec[0, 0] == 1.0
    assert selection.violates_limits is False


def test_pose_graph_does_not_use_jerk_as_a_constraint():
    def candidate(position):
        return SimpleNamespace(
            overall_reprojection_error=0.0,
            reprojection_error=0.0,
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.asarray([[position], [0.0], [200.0]], dtype=np.float64),
        )

    frames = tuple(
        render_script.PoseHypothesisFrame(index, (candidate(position),))
        for index, position in enumerate((0.0, 0.0, 0.0, 1.0))
    )
    selection = render_script.select_pose_graph_path(
        frames,
        render_script.PoseMotionFilterConfig(
            dt_seconds=1.0,
            max_acceleration_units_s2=2.0,
            max_jerk_units_s3=0.001,
            max_angular_acceleration_rad_s2=5.0,
            max_angular_jerk_rad_s3=0.001,
        ),
    )

    assert selection is not None
    assert selection.violates_limits is False


def test_pose_assignment_keeps_selected_sticker_correspondences_fixed():
    image_size = (256, 256)
    camera_matrix = render_script.default_camera_matrix(image_size, 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [240.0]], dtype=np.float64)
    object_points, class_ids = render_script.visible_cube_lattice_points(rvec, tvec)
    observed, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    observed = observed.reshape(-1, 2)

    _score, rows, columns, _projected, visible_objects, _visible_classes = (
        render_script.evaluate_pose_hypothesis(
            rvec,
            tvec,
            observed,
            class_ids,
            image_size,
            camera_matrix,
            dist_coeffs,
            fixed_observed_indices=np.asarray([0, 1, 2, 3], dtype=np.int32),
            fixed_object_points=object_points[:4],
        )
    )

    assert len(rows) == len(observed)
    np.testing.assert_array_equal(rows[:4], np.asarray([0, 1, 2, 3]))
    np.testing.assert_allclose(visible_objects[columns[:4]], object_points[:4])


def test_naive_assignment_allows_multiple_observations_to_choose_one_target():
    observed = np.asarray([(10.0, 10.0), (10.5, 10.25), (80.0, 80.0)])
    projected = np.asarray([(10.0, 10.0), (40.0, 40.0), (80.0, 80.0)])

    score, rows, columns = render_script.observed_keypoint_reprojection_nearest_assignment(
        observed,
        projected,
        image_size=(100, 100),
    )

    assert np.isfinite(score)
    np.testing.assert_array_equal(rows, np.asarray([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(columns, np.asarray([0, 0, 2], dtype=np.int32))


def test_pivot_consistency_penalizes_a_swapped_free_assignment():
    image_size = (640, 480)
    camera_matrix = render_script.default_camera_matrix(image_size, 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    true_rvec = np.asarray([[0.22], [-0.16], [0.10]], dtype=np.float64)
    true_tvec = np.asarray([[5.0], [-4.0], [250.0]], dtype=np.float64)
    object_points, class_ids = render_script.visible_cube_lattice_points(
        true_rvec,
        true_tvec,
    )
    observed, _ = cv2.projectPoints(
        object_points,
        true_rvec,
        true_tvec,
        camera_matrix,
        dist_coeffs,
    )
    observed = observed.reshape(-1, 2)
    candidate_rvec = true_rvec + np.asarray([[0.03], [-0.02], [0.025]])
    candidate_tvec = true_tvec + np.asarray([[2.0], [-1.5], [1.0]])
    projected, _ = cv2.projectPoints(
        object_points,
        candidate_rvec,
        candidate_tvec,
        camera_matrix,
        dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    rows = np.arange(len(observed), dtype=np.int32)
    correct_columns = rows.copy()
    swapped_columns = correct_columns.copy()
    swapped_columns[6], swapped_columns[7] = swapped_columns[7], swapped_columns[6]
    fixed_rows = np.arange(4, dtype=np.int32)

    correct_score = render_script.pivot_residual_consistency_score(
        observed,
        projected,
        object_points,
        rows,
        correct_columns,
        fixed_rows,
        camera_matrix,
        dist_coeffs,
        candidate_rvec,
        candidate_tvec,
    )
    swapped_score = render_script.pivot_residual_consistency_score(
        observed,
        projected,
        object_points,
        rows,
        swapped_columns,
        fixed_rows,
        camera_matrix,
        dist_coeffs,
        candidate_rvec,
        candidate_tvec,
    )
    class_aware_swapped_score = render_script.pivot_residual_consistency_score(
        observed,
        projected,
        object_points,
        rows,
        swapped_columns,
        fixed_rows,
        camera_matrix,
        dist_coeffs,
        candidate_rvec,
        candidate_tvec,
        observed_class_ids=class_ids,
        projected_class_ids=class_ids,
        use_class_aware_assignment=True,
    )

    assert swapped_score > correct_score
    assert class_aware_swapped_score > swapped_score


def test_assignment_residual_direction_statistics_detects_common_direction():
    observed = np.asarray([(4.0, 1.0), (12.0, 2.0), (8.0, 8.0)])
    projected = observed - np.asarray((3.0, 2.0))

    consistency, mean_residual = (
        render_script.assignment_residual_direction_statistics(observed, projected)
    )

    assert consistency > 0.99
    np.testing.assert_allclose(mean_residual, np.asarray((3.0, 2.0)))


