"""Evaluation-only full-wireframe agreement with a trusted fitted reference.

These are geometric reference comparisons, never live detector confidence or
evidence available to the tracker. Face-color identity is not evaluated.
"""
import cv2
import numpy as np

from scripts import render_yolo11n_independent_keypoints as m


def center_pixels(candidate, translation):
    return cv2.projectPoints(np.zeros((1, 3)), np.zeros(3), np.asarray(translation),
        candidate.camera_matrix, candidate.dist_coeffs)[0].reshape(2)


def wire_mask(candidate, center, diameter, size=256):
    scale = size/(2.4*diameter)
    image = np.zeros((size, size), np.uint8)
    for _, polygon, _ in m.projected_cubelet_faces(candidate, candidate.rvec, candidate.tvec):
        pixels = np.rint((polygon-center)*scale+size/2).astype(np.int32)
        cv2.polylines(image, [pixels], True, 1, 1, cv2.LINE_8)
    return image, scale


def wire_error(candidate, reference, target_mask=None):
    """Symmetric p90 edge distance, divided by the reference cube diameter."""
    diameter = m.projected_cube_diameter(reference)
    center = center_pixels(reference, reference.tvec)
    target, scale = wire_mask(reference, center, diameter) if target_mask is None else target_mask
    actual, _ = wire_mask(candidate, center, diameter)
    if not actual.any() or not target.any():
        return 1.
    to_actual = cv2.distanceTransform(1-actual, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    to_target = cv2.distanceTransform(1-target, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return float(max(np.percentile(to_actual[target.astype(bool)], 90),
        np.percentile(to_target[actual.astype(bool)], 90))/(scale*diameter))
