"""Clockwise cubie-face corner ordering in image coordinates (x right, y down)."""
import numpy as np


LANDMARK_SEQUENCES = {
    "corner": (0, 1, 2, 1),
    "edge": (1, 1, 2, 2),
    "center": (2, 2, 2, 2),
}


def clockwise_sticker_order(points, landmark_classes, sticker_type):
    """Return source indices, anchoring corner0=C and edge0->1=outside CE edge.

    Classes are the ORIGINAL geometric landmark roles, not inferred from the
    turned shape's silhouette. Center start is arbitrary: all four cyclic
    shifts of the returned clockwise order must be accepted by its loss.
    """
    points = np.asarray(points, dtype=float)
    classes = np.asarray(landmark_classes, dtype=int)
    expected = LANDMARK_SEQUENCES[sticker_type]
    if points.shape != (4, 2) or classes.shape != (4,) or not np.isfinite(points).all():
        raise ValueError("Expected four finite pixel points and four landmark classes")
    if sorted(classes) != sorted(expected):
        raise ValueError(f"Incorrect landmark roles for {sticker_type}: {classes.tolist()}")
    centered = points - points.mean(0)
    order = np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))
    vertices = points[order]
    edges = np.roll(vertices, -1, axis=0) - vertices
    next_edges = np.roll(edges, -1, axis=0)
    cross = edges[:, 0]*next_edges[:, 1] - edges[:, 1]*next_edges[:, 0]
    if np.any(cross <= 1e-10):
        raise ValueError("Sticker outline must be a nondegenerate convex quadrilateral")
    if sticker_type == "corner":
        start = int(np.flatnonzero(classes[order] == 0)[0])
    elif sticker_type == "edge":
        roles = classes[order]
        start = int(np.flatnonzero((roles == 1) & (np.roll(roles, -1) == 1))[0])
    else:
        # Reference display convention only, NOT a privileged center target.
        start = int(np.argmin(vertices.sum(-1)))
    order = np.roll(order, -start)
    if tuple(classes[order]) != expected:
        raise ValueError("Landmark roles do not form the requested clockwise outline")
    return order


def center_cyclic_orders(order=(0, 1, 2, 3)):
    """The four equivalent center targets; winding-reversed targets are excluded."""
    order = np.asarray(order, dtype=int)
    if order.shape != (4,) or sorted(order) != [0, 1, 2, 3]:
        raise ValueError("Expected a permutation of four corner indices")
    return np.stack([np.roll(order, -shift) for shift in range(4)])
