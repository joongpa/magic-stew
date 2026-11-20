"""Shared 18-class, clockwise four-corner sticker annotation schema."""

from itertools import product

import numpy as np

COLORS = ("white", "yellow", "orange", "red", "blue", "green")
TYPES = ("corner", "edge", "center")
CLASS_NAMES = tuple(f"{color} {kind}" for color in COLORS for kind in TYPES)
KEYPOINT_SHAPE = [4, 3]
CENTER_CLASS_IDS = tuple(range(2, 18, 3))
FORMAT = "sticker_quads_v1"


def class_id(color, kind):
    return COLORS.index(color.lower()) * 3 + TYPES.index(kind)


def manifest():
    return {
        "format": FORMAT,
        "class_names": list(CLASS_NAMES),
        "keypoint_shape": KEYPOINT_SHAPE,
        "keypoints_per_sticker": 4,
        "landmark_class_names": [],
        "center_class_ids": list(CENTER_CLASS_IDS),
        "ordering": {
            "winding": "clockwise in image coordinates",
            "corner": "C, CE, CEC, CE; 0 is cube corner",
            "edge": "CE, CE, CEC, CEC; 0->1 is outside edge",
            "center": "minimum loss over four cyclic targets; coordinates and visibility together",
        },
        "geometry": "cubie-face grid corners",
        "horizontal_flip": "disabled; class-dependent reordering required",
    }


def face_quad(owner, normal):
    """One cubie face on a unit-spaced canonical lattice, plus old landmark roles."""
    owner, normal = np.asarray(owner, float), np.asarray(normal, float)
    axis = int(np.argmax(abs(normal)))
    tangents = [i for i in range(3) if i != axis]
    offsets = np.zeros((4, 3))
    offsets[:, axis] = np.sign(normal[axis]) * 0.5
    offsets[:, tangents] = np.asarray(((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)))
    points = owner + offsets
    roles = 3 - (abs(points) > 1.49).sum(-1)
    return points, roles.astype(int), TYPES[3 - int(np.count_nonzero(owner))]


def surface_template():
    quads, owners, normals, roles, kinds = [], [], [], [], []
    for owner in product((-1, 0, 1), repeat=3):
        for axis in range(3):
            if not owner[axis]:
                continue
            normal = np.eye(3)[axis] * owner[axis]
            q, r, kind = face_quad(owner, normal)
            quads.append(q)
            owners.append(owner)
            normals.append(normal)
            roles.append(r)
            kinds.append(kind)
    return np.asarray(quads), np.asarray(owners), np.asarray(normals), np.asarray(roles), kinds


def transformed_cubie_face(matrix, solved_location, solved_rotation, world_normal):
    """Exact face corners and semantic roles from the cubie's current transform.

    The solved rotation maps mesh-local axes into the original cube lattice.
    Thus scramble moves and arbitrary partial turns cannot change point roles.
    """
    matrix, solved_rotation = np.asarray(matrix, float), np.asarray(solved_rotation, float)
    local_normal = np.linalg.solve(matrix[:3, :3], np.asarray(world_normal, float))
    local_axis = int(np.argmax(abs(local_normal)))
    local_normal = np.eye(3)[local_axis] * np.sign(local_normal[local_axis])
    tangents = [i for i in range(3) if i != local_axis]
    offsets = np.zeros((4, 3))
    offsets[:, local_axis] = local_normal[local_axis] * 0.5
    offsets[:, tangents] = ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))
    solved = np.asarray(solved_location, float) + offsets @ solved_rotation.T
    roles = 3 - (abs(solved) > 1.49).sum(-1)
    kind = TYPES[3 - int(np.count_nonzero(np.rint(solved_location)))]
    return offsets @ matrix[:3, :3].T + matrix[:3, 3], roles.astype(int), kind


def yolo_row(label_class, pixels, visibility, size):
    """Use the clipped full quad bounds; unavailable points have zero triplets."""
    pixels, size = np.asarray(pixels, float), np.asarray(size, float)
    lo = np.clip(pixels.min(0), 0, size)
    hi = np.clip(pixels.max(0), 0, size)
    if np.any(hi <= lo):
        raise ValueError("Sticker has no positive in-frame bounding box")
    fields = [int(label_class), *((hi + lo) / (2 * size)), *((hi - lo) / size)]
    for point, v in zip(pixels, visibility):
        if int(v) and np.all(point >= 0) and np.all(point < size):
            fields.extend([*(point / size), int(v)])
        else:
            fields.extend((0, 0, 0))
    return " ".join(str(x) if isinstance(x, int) else f"{x:.8f}" for x in fields)
