"""Physical sticker centers in the existing U,D,F,B,R,L convention."""

import numpy as np

from live_recognition.direct_cube_pose import FACE_AXES


def sticker_template():
    points, owners, normals = [], [], []
    for n in np.asarray(FACE_AXES, dtype=np.float32):
        first = np.eye(3)[(np.argmax(abs(n)) + 1) % 3]
        second = np.cross(n, first)
        for u in (-1, 0, 1):
            for v in (-1, 0, 1):
                points.append(n * 1.5 + u * first + v * second)
                owners.append(n + u * first + v * second)
                normals.append(n)
    return np.asarray(points), np.asarray(owners), np.asarray(normals)
