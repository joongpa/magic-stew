"""Small, renderer-independent comparison of saved sticker assignments."""
import cv2
import numpy as np

def compare_sticker_assignments(geometry, initial, final):
    """Compare face IDs in a common cube frame, plus assignment validity."""
    first, second = np.asarray(initial["nearest"]), np.asarray(final["nearest"])
    valid_first, valid_second = np.asarray(initial["valid"]), np.asarray(final["valid"])
    assert first.shape == second.shape == valid_first.shape == valid_second.shape
    status_changes = valid_first != valid_second
    raw_face_changes = valid_first & valid_second & (first != second)
    r1 = cv2.Rodrigues(np.asarray(initial["params"][:3]))[0]
    r2 = cv2.Rodrigues(np.asarray(final["params"][:3]))[0]
    symmetries = np.asarray(geometry["symmetries"])
    alternatives = r2 @ symmetries.transpose(0, 2, 1)
    symmetry_index = int(np.argmax(np.einsum("nij,ij->n", alternatives, r1)))
    symmetry = symmetries[symmetry_index]
    centers = np.asarray(geometry["faces"]).mean(axis=1)
    transformed = centers @ symmetry.T
    permutation = np.linalg.norm(transformed[:, None] - centers[None], axis=2).argmin(axis=1)
    assert np.max(np.linalg.norm(transformed - centers[permutation], axis=1)) < 1e-5
    common_face_changes = valid_first & valid_second & (first != permutation[second])
    return dict(
        sticker_assignment_changes=int(np.count_nonzero(status_changes | common_face_changes)),
        sticker_assignment_raw_changes=int(np.count_nonzero(status_changes | raw_face_changes)),
        sticker_validity_changes=int(np.count_nonzero(status_changes)),
        sticker_symmetry_index=symmetry_index,
    )

