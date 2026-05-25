from live_recognition.orientation import all_cube_orientations, orientation_hypotheses


def test_all_cube_orientations_has_24_unique_rotations():
    orientations = all_cube_orientations()
    as_tuples = {
        tuple(sorted(orientation.items()))
        for orientation in orientations
    }

    assert len(orientations) == 24
    assert len(as_tuples) == 24


def test_orientation_hypotheses_are_probabilistic():
    hypotheses = orientation_hypotheses(0.75, max_hypotheses=6)

    assert len(hypotheses) == 6
    assert sum(h.probability for h in hypotheses) == 1.0
    assert hypotheses[0].probability > hypotheses[-1].probability

