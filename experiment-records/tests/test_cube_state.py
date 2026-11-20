from live_recognition.cube_state import CubeState, FACE_COLORS, FACELET_KEYS, invert_scramble


def test_solved_state_uses_white_top_and_green_front():
    state = CubeState.solved()

    assert state.color_at(((1, 1, 2), 2, 1)) == FACE_COLORS["U"]
    assert state.color_at(((1, 0, 1), 1, -1)) == FACE_COLORS["F"]
    assert len(state.colors) == 54
    assert all(list(state.colors.values()).count(color) == 9 for color in FACE_COLORS.values())


def test_solved_detection_accepts_rotations_and_rejects_unsolved_or_incomplete_states():
    for moves in ((), ("x",), ("y", "z'"), ("R", "R'"), ("Rw", "Rw'")):
        assert CubeState.from_scramble(moves).is_solved()
    for moves in (("R",), ("Rw",), ("R", "U2", "F'")):
        assert not CubeState.from_scramble(moves).is_solved()
    partial = CubeState.solved()
    partial.colors.pop(next(iter(partial.colors)))
    assert not partial.is_solved()
    assert not CubeState({key: "white" for key in FACELET_KEYS}).is_solved()


def test_face_turn_and_inverse_restore_the_state():
    state = CubeState.from_scramble(("R", "U2", "F'", "F", "U2", "R'"))

    assert state.colors == CubeState.solved().colors


def test_half_turn_is_its_own_inverse():
    state = CubeState.from_scramble(("R2", "R2"))

    assert state.colors == CubeState.solved().colors
    assert len(FACELET_KEYS) == 54


def test_wide_move_aliases_apply_two_layers_and_inverse_cleanly():
    for move in ("Rw", "r", "Uw", "f"):
        state = CubeState.from_scramble((move, move + "'"))
        assert state.colors == CubeState.solved().colors

    wide = CubeState.from_scramble(("Rw",))
    single = CubeState.from_scramble(("R",))
    assert wide.colors != single.colors


def test_wide_half_turn_is_its_own_inverse():
    state = CubeState.from_scramble(("Rw2", "r2"))

    assert state.colors == CubeState.solved().colors


def test_slice_and_opposite_outer_updates_differ_by_only_the_anchor_rotation():
    initial = CubeState.from_scramble('R U F2 D')
    for slice_move, outer, anchor in (
        ('M', "R L'", "x'"), ('E', "U D'", "y'"), ('S', "F' B", 'z'),
    ):
        middle = CubeState(dict(initial.colors))
        middle.apply_move(slice_move)
        outside = CubeState(dict(initial.colors))
        for move in (*outer.split(), anchor):
            outside.apply_move(move)
        assert middle.colors == outside.colors
        middle.apply_move(slice_move + "'")
        assert middle.colors == initial.colors


def test_solved_state_uses_kociemba_facelet_order():
    assert (
        CubeState.solved().to_kociemba_facelets()
        == "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    )


def test_face_grids_and_net_cells_build_a_kociemba_state():
    faces = {
        face: [[FACE_COLORS[face]] * 3 for _ in range(3)]
        for face in ("U", "R", "F", "D", "L", "B")
    }
    assert CubeState.from_face_grids(faces).to_kociemba_facelets() == (
        "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    )

    net_cells = {
        (face, row, column): faces[face][row][column]
        for face in faces
        for row in range(3)
        for column in range(3)
    }
    assert CubeState.from_net_cells(net_cells).to_kociemba_facelets() == (
        "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    )


def test_reorient_to_standard_centers_normalizes_whole_cube_rotation():
    state = CubeState.from_scramble(("R", "U2", "F'"))
    rotated = CubeState(dict(state.colors))
    rotated.apply_move("x")

    assert rotated.reoriented_to_standard_centers().colors == state.colors


def test_invert_scramble_reverses_and_inverts_moves():
    assert invert_scramble("R U2 F'") == "F U2 R'"


def test_scramble_from_solved_inverts_kociemba_solution():
    state = CubeState.from_scramble(("R", "U2", "F'"))
    seen_facelets = []

    def fake_solve(facelets: str) -> str:
        seen_facelets.append(facelets)
        return "R U2 F'"

    assert state.scramble_from_solved(solve=fake_solve) == "F U2 R'"
    assert seen_facelets == [state.to_kociemba_facelets()]
