import pytest

from scripts.independent_keypoint_rotation_budget import FLAT, RotationGeometry, routes_between, double_turn_edges


def costs(previous, current, budget=120.):
    return {route.groups: route.degrees for route in routes_between(previous, current, budget)}


def test_flat_to_flat_only_admits_one_whole_quarter():
    routes = costs(FLAT, FLAT)
    assert routes[()] == 0.
    assert routes[((0, (0, 0, 1)),)] == 90.
    assert routes[((0, (1, 1, 0)),)] == 90.
    assert ((0, (0, 0, 2)),) not in routes
    assert ((0, (0, 0, 1)), (1, (0, 0, 1))) not in routes
    assert all(cost in (0., 90.) for cost in routes.values())
    assert len(routes) == 37  # Face/wide/slice quarters and equivalent same-direction outer motion.


@pytest.mark.parametrize('angle', [30., 45.])
def test_aligned_middle_and_upper_are_a_negative_outer_offset(angle):
    before = RotationGeometry.make(2, (0., -angle, -angle))
    after = RotationGeometry.make(1, (15., 0., 0.))
    finish_d = ((2, (1, 0, 0)),)
    expected = 90. - angle + 15.
    assert costs(before, after)[finish_d] == expected
    assert finish_d not in costs(before, after, expected - .001)
    # Changing the common rigid orientation leaves relative travel unchanged.
    centered = RotationGeometry.make(2, (angle, 0., 0.))
    assert costs(centered, after)[finish_d] == expected


def test_both_outer_offsets_are_measured_from_the_middle():
    before = RotationGeometry.make(2, (0., -30., 15.))
    after = RotationGeometry.make(1, (15., 0., 0.))
    assert costs(before, after)[((2, (1, 0, 0)),)] == 120.


def test_slice_turn_spends_one_quarter_from_shared_budget():
    whole_slice = ((0, (0, 1, 0)),)
    assert whole_slice not in costs(FLAT, FLAT, 89.999)
    assert costs(FLAT, FLAT, 90.)[whole_slice] == 90.
    partial = RotationGeometry.make(0, (0., 30., 0.))
    assert costs(FLAT, partial, 30.)[()] == 30.
    assert () not in costs(FLAT, partial, 29.999)
    # A hidden slice is one turn, with the outer residual travel added.
    before = RotationGeometry.make(2, (0., 0., 3.))
    after = RotationGeometry.make(2, (0., 0., 14.))
    assert whole_slice not in costs(before, after, 106.999)
    assert costs(before, after, 107.)[whole_slice] == 107.


def test_slice_residual_seam_counts_concurrent_relative_distance():
    before = RotationGeometry.make(0, (0., 40., 0.))
    after = RotationGeometry.make(0, (0., -40., 0.))
    assert costs(before, after)[((0, (0, 1, 0)),)] == 10.


def test_rigid_rotation_does_not_spend_turn_budget_or_add_orientation_rescue():
    common = RotationGeometry.make(2, (30., 30., 30.))
    assert costs(common, FLAT, 0.) == {(): 0.}
    assert costs(FLAT, FLAT, 0.) == {(): 0.}


def test_hidden_turn_between_two_wide_residuals_counts_wide_once():
    before = RotationGeometry.make(2, (0., 3., 3.))
    after = RotationGeometry.make(2, (0., 14., 14.))
    assert costs(before, after)[((1, (1, 0, 0)),)] == 107.
    assert ((1, (1, 0, 0)),) not in costs(before, after, 106.999)


def test_finishing_previous_turn_can_leave_room_for_another_whole_turn():
    before = RotationGeometry.make(0, (0., 0., 30.))
    route = ((0, (0, 0, 1)), (2, (0, 0, 1)))
    assert costs(before, FLAT, 150.)[route] == 150.
    assert route not in costs(before, FLAT, 120.)


def test_axis_change_can_unwind_or_complete_the_previous_residual():
    before = RotationGeometry.make(0, (0., 0., 15.))
    after = RotationGeometry.make(2, (0., 0., 30.))
    finish_then_hidden = ((0, (0, 0, 1)), (1, (1, 0, 0)))
    unwind_then_hidden = ((1, (1, 0, 0)),)
    routes = costs(before, after, 150.)
    assert finish_then_hidden not in routes  # 75 + 90 + 30 = 195.
    assert routes[unwind_then_hidden] == 135.


def test_same_layer_continuation_uses_unwrapped_angular_distance():
    before = RotationGeometry.make(0, (0., 0., 30.))
    after = RotationGeometry.make(0, (0., 0., 40.))
    routes = costs(before, after)
    assert routes[()] == 10.
    assert routes[((0, (0, 0, 1)),)] == 100.
    assert routes[((0, (0, 0, -1)),)] == 80.


def test_crossing_the_residual_seam_does_not_charge_a_full_quarter():
    before = RotationGeometry.make(0, (0., 0., 40.))
    after = RotationGeometry.make(0, (0., 0., -40.))
    routes = costs(before, after)
    assert routes[((0, (0, 0, 1)),)] == 10.
    assert routes[((0, (0, 0, 2)),)] == 100.
    assert routes[()] == 80.


def test_hidden_different_axis_requires_an_aligned_boundary():
    before = RotationGeometry.make(0, (0., 0., 15.))
    after = RotationGeometry.make(0, (0., 0., 30.))
    assert ((1, (1, 0, 0)),) not in costs(before, after)
    assert costs(before, after, 135.)[((1, (1, 0, 0)),)] == 135.


@pytest.mark.parametrize('budget', [-1., float('inf'), float('nan')])
def test_invalid_budgets_are_rejected(budget):
    with pytest.raises(ValueError):
        routes_between(FLAT, FLAT, budget)


def test_simultaneous_turns_obey_individual_and_pooled_limits():
    before = RotationGeometry.make(0, (0., 0., 0.))
    after = RotationGeometry.make(0, (40., 0., -40.))
    assert () not in costs(before, after, 60.)
    assert not any(r.groups == () for r in routes_between(before, after, 60., 75.))
    assert any(r.groups == () and r.degrees == 80.
               for r in routes_between(before, after, 60., 80.))
    too_fast = RotationGeometry.make(0, (61., 0., -20.))
    assert () not in costs(before, too_fast, 60.)
    # Merely moving two slabs never unlocks the 50 TPS rate.
    assert not any(r.groups == () for r in routes_between(before, too_fast, 60., 75.))


def test_sequential_different_axes_still_share_frame_time():
    before = RotationGeometry.make(0, (40., 0., 0.))
    after = RotationGeometry.make(1, (0., 0., 40.))
    assert () not in costs(before, after, 60.)
    assert costs(before, after, 80.)[()] == 80.


def test_whole_half_turn_gets_double_rate():
    double = ((0, (2, 0, 0)),)
    assert double not in costs(FLAT, FLAT, 150.)
    assert not any(r.groups == double for r in routes_between(FLAT, FLAT, 150., 187.5))
    witnessed = frozenset({('direct', 0, 2)})
    assert not any(r.groups == double
                   for r in routes_between(FLAT, FLAT, 150., 150., witnessed))
    assert any(r.groups == double and r.degrees == 180. and r.budget == 187.5
               for r in routes_between(FLAT, FLAT, 150., 187.5, witnessed))


def test_double_span_support_includes_motion_before_second_commit():
    frames = (0, 1, 2, 3)
    first = RotationGeometry.make(0, (-20., 0., 0.))
    second = RotationGeometry.make(0, (20., 0., 0.))
    shapes = {0: {0: FLAT}, 1: {0: first}, 2: {0: second}, 3: {0: FLAT}}
    support = double_turn_edges(frames, shapes, dict.fromkeys(frames, 75.))
    assert (1, None, 0, 0, 1) in support  # First 70 degrees, before even the first quarter boundary.
    assert (2, 0, 0, 0, 0) in support
    assert (3, 0, None, 0, 1) in support
    quarter = ((0, (1, 0, 0)),)
    assert quarter not in costs(FLAT, first, 60.)
    assert any(r.groups == quarter and r.degrees == 70.
               for r in routes_between(FLAT, first, 60., 75., frozenset({('direct', 0, 1)})))
    # Stopping after an ordinary 90-degree turn provides no double witness.
    single = {0: {0: FLAT}, 1: {0: first}, 2: {0: FLAT}}
    assert not double_turn_edges((0, 1, 2), single, {0: 75., 1: 75., 2: 75.})


def test_fast_double_does_not_raise_the_other_simultaneous_turn_limit():
    before = RotationGeometry.make(0, (-20., 0., 0.))
    after = RotationGeometry.make(0, (45., 0., 61.))
    assert not any(r.groups == () for r in routes_between(before, after, 60., 75.,
                   frozenset({('direct', 0, 0)})))


def test_possible_double_does_not_raise_an_ambiguous_single_turn_limit():
    frames = (0, 1, 2, 3)
    first = RotationGeometry.make(0, (-20., 0., 0.))
    second = RotationGeometry.make(0, (20., 0., 0.))
    shapes = {0: {0: FLAT}, 1: {0: first}, 2: {0: second, 1: FLAT}, 3: {0: FLAT}}
    support = double_turn_edges(frames, shapes, dict.fromkeys(frames, 75.))
    assert (1, None, 0, 0, 1) not in support


def test_preparation_never_gives_hidden_different_axis_singles_the_double_rate():
    two_singles = ((0, (0, 0, 1)), (1, (0, 0, 1)))
    prepared = routes_between(FLAT, FLAT, 150., 187.5, optimistic_doubles=True)
    assert all(r.groups != two_singles for r in prepared)
    assert any(r.groups == ((0, (0, 0, 2)),) for r in prepared)


def test_slice_specific_limit_is_basis_independent_and_preserves_face_and_wide_rate():
    for axis in range(3):
        limited = routes_between(FLAT, FLAT, 90., slice_budget=90.*40./54.)
        for offsets in ((0, -1, 0), (1, 0, 1)):
            group = ((axis, offsets),)
            assert not any(r.groups == group for r in limited)
            assert any(r.groups == group and r.degrees == 90.
                       for r in routes_between(FLAT, FLAT, 121.5, slice_budget=90.))
        for offsets in ((1, 0, 0), (1, 1, 0)):
            assert any(r.groups == ((axis, offsets),) for r in limited)


def test_reviewed_slice_and_cross_axis_partial_offset_spends_one_slice():
    before = RotationGeometry.make(2, (15.045883504893528, 0., 23.665434536185003))
    after = RotationGeometry.make(0, (0., 0., 28.678335684368218))
    budget = 90.*54.*2./(60000./1001.)
    route = next(r for r in routes_between(before, after, budget, slice_budget=budget*40./54.)
                 if r.groups == ((2, (0, -1, 0)),))
    assert abs(route.degrees - 103.6324521794747) < 1e-10
