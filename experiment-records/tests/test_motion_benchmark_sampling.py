from scripts.benchmark_physics_pose_tracker import limit_update_rate


def test_thirty_fps_selection_keeps_original_timestamps_and_handles_sparse_logs():
    records = [dict(frame_index=i) for i in range(12)]
    selected = limit_update_rate(records, 59.95, 30.)
    assert [r['frame_index'] for r in selected] == [0, 2, 4, 6, 8, 10]
    assert limit_update_rate(selected, 59.95, 30.) == selected
    assert limit_update_rate(records, 24., 30.) == records
    assert limit_update_rate(records, 60., 0.) is records
