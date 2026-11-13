from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.overfit_direct_cube_pose import select_subset  # noqa: E402


@pytest.mark.parametrize("count", [5, 24])
def test_overfit_subset_is_reproducible_unique_and_covers_all_modes(count):
    entries = [{"key": f"{source}_{mode}_{symmetry}_{index}", "source_dataset": str(source),
                "mode": mode, "symmetry_count": symmetry, "pose_supervision": True}
               for source in range(4) for mode in ("none", "mid_turn", "slice_turn", "simultaneous_turn", "corner_cut")
               for symmetry in (1, 4, 24) for index in range(3)]
    entries.append({"key": "empty", "pose_supervision": False})
    subset = select_subset(entries, count, seed=17)
    assert subset == select_subset(entries, count, seed=17)
    assert len(subset) == len({entry["key"] for entry in subset}) == count
    assert {entry["mode"] for entry in subset} == {"none", "mid_turn", "slice_turn", "simultaneous_turn", "corner_cut"}
    assert all(entry["pose_supervision"] for entry in subset)
    if count == 24:
        assert {entry["source_dataset"] for entry in subset} == {"0", "1", "2", "3"}
        assert {entry["symmetry_count"] for entry in subset} == {1, 4, 24}
