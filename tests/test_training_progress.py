"""Progress must occupy one physical row, including narrow Windows consoles."""

import io
import os

import pytest

from live_recognition.training_progress import ConsoleProgress


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("columns", [40, 80, 120])
def test_updates_never_wrap_and_clear_previous_status(monkeypatch, columns):
    monkeypatch.setattr(
        "live_recognition.training_progress.shutil.get_terminal_size",
        lambda **_: os.terminal_size((columns, 24)),
    )
    stream = Terminal()
    progress = ConsoleProgress(stream)
    progress.update("train 10/1005 loss=1.234 R=12.3 turn=17.0 vis=6.2px " * 4)
    progress.update("train 20/1005")
    # Emulate cursor movement and automatic right-edge wrapping. A bare CR
    # cannot recover any earlier row once an update has wrapped.
    row = [" "] * columns
    cursor = 0
    for char in stream.getvalue():
        if char == "\r":
            cursor = 0
        else:
            assert char != "\n" and cursor < columns - 1
            row[cursor] = char
            cursor += 1
    assert "".join(row).rstrip() == "train 20/1005"
    progress.close()
    assert stream.getvalue().count("\n") == 1


def test_terminal_resize_never_pads_to_previous_wide_row(monkeypatch):
    sizes = iter([120, 40])
    monkeypatch.setattr(
        "live_recognition.training_progress.shutil.get_terminal_size",
        lambda **_: os.terminal_size((next(sizes), 24)),
    )
    stream = Terminal()
    progress = ConsoleProgress(stream)
    progress.update("x" * 200)
    progress.update("train 20/1005")
    assert len(stream.getvalue().rsplit("\r", 1)[1]) <= 39


def test_redirected_output_contains_only_final_summary():
    stream = io.StringIO()
    with ConsoleProgress(stream) as progress:
        for batch in range(100):
            progress.update(f"train {batch}/100")
        assert stream.getvalue() == ""
    assert stream.getvalue() == "train 99/100\n"


def test_trainer_keeps_batch_updates_on_one_row_without_changing_metrics(monkeypatch):
    import torch

    from scripts.train_yolo26_staged_cube_pose import run_epoch

    class Criterion:
        def __call__(self, *_args):
            return (
                torch.tensor(1.0),
                {"phase": torch.tensor(0.5), "geometry": torch.tensor(0.25)},
                None,
            )

        def metrics(self, *_args):
            return {
                "moving_turn_deg": (17.0, 1),
                "static_rotation_deg": (12.0, 1),
                "visible_keypoint_rmse_px": (6.0, 1),
            }

    stream = Terminal()
    monkeypatch.setattr("sys.stdout", stream)
    monkeypatch.setattr(
        "live_recognition.training_progress.shutil.get_terminal_size",
        lambda **_: os.terminal_size((80, 24)),
    )
    samples = [({"img": torch.zeros(1, 3, 64, 64)}, {}, ["sample"])] * 21
    metrics = run_epoch(
        torch.nn.Identity(), samples, Criterion(), torch.device("cpu"), "joint", None, None, None
    )
    assert metrics["moving_turn_deg"] == 17.0 and metrics["total"] == 1.0
    assert stream.getvalue().count("\n") == 1
    for update in stream.getvalue().rstrip("\n").split("\r")[1:]:
        assert len(update) < 80
    assert "val 21/21" in stream.getvalue().rsplit("\r", 1)[1]
