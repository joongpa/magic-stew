"""Progress must occupy one physical row, including narrow Windows consoles."""

import ctypes
import io
import os

import pytest

from live_recognition.training_progress import (
    ConsoleProgress,
    _Coord,
    _ScreenBufferInfo,
    _WindowsConsole,
)


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


class WindowsScreen:
    """CMD can have a300-column buffer behind a much narrower viewport."""

    def __init__(self, columns):
        self.columns = columns
        self.left = 10
        self.x, self.y = 23, 4
        self.rows = [[" "] * 300 for _ in range(20)]
        self.positions = []
        self.newlines = 0

    def GetConsoleScreenBufferInfo(self, _handle, result):
        info = result._obj
        info.dwSize = _Coord(300, 20)
        info.dwCursorPosition = _Coord(self.x, self.y)
        info.srWindow.Left, info.srWindow.Right = self.left, self.left + self.columns - 1
        return True

    def SetConsoleCursorPosition(self, _handle, position):
        self.x, self.y = position.X, position.Y
        self.positions.append((self.x, self.y))
        return True

    def WriteConsoleW(self, _handle, text, units, written, _reserved):
        if text == "\r\n":
            self.x = 0
            self.y += 1
            self.newlines += 1
        else:
            # Native writes must fit the WINDOW, not merely the screen buffer.
            assert len(text) < self.columns
            self.rows[self.y][self.x : self.x + len(text)] = list(text)
            self.x += len(text)
        written._obj.value = units
        return True


@pytest.mark.parametrize("columns", [40, 80, 120])
def test_cmd_native_updates_bypass_stream_line_translation(monkeypatch, columns):
    screen = WindowsScreen(columns)
    monkeypatch.setattr(
        "live_recognition.training_progress._open_windows_console",
        lambda _stream: _WindowsConsole(screen, 42),
    )
    # Even a wrapper reporting non-TTY output must use the real native console
    # if GetConsoleMode identified one. No stream writes occur per update.
    stream = io.StringIO()
    progress = ConsoleProgress(stream)
    assert progress.backend == "Windows console cursor"
    progress.update("batch 1/20 " + "x" * 300)
    progress.update("batch 2/20")
    assert screen.positions == [(10, 4), (10, 4)]
    assert "".join(screen.rows[4][10:]).rstrip() == "batch 2/20"
    assert stream.getvalue() == "" and screen.newlines == 0
    progress.close()
    assert screen.newlines == 1 and screen.y == 5


def test_win32_screen_buffer_structure_layout():
    assert ctypes.sizeof(_Coord) == 4
    assert ctypes.sizeof(_ScreenBufferInfo) == 22
    assert _ScreenBufferInfo.srWindow.offset == 10


def test_sampling_training_diagnostics_preserves_every_optimizer_update():
    import copy

    import torch

    from scripts.train_yolo26_staged_cube_pose import run_epoch

    class Criterion:
        calls = 0

        def __call__(self, model, batch, *_args):
            loss = model(batch["img"]).square().mean()
            return loss, {"phase": loss * 0, "geometry": loss * 0}, None

        def metrics(self, *_args):
            self.calls += 1
            return {"moving_turn_deg": (17.0, 1)}

    original = torch.nn.Linear(1, 1, bias=False)
    sampled, full = copy.deepcopy(original), copy.deepcopy(original)
    samples = [({"img": torch.ones(1, 1, 1, 1)}, {}, ["sample"])] * 21
    results = []
    for model, every in [(sampled, 10), (full, 1)]:
        criterion = Criterion()
        result = run_epoch(
            model,
            samples,
            criterion,
            torch.device("cpu"),
            "static",
            torch.optim.SGD(model.parameters(), lr=0.01),
            torch.amp.GradScaler("cuda", enabled=False),
            None,
            metrics_every=every,
        )
        results.append(result)
        assert criterion.calls == (3 if every == 10 else 21)
    assert torch.equal(sampled.weight, full.weight)
    assert results[0]["total"] == results[1]["total"]
    assert results[0]["pose_metric_images"] == 3 and results[1]["pose_metric_images"] == 21
    criterion = Criterion()
    run_epoch(
        sampled,
        samples,
        criterion,
        torch.device("cpu"),
        "static",
        None,
        None,
        None,
        metrics_every=100,
    )
    assert criterion.calls == 21  # Validation always covers the full set.
