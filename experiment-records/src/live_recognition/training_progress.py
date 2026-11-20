"""Single-row console progress without terminal wrapping or captured-log spam."""

import ctypes
import os
import shutil
import sys


class _Coord(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class _SmallRect(ctypes.Structure):
    _fields_ = [(name, ctypes.c_short) for name in ("Left", "Top", "Right", "Bottom")]


class _ScreenBufferInfo(ctypes.Structure):
    _fields_ = [
        ("dwSize", _Coord),
        ("dwCursorPosition", _Coord),
        ("wAttributes", ctypes.c_ushort),
        ("srWindow", _SmallRect),
        ("dwMaximumWindowSize", _Coord),
    ]


class _WindowsConsole:
    """Write into CMD's screen buffer without TextIO/ANSI/CR interpretation."""

    def __init__(self, kernel, handle):
        self.kernel = kernel
        self.handle = handle
        self.previous_width = 0

    def _write(self, text):
        written = ctypes.c_uint32()
        units = len(text.encode("utf-16-le")) // 2
        return (
            bool(self.kernel.WriteConsoleW(self.handle, text, units, ctypes.byref(written), None))
            and written.value == units
        )

    def update(self, message):
        info = _ScreenBufferInfo()
        if not self.kernel.GetConsoleScreenBufferInfo(self.handle, ctypes.byref(info)):
            return False
        # CMD's screen buffer can be wider than its visible window. Use the
        # viewport and position the cursor at its actual left edge.
        left = info.srWindow.Left
        available = max(0, min(info.dwSize.X, info.srWindow.Right + 1) - left - 1)
        text = message[:available]
        width = min(available, max(self.previous_width, len(text)))
        if not self.kernel.SetConsoleCursorPosition(
            self.handle, _Coord(left, info.dwCursorPosition.Y)
        ):
            return False
        if not self._write(text.ljust(width)):
            return False
        self.previous_width = len(text)
        return True

    def close(self):
        return self._write("\r\n")


def _open_windows_console(stream):
    if os.name != "nt":
        return None
    try:
        import msvcrt

        handle = msvcrt.get_osfhandle(stream.fileno())
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel.GetConsoleMode.restype = ctypes.c_int
        kernel.GetConsoleScreenBufferInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ScreenBufferInfo),
        ]
        kernel.GetConsoleScreenBufferInfo.restype = ctypes.c_int
        kernel.SetConsoleCursorPosition.argtypes = [ctypes.c_void_p, _Coord]
        kernel.SetConsoleCursorPosition.restype = ctypes.c_int
        kernel.WriteConsoleW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        kernel.WriteConsoleW.restype = ctypes.c_int
        mode = ctypes.c_uint32()
        if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            return None  # A redirected file/pipe is not a console buffer.
        return _WindowsConsole(kernel, handle)
    except (OSError, ValueError, AttributeError):
        return None


class ConsoleProgress:
    def __init__(self, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.windows = _open_windows_console(self.stream)
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.previous_width = 0
        self.latest = None

    @property
    def backend(self):
        if self.windows is not None:
            return "Windows console cursor"
        return "terminal carriage return" if self.interactive else "captured output (one summary)"

    def update(self, message):
        self.latest = str(message).replace("\r", " ").replace("\n", " ")
        if self.windows is not None:
            self.stream.flush()
            if self.windows.update(self.latest):
                return
            self.windows = None
        if not self.interactive:
            return
        try:
            columns = os.get_terminal_size(self.stream.fileno()).columns
        except (OSError, ValueError, AttributeError):
            columns = shutil.get_terminal_size(fallback=(80, 24)).columns
        # Leave the last column unused: writing it can advance a Windows
        # console to the next row, which a subsequent carriage return cannot
        # erase. Never pad to an old width after the terminal gets narrower.
        available = max(0, columns - 1)
        text = self.latest[:available]
        width = min(available, max(self.previous_width, len(text)))
        self.stream.write("\r" + text.ljust(width))
        self.stream.flush()
        self.previous_width = len(text)

    def close(self):
        if self.latest is None:
            return
        if self.windows is not None and self.windows.close():
            self.latest = None
            return
        self.stream.write("\n" if self.interactive else self.latest + "\n")
        self.stream.flush()
        self.latest = None

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()
