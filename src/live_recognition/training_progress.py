"""Single-row console progress without terminal wrapping or captured-log spam."""

import os
import shutil
import sys


class ConsoleProgress:
    def __init__(self, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.previous_width = 0
        self.latest = None

    def update(self, message):
        self.latest = str(message).replace("\r", " ").replace("\n", " ")
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
        self.stream.write("\n" if self.interactive else self.latest + "\n")
        self.stream.flush()
        self.latest = None

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()
