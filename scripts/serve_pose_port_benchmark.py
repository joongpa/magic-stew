"""Serve the standalone browser benchmark locally and save its posted report."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1] / "pose-fitter"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self):
        if self.path != "/api/report":
            return self.send_error(404)
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 4_000_000:
            return self.send_error(413)
        try:
            value = json.loads(self.rfile.read(size))
            assert isinstance(value["cases"], list)
        except (ValueError, KeyError, AssertionError, TypeError):
            return self.send_error(400)
        output = ROOT / "results/browser.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"saved")


if __name__ == "__main__":
    print("http://127.0.0.1:4177/bench/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 4177), Handler).serve_forever()
