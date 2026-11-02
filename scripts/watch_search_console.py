#!/usr/bin/env python3
"""Display a running search's atomic status snapshot on a fixed console cadence."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

SPEC = importlib.util.spec_from_file_location("watched_mean_search",
    Path(__file__).with_name("search_independent_keypoint_mean.py"))
search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = search
SPEC.loader.exec_module(search)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--interval",type=float,default=10.)
    parser.add_argument("--timezone",default="America/Chicago")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("Interval must be positive")
    ZoneInfo(args.timezone)
    deadline = time.monotonic()
    while True:
        terminal = (args.output_dir/"result.json").exists()
        path = args.output_dir/"status.json"
        try:
            item = json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):
            item = None
        if item is not None:
            suffix = f" | {item.get('status',item.get('event'))}" if terminal else ""
            print(search.format_console_progress(item,args.timezone)+suffix,flush=True)
            if terminal:
                return
        if (args.output_dir/"CONSOLE_STOP").exists():
            return
        deadline += args.interval
        time.sleep(max(0.,deadline-time.monotonic()))


if __name__ == "__main__":
    main()
