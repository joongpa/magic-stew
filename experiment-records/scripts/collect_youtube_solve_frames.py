"""Collect one representative frame from public YouTube cube-solve videos.

The collector is intentionally resumable.  Search results and per-video state are
stored in the output directory, and already extracted frames are never replaced
unless ``--overwrite`` is supplied.

This uses yt-dlp only for public YouTube search/metadata and a short, low-resolution
section around the selected timestamp.  It does not keep the downloaded videos.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_QUERIES = [
    "3x3 solve",
    "3x3 single solve",
    "3x3 speedsolve",
    "3x3 speed solve",
    "3x3 competition solve",
    "3x3 official solve",
    "3x3 WCA solve",
    "3x3 ao5",
    "3x3 average",
    "3x3 finals solve",
    "3x3 round solve",
    "3x3 semifinal solve",
    "3x3 world championship solve",
    "rubik cube competition solve",
    "2x2 solve",
    "2x2 competition solve",
    "4x4 solve",
    "4x4 competition solve",
    "5x5 solve",
    "5x5 competition solve",
    "6x6 solve",
    "7x7 solve",
    "one handed cube solve",
    "OH speedsolve",
    "blindfolded cube solve",
    "megaminx solve",
    "pyraminx solve",
    "skewb solve",
    "3x3 solve 2025",
    "3x3 solve 2024",
    "3x3 solve 2023",
    "3x3 solve 2022",
    "3x3 solve 2021",
    "3x3 solve 2020",
    "3x3 solve 2019",
    "3x3 solve 2018",
    "3x3 solve 2017",
    "3x3 solve 2016",
    "3x3 at home solve",
    "3x3 live solve",
    "rubik speed solve video",
    "cubing solve video",
]

DIMENSION_RE = re.compile(
    r"(?:\b[2-7]x[2-7](?:x[2-7])?\b|\b(?:one[- ]handed|OH|blindfolded|BLD|megaminx|pyraminx|skewb)\b)",
    re.IGNORECASE,
)
SOLVE_RE = re.compile(
    r"\b(?:solve|solves|speedsolve|speed[- ]solve|ao\d+|average|single|round|finals?|competition|official)\b",
    re.IGNORECASE,
)
REJECT_RE = re.compile(
    r"\b(?:review|unboxing|tutorial|how to|method|algorithm|setup|tips?|comparison|magnet|spring|lube|challenge|reaction|reconstruction|news|podcast|walkthrough|explained|guide|compilation|fastest|top\s*\d+|records?|example|advanced)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Candidate:
    video_id: str
    title: str
    url: str
    duration: float | None
    channel: str | None
    description: str | None
    query: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/youtube_solve_frames"))
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--results-per-query", type=int, default=100)
    parser.add_argument("--queries-file", type=Path)
    parser.add_argument("--phase", choices=["search", "download", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument(
        "--video-height",
        type=int,
        default=360,
        help="Maximum source-video height. Use 720 or higher when logos need to remain legible.",
    )
    parser.add_argument("--socket-timeout", type=int, default=15)
    parser.add_argument(
        "--extractor-args",
        default="youtube:player_client=tv_embedded",
        help="yt-dlp extractor arguments; TV embedded exposes more public video formats",
    )
    parser.add_argument("--yt-dlp", default="yt-dlp")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def read_queries(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_QUERIES
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def run_json(command: list[str]) -> dict[str, Any] | None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        print(f"yt-dlp failed ({result.returncode}): {result.stderr.strip()[-500:]}")
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"yt-dlp returned invalid JSON: {result.stdout[:200]!r}")
        return None


def candidate_score(item: dict[str, Any]) -> float | None:
    title = str(item.get("title") or "").strip()
    description = str(item.get("description") or "")
    text = f"{title} {description}"
    if not title or REJECT_RE.search(title):
        return None
    has_dimension = bool(DIMENSION_RE.search(title))
    has_solve_signal = bool(SOLVE_RE.search(title))
    if not (has_dimension and has_solve_signal):
        return None

    score = 0.0
    if re.search(r"\bsolve\b|speedsolve|speed[- ]solve", title, re.IGNORECASE):
        score += 4.0
    if re.search(r"\b(?:competition|official|round|finals?)\b", title, re.IGNORECASE):
        score += 2.0
    if re.search(r"\b(?:single|ao\d+|average)\b", title, re.IGNORECASE):
        score += 1.0
    if re.search(r"\d+(?:\.\d+)?\s*(?:s|sec|seconds?)\b", title, re.IGNORECASE):
        score += 1.0
    if REJECT_RE.search(text):
        score -= 2.0
    duration = item.get("duration")
    if isinstance(duration, (int, float)) and 2 <= duration <= 300:
        score += 1.0
    return score


def search_candidates(yt_dlp: str, queries: Iterable[str], results_per_query: int) -> list[Candidate]:
    by_id: dict[str, Candidate] = {}
    for query in queries:
        print(f"Searching: {query}")
        payload = run_json([
            yt_dlp,
            "--flat-playlist",
            "--dump-single-json",
            "--skip-download",
            "--no-warnings",
            f"ytsearch{results_per_query}:{query}",
        ])
        if not payload:
            continue
        for item in payload.get("entries") or []:
            if not item or not item.get("id"):
                continue
            score = candidate_score(item)
            if score is None:
                continue
            candidate = Candidate(
                video_id=str(item["id"]),
                title=str(item.get("title") or ""),
                url=f"https://www.youtube.com/watch?v={item['id']}",
                duration=float(item["duration"]) if isinstance(item.get("duration"), (int, float)) else None,
                channel=item.get("channel") or item.get("uploader"),
                description=item.get("description"),
                query=query,
                score=score,
            )
            previous = by_id.get(candidate.video_id)
            if previous is None or candidate.score > previous.score:
                by_id[candidate.video_id] = candidate
    return sorted(by_id.values(), key=lambda item: (-item.score, item.video_id))


def selected_timestamp(duration: float | None) -> float:
    if duration is None or duration <= 0:
        return 3.0
    return max(1.0, min(duration - 1.0, duration * 0.5))


def error_summary(stderr: str) -> str:
    # yt-dlp/ffmpeg may print signed media URLs.  Keep diagnostics useful without
    # persisting those short-lived credentials in the failure manifest.
    lines = [line for line in stderr.splitlines() if "https://" not in line and "http://" not in line]
    return "\n".join(lines)[-1000:]


def extract_frame(
    candidate: Candidate,
    frames_dir: Path,
    temp_dir: Path,
    yt_dlp: str,
    ffmpeg: str,
    overwrite: bool,
    image_format: str,
    video_height: int,
    socket_timeout: int,
    extractor_args: str,
) -> tuple[bool, str | None]:
    destination = frames_dir / f"{candidate.video_id}.{image_format}"
    if destination.exists() and not overwrite:
        return True, None

    timestamp = selected_timestamp(candidate.duration)
    temp_video = temp_dir / f"{candidate.video_id}.mp4"
    temp_frame = temp_dir / f"{candidate.video_id}.{image_format}"
    command = [
        yt_dlp,
        "--no-playlist",
        "--no-warnings",
        "--no-part",
        "--force-overwrites",
        "--socket-timeout",
        str(socket_timeout),
        "--retries",
        "1",
        "--fragment-retries",
        "1",
        "--extractor-args",
        extractor_args,
        "-f",
        f"bestvideo[height<={video_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={video_height}][ext=mp4]/best[height<={video_height}]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(temp_video),
        candidate.url,
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0 or not temp_video.exists():
        for path in temp_dir.glob(f"{candidate.video_id}*"):
            path.unlink(missing_ok=True)
        return False, error_summary(result.stderr)

    frame_command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(temp_video),
        "-frames:v",
        "1",
    ]
    if image_format == "jpg":
        frame_command.extend(["-q:v", "2"])
    else:
        frame_command.extend(["-compression_level", "3"])
    frame_command.append(str(temp_frame))
    result = subprocess.run(
        frame_command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not temp_frame.exists():
        for path in temp_dir.glob(f"{candidate.video_id}*"):
            path.unlink(missing_ok=True)
        return False, error_summary(result.stderr)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_frame.replace(destination)
    for path in temp_dir.glob(f"{candidate.video_id}*"):
        path.unlink(missing_ok=True)
    return True, None


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_candidates(path: Path) -> list[Candidate]:
    if not path.exists():
        return []
    return [Candidate(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def merge_candidates(*groups: Iterable[Candidate]) -> list[Candidate]:
    by_id: dict[str, Candidate] = {}
    for group in groups:
        for candidate in group:
            previous = by_id.get(candidate.video_id)
            if previous is None or candidate.score > previous.score:
                by_id[candidate.video_id] = candidate
    return sorted(by_id.values(), key=lambda item: (-item.score, item.video_id))


def main() -> int:
    args = parse_args()
    if args.target <= 0 or args.results_per_query <= 0:
        raise SystemExit("--target and --results-per-query must be positive")
    root = args.output_dir
    frames_dir = root / "frames"
    root.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = root / "candidates.jsonl"
    failures_path = root / "download_failures.jsonl"

    candidates: list[Candidate] = []
    if args.phase in {"search", "all"} or not candidates_path.exists():
        fresh_candidates = search_candidates(args.yt_dlp, read_queries(args.queries_file), args.results_per_query)
        candidates = merge_candidates(read_candidates(candidates_path), fresh_candidates)
        write_jsonl(candidates_path, (candidate.as_dict() for candidate in candidates))
        print(f"Found {len(candidates)} filtered, unique candidates")
    else:
        candidates = read_candidates(candidates_path)

    if args.phase == "search":
        return 0

    failures: list[dict[str, Any]] = []
    successful_frames = sum(
        1 for candidate in candidates if (frames_dir / f"{candidate.video_id}.{args.image_format}").exists()
    )
    with tempfile.TemporaryDirectory(prefix="youtube_solve_frames_", dir=root) as temporary:
        temp_dir = Path(temporary)
        for index, candidate in enumerate(candidates, start=1):
            if successful_frames >= args.target:
                break
            destination = frames_dir / f"{candidate.video_id}.{args.image_format}"
            existed_before = destination.exists()
            ok, error = extract_frame(
                candidate,
                frames_dir,
                temp_dir,
                args.yt_dlp,
                args.ffmpeg,
                args.overwrite,
                args.image_format,
                args.video_height,
                args.socket_timeout,
                args.extractor_args,
            )
            if ok:
                if not existed_before:
                    successful_frames += 1
                    print(f"[{successful_frames}/{args.target}] {candidate.video_id} {candidate.title}")
            else:
                failures.append({**candidate.as_dict(), "error": error})
                print(f"[candidate {index}/{len(candidates)}] FAILED {candidate.video_id}: {error}")
    write_jsonl(failures_path, failures)
    print(f"Extracted {len(list(frames_dir.glob('*.jpg')))} frames into {frames_dir}")
    if successful_frames < args.target:
        print(f"Only {successful_frames} successful frames were available from {len(candidates)} candidates")
    if failures:
        print(f"{len(failures)} failures recorded in {failures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
