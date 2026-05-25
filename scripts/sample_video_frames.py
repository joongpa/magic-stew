from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample frames from videos at a fixed rate, optionally letterboxed to a model size."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.home() / "Downloads" / "cubingVideos",
        help="Directory containing videos to sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "cubing_frames",
        help="Directory where sampled frames will be written.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Sampling rate in frames per second.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for videos recursively inside --input-dir.",
    )
    parser.add_argument(
        "--image-format",
        choices=["jpg", "png"],
        default="jpg",
        help="Output image format.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable to use.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing sampled frames for each video.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        help=(
            "Optional square output size. Frames are resized with preserved aspect ratio "
            "and padded to imgsz x imgsz. Use 644 for models with max stride 14."
        ),
    )
    return parser


def iter_videos(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def sample_video(
    video_path: Path,
    output_root: Path,
    sample_fps: float,
    image_format: str,
    ffmpeg: str,
    overwrite: bool,
    image_size: int | None,
) -> int:
    output_dir = output_root / f"{video_path.stem}__{video_path.suffix.lower().lstrip('.')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = list(output_dir.glob(f"*.{image_format}"))
    if existing_frames and not overwrite:
        print(f"Skipping existing output: {output_dir}")
        return 0
    for frame_path in existing_frames:
        frame_path.unlink()

    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-vf",
        build_video_filter(sample_fps, image_size),
        "-vsync",
        "0",
        str(output_dir / f"frame_%06d.{image_format}"),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"Skipping video ffmpeg could not process: {video_path}")
        return 0

    return count_frames(output_dir, image_format)


def build_video_filter(sample_fps: float, image_size: int | None) -> str:
    filters = [f"fps={sample_fps}"]
    if image_size is not None:
        filters.extend(
            [
                (
                    f"scale={image_size}:{image_size}:"
                    "force_original_aspect_ratio=decrease"
                ),
                f"pad={image_size}:{image_size}:(ow-iw)/2:(oh-ih)/2",
            ]
        )
    return ",".join(filters)


def count_frames(output_dir: Path, image_format: str) -> int:
    return sum(1 for path in output_dir.glob(f"*.{image_format}") if path.is_file())


def main() -> int:
    args = build_parser().parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be greater than 0")
    if args.imgsz is not None and args.imgsz <= 0:
        raise SystemExit("--imgsz must be greater than 0")
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    videos = iter_videos(args.input_dir, args.recursive)
    if not videos:
        raise SystemExit(f"No videos found in: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    for video_path in videos:
        frame_total = sample_video(
            video_path,
            args.output_dir,
            args.fps,
            args.image_format,
            args.ffmpeg,
            args.overwrite,
            args.imgsz,
        )
        total_frames += frame_total
        print(f"{video_path.name}: wrote {frame_total} frames")

    print(f"Done. Wrote {total_frames} frames to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
