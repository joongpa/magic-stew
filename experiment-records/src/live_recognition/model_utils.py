from __future__ import annotations

from pathlib import Path


def resolve_model_path(model_name: str, description: str) -> str:
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)
    raise FileNotFoundError(f"{description} weights not found: {model_name}")


def resolve_sam3_model_path(model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)

    raise FileNotFoundError(
        f"SAM 3 weights not found: {model_name}. "
        "Download sam3.pt from the gated Meta SAM 3 Hugging Face page after access is approved, "
        "place it in this project directory, or pass its path with --model /path/to/sam3.pt."
    )


def ensure_yolo_config_dir(path: str = ".ultralytics") -> None:
    config_path = Path(path)
    config_path.mkdir(parents=True, exist_ok=True)
