#!/usr/bin/env python3
"""Compare two F-Clip checkpoints on deterministic full-frame train samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io


REPO = Path(__file__).resolve().parents[1]
FCLIP_ROOT = REPO / "F-Clip"
sys.path.insert(0, str(FCLIP_ROOT))

from FClip.config import C, M  # noqa: E402
from FClip.datasets import LineDataset  # noqa: E402
from FClip.models import MultitaskHead, hg, hgl, hr  # noqa: E402
from FClip.models.stage_1 import FClip as Stage1FClip  # noqa: E402


def build_model(device: torch.device) -> torch.nn.Module:
    if M.backbone == "stacked_hourglass":
        backbone = hg(
            depth=M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
        )
    elif M.backbone == "hourglass_lines":
        backbone = hgl(
            depth=M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
        )
    elif M.backbone == "hrnet":
        backbone = hr(
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
        )
    else:
        raise ValueError(f"Unsupported backbone: {M.backbone}")
    return Stage1FClip(backbone).to(device)


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def parsed_lines(model: torch.nn.Module, image: torch.Tensor, device: torch.device):
    with torch.no_grad():
        result = model({"image": image.to(device)}, isTest=True)
    heatmaps = result["heatmaps"]
    return (
        heatmaps["lines"][0].detach().cpu().numpy(),
        heatmaps["score"][0].detach().cpu().numpy(),
    )


def draw_lines(ax, lines, scores, color_map="magma", limit=80, min_score=0.0):
    if len(lines) == 0:
        return 0.0, 0
    keep = scores >= min_score
    lines = lines[keep][:limit]
    scores = scores[keep][: len(lines)]
    if len(lines) == 0:
        return 0.0, 0
    top = max(float(scores[0]), 1e-6)
    cmap = plt.get_cmap(color_map)
    for line, score in zip(lines, scores):
        strength = float(np.clip(score / top, 0.0, 1.0))
        ax.plot(
            [line[0, 1], line[1, 1]],
            [line[0, 0], line[1, 0]],
            color=cmap(0.25 + 0.75 * strength),
            linewidth=1.4,
            alpha=0.25 + 0.75 * strength,
        )
    return float(scores[0]), len(lines)


def draw_ground_truth(ax, lpos):
    for line in lpos:
        ax.plot(
            [line[0, 1], line[1, 1]],
            [line[0, 0], line[1, 0]],
            color="#25d366",
            linewidth=1.5,
            alpha=0.9,
        )


def style_axis(ax, image, title):
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-score", type=float, default=0.40)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out or run_dir / f"train_viz_best_vs_latest_score_{args.min_score:.2f}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    C.update(C.from_yaml(filename=str(FCLIP_ROOT / "config" / "base.yaml")))
    C.update(C.from_yaml(filename=str(run_dir / "config.yaml")))
    M.update(C.model)
    # The run config uses paths relative to F-Clip. Make the dataset explicit.
    data_dir = (FCLIP_ROOT / str(C.io.datadir)).resolve()
    C.io.datadir = str(data_dir)

    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = LineDataset(str(data_dir), split="train", dataset=str(C.io.dataname))
    # Keep the train split/file list, but remove stochastic crop augmentation so
    # both checkpoints receive identical full-frame inputs.
    dataset.crop = False

    rng = np.random.default_rng(args.seed)
    count = min(max(1, args.count), len(dataset))
    indices = np.sort(rng.choice(len(dataset), size=count, replace=False))

    best_path = run_dir / "checkpoint_best.pth.tar"
    latest_path = run_dir / "checkpoint_lastest.pth.tar"
    best = load_model(best_path, device)
    latest = load_model(latest_path, device)

    input_size = float(M.resolution * 4)
    prediction_pscale = input_size / float(M.resolution)
    # Prepared lpos values remain in F-Clip's canonical 128x128 coordinates.
    ground_truth_pscale = input_size / 128.0
    panels = []
    manifest = []

    for sample_number, index in enumerate(indices):
        image_tensor, meta, _ = dataset[int(index)]
        image_tensor = image_tensor.unsqueeze(0)
        image = io.imread(dataset._get_im_name(int(index)))[:, :, :3]
        lpos = meta["lpre"][meta["lpre_label"] == 1].numpy() * ground_truth_pscale

        best_lines, best_scores = parsed_lines(best, image_tensor, device)
        latest_lines, latest_scores = parsed_lines(latest, image_tensor, device)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        style_axis(axes[0], image, f"train #{int(index)} — ground truth")
        draw_ground_truth(axes[0], lpos)
        style_axis(axes[1], image, "best")
        best_top, best_count = draw_lines(
            axes[1], best_lines * prediction_pscale, best_scores, min_score=args.min_score
        )
        axes[1].set_title(f"best — {best_count} lines ≥ {args.min_score:.2f}", fontsize=9)
        style_axis(axes[2], image, "latest")
        latest_top, latest_count = draw_lines(
            axes[2], latest_lines * prediction_pscale, latest_scores, min_score=args.min_score
        )
        axes[2].set_title(f"latest — {latest_count} lines ≥ {args.min_score:.2f}", fontsize=9)

        output_path = out_dir / f"sample_{sample_number:03d}_train_{int(index):06d}.jpg"
        fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        panels.append(output_path)
        manifest.append(
            f"{sample_number:03d},{int(index):06d},{best_top:.6f},{latest_top:.6f},{best_count},{latest_count},{len(lpos)}"
        )

    # A compact contact sheet is convenient for scanning the whole comparison.
    columns = 1
    rows_per_sheet = 6
    sheet_paths = []
    for sheet_number, start in enumerate(range(0, len(panels), rows_per_sheet)):
        selected = panels[start : start + rows_per_sheet]
        fig, axes = plt.subplots(len(selected), columns, figsize=(15, 5 * len(selected)))
        axes = np.atleast_1d(axes)
        for ax, path in zip(axes, selected):
            ax.imshow(io.imread(path))
            ax.axis("off")
        sheet_path = out_dir / f"contact_sheet_{sheet_number:02d}.jpg"
        fig.savefig(sheet_path, dpi=110, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        sheet_paths.append(sheet_path)

    (out_dir / "manifest.csv").write_text(
        "sample,train_index,best_top_score,latest_top_score,best_lines_kept,latest_lines_kept,ground_truth_lines\n"
        + "\n".join(manifest)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(panels)} comparisons to {out_dir}")
    print("Contact sheets:")
    for path in sheet_paths:
        print(path)


if __name__ == "__main__":
    main()
