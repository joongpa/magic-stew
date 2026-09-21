# LiveRecognition

POC for reconstructing 3x3x3 Rubik's Cube solves from video.

This first slice focuses on step 1: detecting a cube-like object in a live or recorded frame and drawing a bounding box around it.

## What is implemented

- Webcam/video/image CLI
- Local SAM 3 text-prompt cube detector
- OpenCV heuristic cube candidate detector
- Bounding box overlay, with optional SAM mask overlay
- PyTorch model interface stub for replacing the heuristic detector with a learned keypoint/segmentation model later

The current detector is deliberately simple. It is useful for plumbing, overlays, data collection, and shaping the interfaces before training data exists.

## Setup

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If `torch` installation fails on your Python version, install a PyTorch build from the official selector first, then rerun the editable install:

```bash
python -m pip install -e ".[dev]"
```

## Run

By default, `cube-orient` runs headlessly and does not open a preview window.

Webcam:

```bash
cube-orient --source 0
```

Video file:

```bash
cube-orient --source /path/to/solve.mp4
```

Single image:

```bash
cube-orient --source /path/to/frame.jpg --once --save runs/frame_bbox.jpg
```

Piece graph overlay:

```bash
cube-piece-graph --source /path/to/frame.jpg --display
```

`cube-piece-graph` runs the cube bounding-box model first, runs the center/edge/corner
piece segmentation model, clips piece masks to the detected cube, and draws legal graph
edges between adjacent sticker centroids. Legal edges are center-edge and edge-corner.

Observed-color net from one image:

```bash
cube-net --source /path/to/frame.jpg --save-net runs/frame_net.png --save-graph runs/frame_graph.jpg
```

`cube-net` uses the sticker segmentations' average BGR colors directly. The net is a
view-relative preview of the observed faces.

Add `--display` if you want an OpenCV preview window. Press `q` to exit the video window.

```bash
cube-orient --source 0 --display
```

The default detector is local SAM 3:

```bash
cube-orient --source 0 --detector sam3 --prompt "3x3x3 rubik's cube"
```

The default output is bounding-box only. If you also want to see SAM's mask overlay:

```bash
cube-orient --source /path/to/frame.jpg --once --mask-overlay
```

SAM 3 weights are gated and are not auto-downloaded by Ultralytics. Request access on Meta's SAM 3 Hugging Face page, download `sam3.pt`, then put it in this project directory or pass its path:

```bash
cube-orient --source 0 --detector sam3 --model /path/to/sam3.pt
```

If you want the dependency-free bootstrap detector:

```bash
cube-orient --source 0 --detector heuristic
```

## Project direction

The intended next steps are:

1. Collect frames with bounding boxes enabled.
2. Use SAM 3 boxes to bootstrap cube detection labels.
3. Label visible corners and facelet centers after box quality is acceptable.
4. Train a PyTorch detector/keypoint model.
5. Add orientation only after detection quality is acceptable.

## Dense color-map head

`live_recognition.color_segmentation.YOLOColorMapModel` attaches a PyTorch
decoder to a trained Ultralytics whole-cube segmentation model. It captures the
multiscale features immediately before the existing YOLO segmentation head,
adds explicit RGB/chromaticity/luminance/saturation features, and returns a
seven-channel per-pixel map:

```text
white, yellow, red, orange, blue, green, none
```

The base YOLO model is frozen by default. To train the head using the generated
sticker polygons:

```bash
.venv/bin/python scripts/train_yolo_color_map_head.py \
  --source-dir data/synthetic_cube_dataset \
  --model runs/cube_whole_seg_local/runs/cube_whole_yolo11n_seg/weights/best.pt
```

The script derives a square ROI from the whole-cube labels, rasterizes the six
sticker-color polygons into dense targets, assigns uncovered pixels to `none`,
and ignores a small boundary band. Add `--unfreeze-backbone` only after the
frozen-head experiment is working.

To restore the trained head and obtain probabilities for an RGB ROI tensor:

```python
import torch

from live_recognition.color_segmentation import YOLOColorMapModel

model = YOLOColorMapModel.from_color_checkpoint(
    "runs/cube_color_map_head/best_color_map_head.pt"
).eval()
with torch.no_grad():
    probabilities = model(roi_rgb_float_0_to_1)["color_probabilities"]
```

To inspect the predictions visually, add `--color-map-model` to
`scripts/render_yolo_sticker_segmentation.py`. The output is a compact 2x2 grid
with the segmentation view, top-1 color mask, top-2 color mask, and a
distribution panel. Click any pixel inside either color mask to display the
seven probabilities, predicted class, and normalized entropy for that pixel.
