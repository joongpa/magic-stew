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

## Curate blurry move pairs from videos

Use `scripts/curate_move_cases.py` to build a labeled image-pair dataset from local
videos or YouTube URLs. YouTube videos are downloaded with `yt-dlp` and cached under
`data/move_case_video_cache` (720p maximum by default). Put one URL or local video path
per line in a text file to work through a batch:

```text
# videos_to_review.txt
https://www.youtube.com/watch?v=VIDEO_ID
/path/to/another_solve.mp4
```

Run the curator:

```bash
.venv/bin/python scripts/curate_move_cases.py --source-list videos_to_review.txt
```

For each video, first use the ROI selector to watch and navigate the full video while
choosing a rectangle around the cube. Drag the rectangle, then press Enter when it fits;
Space plays/pauses, `,`/`.` step frames, `j`/`l` jump a second, and `c` clears the rectangle.
Instructions and ROI coordinates appear in a separate status window, never over the video.
Saved frames are cropped to this fixed rectangle, whose coordinates are kept in metadata.
The video then opens as a clean cropped feed beside separate status and **Camera position
relative to solver** windows. Drag/click the camera marker in both diagrams: top view sets
horizontal position, side view sets vertical position. The diagrams show the camera relative to a solver facing
the cube: horizontal `0°` is straight ahead, positive is to the solver's right; vertical
`0°` is level, positive is above. Tilt is assumed to be zero. Both angles must be set before
a move can be labeled.

Use Space to play/pause, `,` and `.` to step one frame, and `j` and `l` to jump about one
second. When viewing frame `n`, press `m` to label the transition from frame `n-1` to `n`,
then enter the move in standard notation in the terminal (for example, `R`, `U2`, `F'`,
`M`, or `x`). The GUI stays live while the terminal prompt is open, and the pending label
remains tied to the frame index where you pressed `m`. Leave the prompt blank to save that
pair as a no-move transition. Frames you step past without pressing `m` are treated as no
move and are not saved. Frame 0 has no preceding transition. Press `n` to move to the next
video or `q` to quit. Case IDs continue across runs.

Each accepted pair is saved under `data/move_cases/case_NNNNNN/` with `frame_a.png`,
`frame_b.png`, and `metadata.json`. PNG preserves the decoded video frames without adding
another lossy compression step. Each case contains adjacent frames (`n-1`, `n`), so its
time delta is one video frame. Metadata records source frame numbers, timestamps, the source
URL when available, the move applied on that transition, crop rectangle, and manually
annotated horizontal/vertical camera angles. Start/end cube states are reserved as nullable
fields so they can be added later. The case folders are ignored by Git through the existing
`data/` ignore rule.

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

## Whole-cube segmentation training

The local YOLO11 whole-cube segmentation trainer defaults to the latest MCP
dataset and includes the negative ROI backgrounds in
`data/negative_examples/negative_ROI_backgrounds` as background-only samples
with empty labels:

```bash
.venv/bin/python scripts/train_yolo11n_whole_cube_seg_local.py
```

The prepared dataset and training run are written below
`runs/cube_whole_seg_local`. Use `--prepare-only` to inspect the converted
dataset without training, or `--no-negative-examples` to omit the background
images.

For joint cube-and-sticker segmentation on the combined MCP dataset, use:

```bash
.venv/bin/python scripts/train_yolo11n_cube_sticker_seg_local.py
```

This defaults to `data/mcp_dataset_v5.3`, expects class `0` to be `cube` and
class `1` to be `sticker`, and writes the prepared split and training run below
`runs/cube_sticker_seg_local`. Multiple datasets can be combined by passing
several roots after `--source-dirs` or by repeating the option. For example:

```bash
.venv/bin/python scripts/train_yolo11n_cube_sticker_seg_local.py \
  --source-dirs data/mcp_dataset_v5.3 data/another_cube_dataset \
  --extra-negative-dir data/negative_examples/negative_ROI_backgrounds \
  --device cuda
```

`--device` also accepts `cuda:0`, a GPU index such as `0`, `mps`, or `cpu`;
omitting it lets Ultralytics choose. Negative examples are included by default
from `data/youtube_solve_frames_hd/cropped-negatives`; disable them with
`--no-extra-negatives`. Mosaic augmentation is disabled by default (`--mosaic
0.0`). Use `--prepare-only` to validate and prepare the dataset without starting
YOLO training. The default segmentation `mask_ratio` is `2`; override it with
`--mask-ratio` when trading mask detail for memory and speed. Overlapping
cube/sticker instances are kept as independent masks by default; use
`--overlap-mask` only if you explicitly want Ultralytics to merge overlapping
targets. `--workers` defaults to `0` to avoid CUDA shared-memory/event errors;
for a CUDA out-of-memory failure, also lower `--batch` (for example,
`--batch 1`) and use `--no-cache`.

## Automated whole-cube failure mining

The Blender generator can be paired with the existing whole-cube model to
search its realistic scene distribution for hard negatives. Each candidate is
rendered with a ground-truth cube label, scored by model-vs-ground-truth IoU
and lower-region recall, and the hardest examples are saved with overlays and
the original scene metadata:

```bash
.venv/bin/python scripts/mine_whole_cube_failures.py \
  --candidate-count 256 \
  --keep 24 \
  --output-dir runs/whole_cube_failure_mining
```

The output contains the selected images, whole-cube labels, prediction
overlays, per-sample metadata, and condition summaries for camera viewpoint,
lighting, hand visibility, layer state, and background family. The miner only
selects from the generator's existing physically plausible variations; it does
not create arbitrary adversarial images.

## Independent cube intersection-point pose labels

The Blender generator writes independent one-keypoint pose objects by default
when run with `--pose-label-mode independent`. The three pose classes are
`corner`, `corner-edge`, and `corner-edge-center`, for 56 intersection points
total. Corner-edge points are attached to the physical edge cubies, corner
points to corner cubies, and corner-edge-center points to center cubies. Aligned
cubes retain the original 56-point set. When
`--enable-mid-turns` produces a partial layer turn, edge and center-sticker
intersections on the layer seam receive a turned-side counterpart; all other
points remain singletons and corner points are never duplicated.
Seam counterparts begin only once the partial turn exceeds 10°.

```bash
.venv/bin/python scripts/train_yolo11n_piece_seg_local.py \
  --source-dir data/synthetic_cube_dataset
```

Use `--pose-subdir` if the pose labels are stored under another directory. To
combine multiple datasets, provide multiple roots, for example:

```bash
.venv/bin/python scripts/train_yolo11n_piece_seg_local.py \
  --source-dirs data/independent_keypoints_v2 data/another_dataset
```

The trainer does not include cube-free images unless requested. To add them,
pass a directory with `--extra-negative-dir`; the images are written with empty
pose labels.

To prepare a fixed direct-correspondence target from the v3 independent
dataset, use the 37-point three-face converter:

```bash
.venv/bin/python scripts/prepare_three_face_correspondence_dataset.py
```

The output is written to
`data/independent_keypoints_dataset_v3_three_face/`. It maps the three
camera-facing source faces into a canonical U/F/R frame, emits the 37 unique
boundary-lattice points in face-major row-major order, uses visibility `0` for
blank observations, and inverse-applies recorded partial turns before lattice
assignment. The converter tolerates the currently truncated final v3 metadata
record and skips its orphan image.

To inspect the generated correspondence labels on their original images,
render selected samples with the point indices overlaid:

```bash
.venv/bin/python scripts/render_three_face_correspondence.py \
  --stems 000000 000001 000349 \
  --show-blank \
  --output-dir runs/three_face_correspondence_renders
```

By default the viewer renders the first sample only; use `--all` to render
the full dataset or `--show` to step through the rendered images in an
OpenCV window. Visible points are colored by canonical face (`U`, `F`, or
`R`) and annotated with their fixed correspondence index. Blank indices are
listed in the lower-left legend when `--show-blank` is enabled.

Train the separate 37-point correspondence model with the symmetry-aware
criterion:

```bash
.venv/bin/python scripts/train_yolo11n_three_face_correspondence_local.py \
  --source-dir data/independent_keypoints_dataset_v3_three_face \
  --workdir runs/three_face_correspondence_pose_local \
  --device mps
```

The model uses the standard YOLO11 pose backbone and a 37-point pose head.
Its loss derives the proper rotations that preserve the observed face subset:
four in-plane relabelings for one visible face, two relabelings for two
adjacent visible faces, and three cyclic relabelings for all three faces. It
uses face-exclusive points to identify that subset, so shared seams such as
the F face's indices 12–15 are not incorrectly counted as evidence for both
faces. If only shared points remain, it uses a conservative union of valid
relabelings. It routes gradients through the lowest-cost valid relabeling for
each sample. Blank points are handled through the visibility term and do not
contribute coordinate error. Use `--prepare-only` to inspect the generated
training layout without starting training.

The Blender generator uses an 8-degree minimum face-to-camera angle by
default, so nearly edge-on faces do not contribute weak, clustered keypoints.
It also omits independent points occluded by another object by default; those
points have no direct visual evidence and can otherwise teach the detector to
make confident guesses. To reproduce the older occluded-target behavior, pass
`--include-occluded-independent-keypoints` when generating the dataset. These
changes affect labels only, so the model must be retrained on regenerated data.

Render independent-keypoint model predictions with the same bbox -> padded
square ROI -> pose -> ROI-tracking flow used by the live pipeline:

```bash
.venv/bin/python scripts/render_yolo11n_independent_keypoints.py \
  --source data/independent_keypoints_dataset_v2/images/000000.png \
  --model runs/pose/runs/independent_keypoint_pose_v2/runs/cube_piece_keypoint_yolo11n_pose/weights/best.pt \
  --save runs/independent_keypoints_render.png \
  --no-display
```

The renderer filters detections using per-class `boxes.conf` thresholds. The
defaults are `corner=0.25`, `corner-edge=0.30`, and
`corner-edge-center=0.25`; override them with, for example,
`--pose-class-conf corner=0.30,corner-edge=0.40,corner-edge-center=0.30`.
`--min-keypoint-conf` remains an optional point-visibility filter, and
`--track-conf` controls the minimum point-visibility score for ROI tracking.

To collect grouped sticker colors from a video, add `--rgb-html-plot` and use
the combined cube/sticker segmentation model. The renderer projects all 54
canonical stickers from the tracked pose, intersects each projected sticker
with the predicted sticker/cube mask, averages up to 128 reliable pixels per
sticker per frame, and omits stickers with no sampleable pixels. The plot has
stable center (`C`), edge (`E`), and corner (`K`) traces, plus a normalized RGB
companion file:

```bash
.venv/bin/python scripts/render_yolo11n_independent_keypoints.py \
  --source /path/to/solve.mp4 \
  --model runs/pose/runs/independent_keypoints_v3_weights.pt \
  --combined-seg-model runs/segment/runs/combinedSegmentWeights.pt \
  --rgb-html-plot runs/sticker_colors.html \
  --color-plot-frames 180 \
  --no-display
```

Use `--color-samples-per-sticker` and `--color-min-samples-per-sticker` to
control the per-frame averaging and the minimum reliable mask area. Use
`--sticker-mask-erosion N` to shrink each predicted sticker mask by `N` ROI
pixels before rendering and color sampling, which avoids sampling near sticker
boundaries.
By default the renderer is in static-pose mode: it selects a sticker, refines
the full-cube pose from the independent keypoint assignment, and keeps the
cube symmetry continuous across video frames. A projected red/green/blue XYZ
gizmo is shown at the cube center; use `--no-show-orientation-axes` to hide it
or `--orientation-axis-length MM` to resize it. Layer-turn recovery and the
motion graph are disabled until explicitly enabled with `--recognize-moves`.
When move recognition is enabled, `--turn-candidate-log OUTPUT_DIR` writes one
JSONL record per video frame. Each record includes the complete set of tested
turn configurations, their global mean and robust scores, keypoint/heatmap
components, inlier counts, trigger diagnostics, and the selected result. It
also saves annotated candidate images unless `--turn-candidate-images` points
elsewhere.
With `--recognize-moves`, pose refinement and geometric turn search run on
every pose frame.
During the initial pose-hypothesis pass, sticker segmentation adds a second
refinement step after the keypoint assignment/PnP refinement. Its objective
compares that pose against the collective projected sticker wireframe and mask boundaries:
both directions use robust soft distance, and transverse crossings add a
finite penalty. Missing internal boundaries, including masks that merge
same-color stickers, are tolerated rather than treated as a hard failure.
For a debugging overlay that labels only the currently visible projected
stickers, add `--show-sticker-identities`; the labels use the stable `C##`,
`E##`, and `K##` identity numbers used by the HTML plot.
To classify center colors from the collected traces, classify the remaining
stickers against those center profiles, and replay the source in a second pass
with visible projected stickers colored, add `--show-classified-colors` (the
older `--show-classified-stickers` and `--show-classified-centers` spellings
remain supported). The standard
color scheme is evaluated over its 24 legal cube rotations in OKLab, while
sticker profiles use two-segment RGB regressions whose inflection reaches 255
in at least one channel; those regressions are compared in OKLab subject to
legal edge/corner constraints, including unique use of all 12 edge pieces and
all 8 corner pieces. Use `--save-final-video OUTPUT.mp4` if the final cached
pass should also be written to a separate video. This saves the pose output
even when color classification is disabled; the older
`--save-colored-video OUTPUT.mp4` spelling remains accepted as an alias.
For `.mp4`, `.m4v`, and `.mov` outputs, the script adds a QuickTime timecode
track after rendering so QuickTime Player can enable its frame-counter display.
This requires `ffmpeg` to be available on `PATH`.

For ground-truth overlays on the prepared dataset, use
`scripts/render_independent_keypoints.py`:

```bash
.venv/bin/python scripts/render_independent_keypoints.py \
  --dataset data/independent_keypoints_dataset_v2 \
  --stems 000000 005000 \
  --output-dir runs/independent_keypoint_ground_truth
```

## Ordered cube pose labels

The ordered 26-point pose target represents one rigid cube transform. When
`scripts/apply_blender_cube_moves.py` is run with `--enable-mid-turns`,
mid-turn and corner-cut images receive layer-aware keypoints: each point stays
attached to its owning cubie while that cubie is transformed by the partial
turn. Empty pose files remain supported for older or deliberately excluded
samples, and the pose trainer skips those files when building training pairs.
Pass `--pose-label-mode ordered` to the Blender generator when creating this
26-point target.

## F-Clip cube-wireframe training

The MCP dataset also contains cube wireframe endpoint labels. Prepare those
labels in F-Clip's processed format at 256x256, including the same negative ROI
images as all-zero line targets:

```bash
.venv/bin/python scripts/prepare_fclip_cube_dataset.py
```

Then train the corrected 256x256 F-Clip HG1-D2 configuration. It starts from
the official HG1-D2 weights, disables random crop augmentation, strengthens
negative heatmap supervision, and uses resolution 64. The trainer accepts MPS on Apple
Silicon, CUDA, or CPU through `--device auto`:

```bash
cd F-Clip
../.venv/bin/python train.py \
  --device auto \
  --datadir data/cube_wireframe_mcp_v4_256 \
  --logdir ../runs/fclip_cube \
  --num-workers 0 \
  -i cube_mcp_HG1_D2_256 \
  config/fclip_cube_256.yaml
```

The best checkpoint is written under `runs/fclip_cube/` in the timestamped
training directory. Use another `config/fclip_*.yaml` file to select a larger
F-Clip variant.

## Dense color-map head

`live_recognition.color_segmentation.YOLOColorMapModel` attaches a PyTorch
decoder to a trained Ultralytics whole-cube segmentation model. It captures the
multiscale features immediately before the existing YOLO segmentation head,
adds explicit RGB/chromaticity/luminance/saturation/OkLab features, and returns
a seven-channel per-pixel map:

```text
white, yellow, red, orange, blue, green, none
```

The base YOLO model is frozen by default. To train the head using the generated
sticker polygons, with class-balanced focal loss and per-color validation
metrics. By default, `none` pixels have zero loss weight because the runtime
segmentation mask is expected to remove most background pixels; pass a small
`--none-weight` if weak background supervision is useful:

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

For the combined cube/sticker model (`0: cube`, `1: sticker`), use
`--segmentation-type combined`. Its display and ROI mask are the pixelwise
intersection of the predicted cube and sticker masks:

```bash
.venv/bin/python scripts/render_yolo_sticker_segmentation.py \
  --source path/to/cube.png \
  --segmentation-type combined
```
