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

## RF-DETR landmark and sticker keypoint training

`scripts/train_rfdetr_keypoints_local.py` fine-tunes RF-DETR Keypoint Preview
on the same independent one-keypoint YOLO pose rows. It reads class order from
the generator's `sticker_keypoint_manifest.json` (combined landmarks and six
sticker colors), or `independent_keypoint_manifest.json` (legacy three-class
landmarks). Without a manifest it reads `data.yaml`, falling back to the three
landmark classes when no metadata exists. Multiple sources must have matching
class orders. `--class-names` can explicitly supply a comma-separated class
order for custom datasets; label IDs must already match that order.

Install the optional training dependencies in the environment you will use:

```bash
python -m pip install -e ".[rfdetr]"
```

Prepare and validate without importing RF-DETR or downloading model weights:

```bash
.venv/bin/python scripts/train_rfdetr_keypoints_local.py \
  --source-dirs data/synthetic_cube_dataset \
  --extra-negative-dir data/youtube_solve_frames_hd/cropped-negatives \
  --prepare-only
```

Remove `--prepare-only` to train. `--extra-negative-dir` is optional and has
the same meaning as in the YOLO trainer: cube-free images are copied with empty
pose labels. `--pose-subdir` defaults to `pose`. The trainer validates labels
before replacing its own prepared dataset, keeps labeled images in both splits
when possible, and records source paths and negatives in
`conversion_manifest.json` alongside `dataset_summary.json`.

The default `--imgsz 288` can be changed with `--imgsz` or `--resolution` to a
positive multiple of 24, including 144 or 264. The fixed-size training defaults
disable multi-scale resizing, crop jitter, and optional augmentation, preserving
the generator's physical handedness and color labels. Dataset image sizes do
not have to match the model input size. `--device` accepts `cpu`, `mps`, `cuda`,
`cuda:N`, or a GPU index such as `0`; omission uses RF-DETR's automatic choice.
Start with `--batch 4 --grad-accum-steps 4`, lowering the physical batch if needed.
Other controls include `--epochs`, `--lr`, `--lr-encoder`, `--workers`,
`--patience` (0 disables early stopping), `--no-ema`, and an optional pretrained
checkpoint override with `--model`. To enable `--tensorboard`, install
`tensorboard` separately.

Prepared data goes to `runs/rfdetr_keypoints_local/dataset`; training outputs go
to `runs/rfdetr_keypoints_local/runs/cube_keypoints_rfdetr` with numbered names
for subsequent runs. Override these with `--workdir` and `--name`. The final
message reports `checkpoint_best_total.pth`, containing the best validation
checkpoint. No mask labels or uncertainty ellipse annotations are required.

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
  --initial-scramble "" \
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
  --initial-scramble "R U2 F'" \
  --model runs/pose/runs/independent_keypoints_v3_weights.pt \
  --combined-seg-model runs/segment/runs/combinedSegmentWeights.pt \
  --rgb-html-plot runs/sticker_colors.html \
  --no-display
```

Use `--color-samples-per-sticker` and `--color-min-samples-per-sticker` to
control the per-frame averaging and the minimum reliable mask area. Use
`--sticker-mask-erosion N` to shrink each predicted sticker mask by `N` ROI
pixels before rendering and color sampling, which avoids sampling near sticker
boundaries.
When `--show-sticker-mask` is enabled, the final output displays the
transparent red distance-transform heatmap used to visualize sticker-mask
peaks instead of the binary mask fill and boundary overlay.
Use `--show-colored-masks` to replace that heatmap in the final replay with
color predictions drawn on each projected sticker's intersection with the
sticker/cube segmentation. Each intersection is classified independently
from its pixels using the global profiles, without sticker identity or move
history priors. With `--recognize-moves`, these include the labeled-sample
calibration; otherwise the initial global profiles are used. A usable pose and
enough reliable pixels are needed for an intersection to be colored. Add
`--no-show-pnp` to hide the pose drawing while retaining the pose calculation.
That same peak-to-centroid confidence tempers mask-center influence during pose
refinement, so irregular or merged masks cannot pull the solve as strongly as
clean, centered masks.
Before pose fitting and display, sticker detections are filtered so a mask
with at least 90% of its area covered by the union of larger retained sticker
masks is removed; identical masks keep the first detection. This check uses
the raw predicted masks before `--sticker-mask-erosion`.
By default the renderer is in static-pose mode: it selects a sticker, refines
the full-cube pose from the independent keypoint assignment, and keeps the
cube symmetry continuous across video frames. A projected red/green/blue XYZ
gizmo is shown at the cube center; use `--no-show-orientation-axes` to hide it
or `--orientation-axis-length MM` to resize it. Pose recovery can use the
color-profile symmetry recovery for whole-cube orientation, followed by the
layer-turn search when move recognition is enabled.
When recovery is enabled, `--turn-candidate-log OUTPUT_DIR` writes one
JSONL record per video frame. Each record includes the complete set of tested
turn configurations, their global mean and robust scores, keypoint/heatmap
components, inlier counts, trigger diagnostics, and the selected result. It
also includes the search-time camera/pose inputs, observed keypoints/classes,
fixed assignments, and heatmap centers needed to reproject every logged pose.
New logs include a search trace: the last committed move, current-frame winner
versus sequence winner, retained geometric options, cache/batch counts, and
per-depth counts for collapsed holds, states scored, equivalent-prefix merging,
bound pruning, completed paths and queued work. Logs include future lower
bounds, incumbent updates, optimality status and examples of pruned bounds.
Inverse and state-revisit pruning are disabled; older logs may record beam or
history restrictions. Every starting candidate has a recorded fate. Up to eight best
forecasts for distinct starting candidates, plus static-first and complete
no-turn comparisons when needed, include the predicted move, actual video frame index, score components,
and weighted contribution at each step. These diagnostics are collected only
when logging is enabled; future steps are forecasts, not committed moves.
Stable trial IDs distinguish sticker offsets that share identical geometry.
Annotated candidate images are saved unless `--turn-candidate-images` points
elsewhere.
With `--recognize-moves`, pose refinement runs on every processed frame with a
usable pose; geometric turn search skips only the forced-static prefix. The turn search is not throttled by
frame interval, and it remains enabled even when pose drawing is disabled.
Unsupervised color regressions use observations from the entire video,
including turns. Ground-truth labels are a separate inspection-only sample
bank; adding them never replaces the full-video regression fits or their
sample counts. With `--turning-start-frame N`, accepted stationary samples
through frame N are eligible for labels, and calibration is finalized before
the first frame after N is processed for moves. Later holds and reversals
cannot add labels or reopen calibration. Without an explicit boundary, the
first committed move closes the inferred inspection interval permanently;
the moving frame and all later frames are excluded. If no move occurs,
accepted stationary observations from the clip are eligible. Partial turns
and unreliable fits are excluded from labeling.
After calibration, color likelihoods combine the regression distance with the
distance to up to three nearby labeled samples. Sample distance contributes
at most 30%, tapering with weak or distant sample coverage; the regression
remains the fallback for colors without suitable labeled samples.
Once calibration closes, solve-time orientation continuity is rescored with
these same labeled profiles before future move-search projections are built.
Inspection poses remain fixed. With an inferred inspection boundary, the
first committed move also remains fixed and only subsequent poses are rescored.
Without `--recognize-moves`, the default pose-symmetry option uses the same
closest-color-profile orientation recovery as the original pose pipeline. It
evaluates the 24 proper cube symmetries and uses profile likelihood plus pose
continuity to select the closest orientation before rendering.
The outright reprojection threshold and the two-sided 40 mm translation
rejection are temporarily disabled. Reprojection and translation still
contribute to pose selection, and invalid geometry remains ineligible.

Move recognition retains every partial-angle fit for each single/double layer
choice over `-45°` through `45°` in 6° increments, also testing zero. It scores
the `-90°`, zero, and `90°` sticker offsets against each candidate cube state
before selecting an angle. For each layer and offset, the best combined
geometry-plus-color angle survives to state search. This prevents a slightly
better geometry-only fit from discarding the correct partial angle before
color evidence is considered. Keypoint reprojection, sticker-mask-centroid fit, and available color
likelihoods each contribute with weight one; an impossible color fit is invalid.
For each frame/state pair, the inherited orientation is evaluated first. If
its best combined turn fit has color error **greater than 5**, move search
also evaluates the other 23 cube orientations against that physical state.
Each orientation includes the same full partial-angle and color-offset search.
The best pose/angle for each physical move survives before the usual top-10
shortlist. Geometry projections are shared across orientations; the physical
state stays in its existing coordinates, and pose corrections do not count as
cube moves. At color error 5 or below, orientation is unchanged. This fallback
also applies to future lookahead states and single-frame move fitting. Logs
record the inherited color error, tested orientation count, and selected pose
symmetry; the reviewer uses that symmetry when projecting alternative trials.
The cached video search uses adaptive branch-and-bound over the next ten
frames at 30 FPS, scaled linearly with the source rate (20 frames at 60 FPS),
with no beam width. It first scores a complete no-turn continuation.
Use the optional `--turning-start-frame 380` to force frames **0 through 380,
inclusive**, to static holds with no partial layer angles, color offsets, or
committed moves. Move search starts at frame 381. The boundary uses original
zero-based frame indices, including frames with no usable pose. Omitting the
flag leaves move recognition unrestricted. The restriction also applies to
live frame processing, and logs/reviewer identify these frames as forced static.
Each parent then keeps one best hold fit plus the top 10 state changes ranked
by combined geometry-plus-color fit (`state_score`, controlled by
`PNP_TURN_LOOKAHEAD_STATE_CHANGE_LIMIT`). This cutoff
applies at every depth; equivalent hold geometries are collapsed before expansion.
Promising prefixes are scheduled by optimistic full-window cost, with current
combined current fit breaking ties. Warm-start and greedy seeds use the same
combined-fit shortlist. At every depth, a committed move that returns
to the state at the start of the current window is pruned before shortlisting.
Holds and partial geometric offsets remain allowed. Intermediate-state revisits
and returns to states from earlier windows remain eligible. Prefixes reaching
the same cube state at the same frame with the same last-update frame have
identical future options, so a cheaper prefix dominates. State-changing frames
use a source-FPS-scaled 30-per-second spacing policy:
`ceil(source_fps / 30)` frames. At 24 or 30 FPS, different-layer moves can
occur on adjacent frames; at 60 FPS, an intervening frame is required. The one exception is a
single consecutive same-layer, same-direction continuation, which represents
the second quarter-turn of a double turn. The rule applies within a lookahead
window and across adjacent windows; holds and partial turns remain available
on the intervening frames.

For each future frame, an optimistic color fit lets every observation use its
best possible sticker identity while retaining the actual geometry, coverage
and uncertainty weighting. This relaxes physical sticker constraints and
bounds every feasible state from below, including negative color errors.
Among eligible transitions, a prefix is pruned only if dominated or if its
accumulated cost plus the remaining lower bound cannot beat the best complete solution. Unknown bounds
disable that pruning rather than assuming zero. Finishing the search proves the best sequence within the per-parent color
shortlists. Lower-ranked changes can therefore be missed. Runtime varies
with ambiguity. A default **4096 frame/state evaluation budget per decision**
limits exploration, including the current state, complete-path seeds and cache hits. Configure it with `--turn-search-state-budget 512`,
or use `--turn-search-state-budget unlimited` for exhaustive search within the
shortlists. A budget stop returns the best **complete** forecast found, rather
than comparing unfinished prefixes. It is approximate; logs and the reviewer
show `budget_exhausted`, evaluations used, the remaining lower bound and its
gap from the winning score. Unknown bounds leave the gap unavailable.

Alongside the complete no-turn challenger, the search re-scores and extends the
previous decision's winning forecast, provided its expected state matches the
actual current state. A greedy full-window rollout supplies another starting
solution. Seeds obey the same root-return restriction, per-parent shortlist
and finite-score checks, and are re-evaluated when color profiles change. They improve the incumbent before
branch-and-bound exploration; they do not force later decisions.

Projection and observation-to-face assignments are cached per frame across
state batches and lower-bound scoring. Changing pose, layer geometry, observation
centers, uncertainty weights or the assignment threshold invalidates the relevant
cache. Color likelihood changes reuse only the geometry/assignments and recompute
color costs. Past-frame caches are evicted as the window advances.

Uncached states at the same depth share GPU calls. The scheduling batch size
`PNP_TURN_LOOKAHEAD_STATE_BATCH_SIZE` (32) limits work per call, not which
states can survive. Scores use two uniform blocks: the front half contributes
30% total and the back half 70%. For an odd window, the back half contains the
extra frame. The sequence objective adds `PNP_TURN_LOOKAHEAD_MOVE_COST`
(0.01) per turn outside those temporal weights. Strong future evidence can
therefore justify a worse current fit without relative inlier/color acceptance
gates. Static-first and complete no-turn comparisons are logged separately.
Color error remains a finite observation cost with no absolute cutoff. Large
errors stay in both candidate forecasts and the no-turn comparison. A future scoring
failure invalidates its path instead of triggering evidence omission and retry;
only unavailable geometry is skipped. Nonfinite fits remain invalid, and an unusable current
frame holds without training color models. Runtime is controlled separately by
`--turn-search-state-budget`.
There is no angular speed limit on partial layer geometry. Discrete state
updates follow the 30-per-second frame-spacing cap described above. Physical
sticker projection and color sampling include the selected quarter-turn offset
on the current frame.
Move search stops automatically when the accepted current-frame state has six
uniform faces and the fitted layers are aligned. This uses the known initial
sticker colors and committed moves, and accepts any whole-cube orientation.
A solved current-frame fit skips future rollout; a solved state found only
in a future forecast does not stop the current search. The final solving move
is retained. Later frames continue rendering with the solved state and no
further move search. Logs mark the terminal decision with `stop_reason:
"cube_solved"` and `solved_frame_index`; later posed frames use search status
`stopped_solved` with zero search evaluations. Partial turns, rejected poses,
unreliable color fits and incomplete color states cannot trigger the stop.
Mask centroids use closest projected sticker faces so the accelerator can
score the complete batch without Hungarian matching.
The live output also shows the accumulated standard moves in the fixed
white-top/U and green-front/F frame.
Neural inference runs on CPU with segmentation and pose calls in parallel.
Move search runs on Apple MPS; requesting move recognition without an
available MPS device produces an error.

To review the logged alternatives interactively, run the standalone browser
reviewer against the same source video and log directory:

```bash
.venv/bin/python scripts/review_turn_candidates.py \
  --video /path/to/solve.mp4 \
  --log /path/to/turn-candidate-log
```

Open the printed local URL. Step through frames in either direction, click a
candidate to inspect its reprojected pose and geometry-plus-color score, and
save `Correct`, `Uncertain`, or `No valid pose`. The
labels are written to `turn_candidate_scores.labels.json` beside the JSONL and
do not modify the inference log. Tested options are grouped into collapsible
single-layer and double-layer sections for each of the six layer sides.
The Search decision panel compares the current-frame and sequence winners.
Choose a retained forecast to inspect its move/hold decisions and weighted
scores; frame buttons open each frame's own recorded decision. Expand the
search table to see collapsed holds, merges, bound pruning, and future lower bounds, or expand
the geometric-fit table to inspect stage one. Candidate cards show their
search fate and best retained sequence score. Older logs remain viewable,
but need a new recognition run to populate these traces.
Click a visible projected layer to select it, double-click for a double-layer
turn, then use `,` and `.` to step through the logged angles. Click the center
sticker for that face's layer or an outer row for an adjacent layer. New logs
also include a per-identity `p(observation | prior)` table for inspecting the
individual color evidence behind the aggregate score. The rendered cube also
shows the matched likelihood beside each visible sticker identity when those
values are available.

New JSONL records also contain `pose_continuity` for diagnosing whole-cube
orientation jumps. It records the input pose, previous accepted frame and
rotation, all 24 symmetry matrices and resulting rotation vectors, their
angular distances in degrees, the selected symmetry, and its runner-up margin.
A small angular margin indicates an ambiguous geometry-only choice. Alternatives
also record `angular_score`, `color_score` and `combined_score`; `selected_score`
and `runner_up_margin` describe the combined decision. The first pose after
a tracker reset uses identity with null reference distances. These diagnostics
identify their frame with `coordinate_frame`; the initial pass runs before
color orientation recovery, while calibrated solve continuity uses the
scramble-anchored frame. Calibrated records include
`color_profile_source: "inspection_labeled"`, `color_calibration_generation`
and the frozen prefix boundary in `frozen_prefix_frame`.
`selected_pose` records the later final pose. Diagnostics are also logged
before `--turning-start-frame`, when turn search is held static. Existing logs
need a new recognition run to populate this field.

The renderer first fits one global color profile set from segmentation-mask
RGB samples across the full video, without poses, sticker identities or viewing
angle weights. It then refines static geometry and scores all 24 equivalent
orientations using temporal color likelihood plus a soft angular-change cost:
`score = 0.2 * mean color NLL + angle_degrees / (90 * 30 * elapsed_seconds)`.
The color weight balances the confirmed Xuanyi orientations at inspection frame
148 and solve frame 212; the replay preserves inspection through frame 198.
Colors are compared with the latest accepted appearance of each location in
the continuity frame, with uniform priors for unseen locations. Rejected
frames cannot update this history, and appearance measurements are replaced
rather than accumulated indefinitely so layer turns can change local colors.
Without color evidence, selection falls back to angular distance. This score
has no hard rotation-speed limit. It then matches the required initial scramble
to the observed colors to establish the cube orientation. Move search uses the
same profiles and updates the cube state. The independent color debug view
remains available with `--debug-independent-color-posteriors`.

Use `--show-sticker-identities` for visible projected `C##`, `E##`, and `K##`
labels, and `--show-classified-colors` to display the known sticker colors.
Use `--save-final-video OUTPUT.mp4` to save the final cached pass; the older
`--save-colored-video` spelling remains an alias.
For `.mp4`, `.m4v`, and `.mov` outputs, the script adds a QuickTime timecode
track after rendering so QuickTime Player can enable its frame-counter display.
This requires `ffmpeg` to be available on `PATH`.

Provide the initial scramble so the color matcher uses the known sticker
state and solves the whole-cube orientation. Use `--initial-scramble ""` for
a solved cube:

```bash
.venv/bin/python scripts/render_yolo11n_independent_keypoints.py \
  --source /path/to/solve.mp4 \
  --model runs/pose/runs/independent_keypoints_v3_weights.pt \
  --combined-seg-model runs/segment/runs/combinedSegmentWeights.pt \
  --initial-scramble "R U2 F'"
```

Scrambles use standard face-turn notation (`R`, `U2`, `F'`), wide-turn
notation (`Rw`, `Uw2`, `Fw'`, or lowercase aliases such as `r`), and whole-
cube rotations (`x/y/z`). They are interpreted in the fixed frame
white-on-top/U and green-on-front/F. Measured color posteriors do not
change the known sticker colors.
`--known-scramble` is an alias for `--initial-scramble`.

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
