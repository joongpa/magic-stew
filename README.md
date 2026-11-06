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

To train on a Colab GPU, open `notebooks/train_rfdetr_keypoints_colab.ipynb`
in Google Colab and run the cells in order. The notebook accepts a generated
dataset ZIP containing one or more dataset folders plus an optional
negative-images ZIP and can save checkpoints
directly to Google Drive. It bundles a snapshot of the trainer, so it doesn't
require a repository checkout or the project's `kociemba` dependency.

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
Preparation prints validation and copy counts every 1,000 images and at the end
of each source/split. Training enables RF-DETR's live batch/epoch progress bar
by default and prints the selected settings and output directory before fitting.

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

Render independent-keypoint predictions with the bbox → padded square ROI →
pose → ROI-tracking flow. The default checkpoint is the YOLO medium model in
`runs/pose/runs/yoloPoseMedium/weights/best.pt`. It predicts three geometric
landmark classes (`corner`, `corner-edge`, `corner-edge-center`) and six sticker
color classes (`white`, `yellow`, `orange`, `red`, `blue`, `green`) in one pass.
Class IDs are resolved from the checkpoint's class names. No segmentation model
or image-pixel color sampling is used.

```bash
.venv/bin/python scripts/render_yolo11n_independent_keypoints.py \
  --source /path/to/solve.mp4 \
  --initial-scramble "R U2 F'" \
  --recognize-moves \
  --save-final-video runs/recognized_solve.mp4 \
  --turn-candidate-log runs/recognized_solve_log \
  --no-display
```

Use `--initial-scramble ""` for a solved cube. Scrambles accept standard face
turns, wide turns (`Rw`, `Uw2`, `Fw'`, or lowercase aliases), and whole-cube
rotations (`x/y/z`), in the white-top/U, green-front/F frame. `--known-scramble`
is an alias. Use `--model PATH` to override the checkpoint; it must predict the
three landmark classes and six colors.

Detections are filtered by `boxes.conf` and keypoint confidence. Defaults are
`corner=0.25`, `corner-edge=0.40`, `corner-edge-center=0.25`, and `0.25` for each
color. Override any class with `--pose-class-conf red=0.35,corner-edge=0.45`.
`--min-keypoint-conf` filters point visibility (default `0.05` for landmarks);
sticker color keypoints have a minimum of `0.25` for scoring, drawing, and ROI
tracking. `--track-conf` controls ROI tracking. Neural inference runs on CPU. Move recognition uses Apple MPS and
requires an available MPS device.

Each sticker color keypoint finds its closest class-valid three- and four-landmark
groups by expanding the nearest-landmark neighborhood. Cyclic classes must match
a corner, edge, or center sticker profile. For four points, the seed score is the
RMS distance from the color point to the two diagonal midpoints. For three points,
every class-consistent parallelogram completion is checked and the best center
alignment is retained. Errors are in pixels, without division by candidate size.
Concave, degenerate, non-enclosing groups, and any quadrilateral containing
another sticker color point are rejected (including the inferred fourth corner
for three-point groups). Landmark pairs whose triangle with the seed already
encloses another sticker are pruned before enumerating combinations. Three distinct groups
are tried in center-error order, with four observations preferred on ties and
confidence breaking remaining ties; repeated color detections do not consume
extra seed slots. Failed PnP groups are skipped. Inferred fourth corners only rank
seeds; PnP uses the three actual measurements and their selected correspondence.
Three-point seeds retain the deduplicated union of SQPnP, P3P, and AP3P pose
roots, so all-landmark scoring can select branches that SQPnP alone misses.
Color points are excluded
from lattice assignments and used instead of segmentation centroids for the
existing center-assisted pose refinement and geometric turn fitting.

Lattice visibility uses the direction from each surface point to the actual
camera, including cubes far from the image center, on both CPU and GPU.
Static refinement first tries up to six PnP/rematching iterations with all
landmark observations and fixed sticker anchors. It accepts a better global
fit only when every observation lies inside the existing 5% cube-size gate;
otherwise the inlier-based refinement handles turns and outliers as before.
This condition only accepts the optional rigid proposal. Mid-turn frames remain
eligible with landmarks outside the fitting gate, and those landmarks still
contribute to the full-observation score.
The three-seed limit and translation-based tracking selector are unchanged.

Whole-cube orientation is chosen directly among the 24 proper cube symmetries
by maximizing the number of matching observed sticker colors against the known
scramble, or the committed state during move recognition. All available sticker
colors participate, including edges and corners. There is no temporal angular
continuity score or orientation continuity gate. Translation-based pose
hypothesis selection and the existing geometric refinement remain available.
Before move alternatives are evaluated, an unchanged state with less than 2/3
correct observed sticker colors triggers an orientation rescue. All 24 cube
orientations are compared using that state's colors; moves are then scored in
the best orientation using the combined geometry and color cost. Exactly 2/3
does not trigger rescue. An orientation correction does not commit a cube move.
No RGB regressions, unsupervised color grouping, calibration sample banks, or
ground-truth color-distance checks are performed. The former segmentation and
RGB plotting CLI options have been removed.

Color scoring for each move/state is:

`color_score = correct / observed_stickers_inside_projected_cube`

Only color points contained in visible projected stickers participate. Count
each observed projected sticker once. If multiple color keypoints fall in one
sticker, that sticker is correct if **any** predicted color matches its expected
color. Points outside the projected cube and unobserved stickers do not enter
the denominator. A point on a shared projected boundary belongs to its nearest
containing sticker. The minimized search cost is `color_error = 1 - color_score`,
added to the existing geometric cost. Available colors with no containing
stickers produce an invalid fit; absent color observations are neutral.

Move recognition retains partial-angle fits for each single/double layer over
`-45°` through `45°` in 6° increments, including zero, and scores the `-90°`,
zero, and `90°` sticker offsets before selecting an angle. Every imperfect
color orientation can be reconsidered against the tested physical state;
orientation corrections do not commit a cube move. Geometry projections and
observation assignments are shared across states. They are rebuilt when pose,
layer geometry, or observed positions change.

The cached video search retains its adaptive branch-and-bound rollout over ten
future frames at 30 FPS, scaled with source FPS. Each parent retains its best
hold and top ten state changes. The front half of the window contributes 30%
of the score and the back half 70%; each committed turn adds `0.01` outside
those temporal weights. The default budget is 4096 frame/state evaluations per
decision; use `--turn-search-state-budget N` or `unlimited` to change it. A
budget stop returns the best complete forecast found. The existing move-spacing
policy, state merging, root-return restriction, warm starts, and optimistic
bounds remain in place. Color bounds relax identity constraints using binary
matches, with the same projected-sticker denominator as actual state scores.

`--turning-start-frame N` forces frames **0 through N inclusive** to static
holds; move search begins at N+1. Without this flag, move recognition is
unrestricted. Search stops after the accepted current-frame state is solved
and layers are aligned, retaining the final solving move. A solved future
forecast alone does not stop current-frame search. Subsequent frames keep
rendering with the solved state.

Use `--show-colored-masks` to color observed projected stickers from the model's
classes. Color predictions are always available directly from the pose model;
there is no separate color-classification or posterior-debug step.
`--show-sticker-identities` displays physical sticker labels. The orientation
axes point toward their center colors: X+ red, Y+ white, Z+ green, with matching
arrow colors. They follow center stickers through wide turns and cube rotations.
Hide them with `--no-show-orientation-axes`. `--save PATH.png` saves the
final pose render for an image source; `--save-final-video PATH.mp4` saves the
final cached video. QuickTime outputs
receive a frame-counter track when `ffmpeg` is available.

`--turn-candidate-log OUTPUT_DIR` writes one JSONL record per video frame,
including camera inputs, geometric observations, retained turn options, state
search diagnostics, and selected moves. New records use schema version 3,
`color_scoring: "direct_match_fraction"`, and a binary `turn_color_matches`
matrix indexed by physical sticker identity and observed color keypoint.
The replay search scripts use this categorical schema; old probability logs
must be regenerated before replaying with the updated scorer.

Review alternatives with:

```bash
.venv/bin/python scripts/review_turn_candidates.py \
  --video /path/to/solve.mp4 \
  --log runs/recognized_solve_log
```

The reviewer shows tested geometry and move forecasts and lets you save
`Correct`, `Uncertain`, or `No valid pose` labels beside the log. Labels do not
modify inference results. New logs include detected color names, binary color
matches, and per-candidate correct/observed counts. The reviewer colors observed
keypoints by their predicted class and shows the correct fraction and mismatch
cost. Forced-static frames explicitly show that move search was skipped. Older
probability logs remain viewable; regenerate logs to include the new color labels.

For a full-solve search whose next branch has the fewest repeatedly incorrect
physical stickers across the entire queue, use:

```bash
.venv/bin/python scripts/search_independent_keypoint_frame.py \
  --source /path/to/solve.mp4 \
  --initial-scramble "R U2 F'" \
  --turning-start-frame 380 \
  --output-dir runs/full-solve-frame/my_solve
```

This uses the nine-class `yoloPoseMedium` checkpoint and ROI padding `0.4` by
default; `--model PATH` and `--roi-padding RATIO` override them. The independent
keypoint renderer also defaults to ROI padding `0.4`. The video pass caches landmarks, sticker color classes, and all
retained geometric options in `observations.jsonl`. It performs no local move
rollout or color calibration. Sticker color keypoints retain the `0.25`
visibility floor. Frames through the turning boundary are static; neither turn
geometry nor move alternatives are evaluated there. Source FPS is inferred
from the video, with the existing move-spacing and double-turn rules retained.

The default queue order is current `0.5 * normalized_geometry + 0.5 * color_error`,
plus a soft local-rank penalty, then greater depth and insertion order. History ordering and soft backtracking
penalties are off in this default `--queue-order frame-score` mode.

`--soft-top-k 8` prefers the top eight physical successors for each parent/frame.
Rank is determined by geometry + weighted color after move-spacing checks and
before the activity check. Equal scores at the eighth-place boundary receive
the same treatment. Multiple partial fits of the same cube successor count
once, using its best fit to determine its rank. Choices outside the top eight
receive `--outside-top-k-penalty 0.25`. They remain interleaved with the entire
queue and can still win; they are never discarded or put behind a hard tier.
The penalty applies only to the latest incoming choice, not the full path.
Use `--soft-top-k 0` to disable it.

`--require-layer-activity` is on by default. A committed state change requires
activity on the same physical axis, side and width in the immediately preceding
frame. Activity is a nonzero fitted partial angle or a committed state change,
including a change with zero residual misalignment. Direction is ignored by
this activity check, while the existing move-spacing/double-turn rules still
apply. A partial hold can supply evidence; an aligned hold clears it. The
current change supplies the second activity frame and cannot supply both frames
by itself. Static opening frames do not supply activity. Globally unobserved
bridge frames retain their no-information policy and do not rule out layers.
Use `--no-require-layer-activity` to disable the check.

Activity affects legal continuations, so the scorer keeps the best hold fit
for each distinct active layer and for alignment. These contexts stay distinct
when merging cube states. Equivalent quarter-turn successors still share their
state object; pose alternatives do not create 24 copies of each branch.

Directional color-confusion scoring is on by default. Arrows mean **expected
sticker color → predicted keypoint class**, using only these direct pairs:

| Expected | Permitted incorrect predictions |
|---|---|
| red | orange |
| orange | yellow, red |
| yellow | orange, white, green |
| white | blue, yellow |
| green | yellow |
| blue | none |

Exact matches cost zero. A listed confusion costs `0.5`
(`--color-confusion-cost`, strictly between zero and one). Any other mismatch
costs `1`; it does not reject the branch. Pairs never chain: white → orange
and orange → blue cost `1`. Reverse pairs receive reduced cost only if
explicitly listed, so blue → white costs `1`.

Each observed projected sticker contributes its **best** contained prediction:
an exact match wins over duplicate incorrect points, and a permitted confusion
wins over forbidden duplicates. Points outside the projected cube do not vote.
`color_error = (confusion_cost * permitted_wrong_stickers + unlisted_wrong_stickers) / observed_stickers`.
Counters log exact, permitted, and unlisted counts
separately; the rendered output never presents weighted credit as an exact
match count. Forced-static opening frames and unobserved bridge frames retain
their existing behavior.

Use `--no-color-confusion-scoring` for the old exact-match fraction. There is
no percentage cutoff in either mode. The optional legacy
`--color-confusion-pruning` flag restores rejection of unlisted mismatches;
it is off by default. If every branch dies, the console reports no solution.

The optional `--queue-order sticker-history` adds a graded history penalty.
Each of the 54 physical stickers keeps its last **four observations**, including
the current frame. Its penalty is `(wrong_observations_in_last_four / 4)^2`;
the sum across stickers is multiplied by `--sticker-history-weight` (default
`0.1`) and added to the fit. One through four wrong observations contribute
`0.0625`, `0.25`, `0.5625`, and `1`. The denominator stays four for shorter
histories, so errors cost something immediately without treating one early
error as four. Unobserved stickers keep their histories without adding votes.
The fixed opening poses also collect history, without performing move search
or orientation rescue. Identities persist across moves and pose rescue.
Multiple color keypoints inside one projected sticker contribute one
observation, correct if any of those colors match. Histories can recover as
older wrong observations leave the four-observation window.

The existing fit score still chooses the best pose/partial-angle fit for each
state, before incorporating its history into the queue priority:
`0.5 * normalized_geometry + 0.5 * color_error`, where
`color_error = 1 - correct / observed_stickers_inside_projected_cube` when
confusion scoring is disabled, or the weighted confusion fraction above.
Geometry averages two bounded costs: landmark error divided by the existing
0.05 cube-size tolerance and sticker-position penalty divided by its 0.05
maximum, each capped at 1. The coefficients are 25% landmarks, 25% sticker
positions, and 50% color error; permitted color confusions reduce the color
error's severity as specified above. Invalid fits remain invalid.
Landmark error and validity use the whole projected cube after applying the
candidate partial turn. Layer membership contributes diagnostic counts only;
missing support on the proposed layer cannot erase the global landmark error.
Nonfinite landmark costs remain invalid instead of becoming zero. Cached logs
containing the former zero-keypoint fallback must be regenerated; replay rejects
those stale geometry options. New observation logs record geometry scoring version 2.
All pending branches compete globally. Static exact state agreement below 2/3 triggers
orientation rescue. Frame search jointly scores all 24 orientations with the
cached partial-turn geometry and discrete offsets, selecting the lowest
geometry + color fit separately for each physical successor. History does not
select the fit. This avoids committing to an orientation from the unturned
cube while a layer is turning. Only the winning physical successors enter the
queue, not 24 copies of each branch. The prepared CPU scorer batches the color
checks and reuses its projections. `--no-joint-orientation-rescue` restores the
previous static-only orientation choice. Use `--queue-order sticker-mismatches` to restore
the earlier lexicographic binary mismatch ordering (three incorrect out of four,
requiring four observations). No queue mode imposes the old 2/3 hard color cutoff.
If no solution survives,
the console reports that the search exhausted its branches and stops.
There is no past-path mean, cumulative-cost priority, forecast, combined-score threshold,
beam, or expansion budget. Solved branches are accepted when removed from the
priority queue. In graded mode, equivalent frame/physical-sticker-arrangement/
move-spacing contexts retain the complete path with the lowest current fit +
history penalty. Better histories replace pending entries or reopen expanded
ones; the histories and move logs of different paths are never mixed. This is
an approximation because alternate histories can rank differently after future
observations, though the merged state retains the same legal continuations.
Use `--no-merge-sticker-histories` to retain separate histories. Legacy ordering
modes retain their previous history behavior. Disk overflow preserves the
ordering and the entire selected history.

Graded mode also defaults to a soft four-supported-move window
(`--soft-backtrack-moves 4`; use `0` to disable). Supported moves require
normalized geometry at most `0.5`, at least `2/3` color agreement and no increase
in history penalty on that move. Holds and weak observations earn no support.
Move progress uses a loop-erased physical-state trail modulo whole-cube
rotations, so canceling detours and rotations do not earn extra confidence.
After an expanded path has more than four supported moves, newly generated
nonbest children at or before its older boundary receive a temporary queue
penalty (`--late-fork-penalty 0.5`). Existing branches and each parent's best
continuation remain eligible without a new penalty. The penalty expires when
the fork passes the boundary recorded at its creation. No fork is deleted.
Only the selected complete path's history and support trail survive a merge.

Winning-path logs retain hexadecimal observed/correct physical-sticker masks,
the history penalty, combined queue score, diagnostic binary mismatch count,
number of stickers with four observations, and mismatched identity mask.
`config.json` supplies the history weight, 54 identity labels and their scramble
colors. Console progress, rendered video and the turn reviewer show the history
penalty and queue score. History adds no hard pruning.
Soft-window boundaries, supported move counts, and old-fork penalties are also
logged; console progress and the reviewer expose the scheduling penalty.

The full-search replay scorers apply the rescue trigger separately to each
queued state. In frame search, a triggered rescue jointly selects orientation
and partial-turn fit per physical successor, using the same 50/50 score.
Console JSON logs joint-rescue calls and color-configuration counts; winning
trials record the selected symmetry and `joint_orientation_and_turn_fit` phase.
The prepared scorer reuses
cached static sticker assignments and cube-symmetry permutations; geometry
does not need to be projected again for each state.

Reuse cached nine-class observations with `--log /path/to/observations.jsonl`
in place of `--source`; `--start-frame` and `--end-frame` select a consecutive
source interval. Pose translations and projected geometry stay cached during full search, with categorical colors
scored against each candidate state. Old RGB probability logs must be
regenerated. The output directory contains the input provenance, `config.json`,
`progress.jsonl`, `status.json`, `result.json`, and the SQLite search queue.
Creating an `OUTPUT_DIR/STOP` file requests an orderly stop.

For the experimental recovery across globally weak observations, add
`--bridge-unreliable-frames`. A frame is reliable when at least one geometric
hypothesis has whole-cube `REPROJ fit` at most `0.1` and explains
at least half of the observed landmarks within the existing inlier tolerance.
This is exactly the top-left display metric: median landmark residual plus
half its 90th percentile, divided by projected cube diameter. It is separate
from the clipped 0–1 geometry component used for 50/50 search ranking. This
decision uses geometry and coverage only, independently of cube state or
color agreement. The limits are configurable with `--bridge-max-reprojection`
and `--bridge-min-inlier-fraction`. New observation caches store the display
metric per geometric hypothesis; older caches recover it with one geometric
projection pass per frame, without model inference or state/color scoring.

Short weak gaps carry only legal discrete moves and holds. They do not run
partial-angle fitting, orientation rescue, or color pruning. Equivalent
physical states with equivalent move-spacing context merge inside the gap.
Only reliable frames contribute sticker observations and the existing 50/50
fit score. Weak frames
preserve all sticker histories and inherit the last reliable combined queue
score and fit tie-break. With
`--queue-order frame-score`, the inherited fit plus local-rank penalty remains the priority.
By default, one expansion batch covers at most two weak
frames and 0.1 seconds (`--bridge-max-frames`, `--bridge-max-seconds`), with a
minimum of one frame. Longer gaps continue in unscored batches, carrying the
last reliable queue priority until another reliable observation is reached.
No weak frame's geometry or colors narrow its alternatives, regardless of
gap length. Weak frames cannot confirm a solved goal. Forced-static frames
are never bridged.

`frame_quality.json` records every frame's global reliability decision. The
winning path retains individual source-frame moves inside bridged gaps, marked
`bridge_inferred`, with null scores instead of fabricated zero errors. The
virtual cube replays those inferred moves using the last reliable orientation;
the camera pose overlay is hidden on those frames. The feature is off by
default, so unmodified commands retain the original search behavior.

Add `--save-final-video runs/my_solve.mp4` to the full-search command to render
the accepted solution after the search finishes. `--display` opens the same
replay interactively. The output uses the independent-keypoint renderer's
landmarks, colored sticker keypoints, cube wireframe, moving-layer overlay,
red/white/green axes, and accumulated moves. A colored virtual cube beside the
camera view shows the winning state and partial turn, using the independent
renderer's stationary side panel. The overlay includes the winning frame's combined
score and correct/observed color count. Rendering replays the saved decisions;
it does not run inference or select new moves. After the solved frame, the
committed state stays fixed through the remaining video frames. An exhausted
or interrupted search does not produce a winning-path video.

To render an existing result without rerunning the search:

```bash
.venv/bin/python scripts/render_independent_keypoint_search.py \
  --result runs/full-solve-frame/my_solve \
  --save-final-video runs/full-solve-frame/my_solve/solution.mp4 \
  --turn-candidate-log runs/full-solve-frame/my_solve/rendered_turn_candidate_scores.jsonl
```

The original video path comes from the cached input metadata; `--source PATH`
overrides it (`--render-source PATH` in the search command). The automatic
render also writes `rendered_turn_candidate_scores.jsonl` in the run directory
for `review_turn_candidates.py`. This log contains the selected winning-path
trial for each frame. `--no-show-keypoints` and `--no-show-orientation-axes`
work with either render command. `--no-show-virtual-cube` hides the side panel;
it is enabled by default for both saved output and interactive display.
Saved videos preserve the source FPS, source
frame indices, and QuickTime frame counter.

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

## Hard color-recognition renders

Pass `--hard-color-recognition` to `scripts/apply_blender_cube_moves.py` when
you want sharp images that stress sticker-color recognition:

```bash
blender --background --python scripts/apply_blender_cube_moves.py -- \
  --count 1000 \
  --output-dir data/synthetic_cube_dataset_hard_color \
  --hard-color-recognition
```

This mode disables cube rotational blur and post-render Gaussian/motion blur,
while retaining rescaling, sensor noise, compression, camera variation, hand
variation, and the other normal augmentations. Lighting uses a much more
saturated distribution weighted toward yellow and blue, with occasional
red, magenta, cyan, and green casts. The mode and blur policy are recorded in
`sticker_keypoint_manifest.json` and each sample's `metadata.json` entry.

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
