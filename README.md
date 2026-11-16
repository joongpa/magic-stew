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

## Synthetic solve-station timers

`apply_blender_cube_moves.py` selects one of five reference-based timer meshes
per scene: Stackmat Gen 2, Gen 3 Pro, Gen 4, Gen 5, or GAN Timer 2. All retain
the previous timer's approximately 7.84-unit width. Their molded housings,
sensor artwork, buttons, logos, and LCDs are built into
`syntheticAssets/SolveStationAssets.blend`; generation needs no Downloads files
or external timer textures. The timer colors stay fixed to each reference design,
and the selected model is recorded in the solve-station metadata's `timer` field.

To rebuild the library after editing `scripts/stackmat_timer_assets.py`:

```bash
Blender --background --python scripts/build_solve_station_assets.py -- \
  --output syntheticAssets/SolveStationAssets.blend
Blender --background --python tests/blender_stackmat_timer_assets.py
```

The reference filenames are `gen2.jpg`, `gen3.webp`, `gen4.jpeg`, `gen5.webp`,
and `images.jpeg` (GAN Timer 2) from the local `Downloads/solveVideos/timers`
folder. The saved library shows Gen 5 by default; the generator removes unused
variants before computing placement bounds.

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
`runs/pose/runs/26m-pose/weights/best.pt`. It predicts three geometric
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

The completed staged pose network is available alongside the manual fitter:

```bash
python scripts/render_yolo11n_independent_keypoints.py \
  --source /path/to/solve.mp4 --initial-scramble "" \
  --pose-method staged --staged-model runs/staged_cube_pose/weights/best.pt \
  --device 0 --save-final-video runs/staged_pose.mp4 --no-display
```

`--pose-method manual` remains the default. The staged checkpoint supplies its
own independent detector, static head, and articulation head in one RGB pass;
`--model` selects the independent detector only for manual fitting. Staged
inference uses the checkpoint's training resolution (128px for this run),
regardless of the display `--view-size` or manual detector `--imgsz`. The same
bbox detector, ROI tracking, keypoint thresholds, overlays, and cached video
outputs are used by both paths. Set `--device cpu`, `--device mps`, or a CUDA
index such as `--device 0` for network inference.

The staged path converts its predicted camera and slab angles analytically;
it skips PnP, exhaustive geometry sweeps, numerical refinement, and pose
smoothing. Center-color and temporal symmetry matching still choose an
equivalent cube frame. The existing `--recognize-moves` state search remains
optional and uses the predicted shape; the network does not itself output a
move sequence or a calibrated overall pose confidence.

The standalone `search_independent_keypoint_*.py` tools that accept `--source`
also accept `--pose-method staged --staged-model PATH`. Their
`--inference-device` selects the network device separately from the search
scorer's `--device`. Existing `--log` inputs reuse their saved geometry.

Detections are filtered by pose-model object scores (`boxes.conf`) and keypoint
presence scores (`keypoints.conf`). Object-score defaults are
`corner=0.25`, `corner-edge=0.40`, `corner-edge-center=0.25`, and `0.90` for each
color. Override any class with `--pose-class-conf red=0.35,corner-edge=0.45`.
`--min-keypoint-conf` filters keypoint presence (default `0.05` for landmarks);
sticker color keypoints have a minimum of `0.25` for scoring, drawing, and ROI
tracking. `--track-conf` controls ROI tracking. Neural inference defaults to CPU;
the prepared full-search scorer runs on CPU.

The cached pose pass exhaustively scores every 15-degree slab-angle combination
on all three axes using shared slab projections and visibility tables. It then
refines diverse candidates for each axis and active-slab combination, keeping
inactive outer slabs fixed. The best unrefined sweep results also remain eligible.
The middle angle becomes part of the rigid camera rotation; the two outer offsets
are measured relative to it. Five distinct pose/angle solutions are retained,
including fits less than 0.01 above the fifth score, with their own cached
landmark/sticker assignments. The optional temporal pass runs the same sweep
and refinement on the previous and next cameras from an immutable pose snapshot;
it preserves the existing candidates without repeating their fits. Orientation
continuity and center-color matching run afterward. State exploration reuses
the projection bank. Use `--no-pose-smoothing` to skip the neighboring cameras, or
`--no-joint-pose-geometry` to compare with the older rigid-pose pipeline.
The original geometry score remains in use. A recovery pass
only runs when both observation sets are sparse (fewer than 12 landmarks and
10 stickers), with either a depth jump beyond three prior sigmas or at least
two landmarks outside the cube volume supported by the depth prior.
It compares the original candidates with depth-constrained
rigid and articulated proposals using capped residuals, pose priors, and a cost
for extra slab parameters. Original candidates are saved in recovery diagnostics.
Recovery runs after the baseline trajectory and orientation passes and cannot
change neighboring frames. Use `--no-joint-depth-fallback` for the baseline alone.

The established joint fitter caches repeated angle-dependent slab geometry,
batches landmark/sticker visibility rays, and reuses middle-angle camera
rotations within sweeps. Its exhaustive grid, refinement budgets, scores,
candidate order, diversity and cutoff ties are preserved. Two interleaved
benchmarks across all five reference clips measured 9.3% and 11.4% aggregate
reductions in joint-geometry/temporal-neighbor fitting time, with byte-identical
complete candidate banks on all 15 sampled workloads. Individual timings vary;
this is not an end-to-end throughput measurement or an accuracy improvement.
See `runs/pose-fitting-mechanical-speed/README.md` for per-stage results, the
saved pre-change baseline and the repeatable benchmark command.

The opt-in `--geometry-sweep-device mps` batches the full exhaustive sweep on
Apple GPUs; continuous refinement remains on CPU. The default is `cpu`.
A five-repeat test of 60 sweep and 25 fitting workloads across all five
references reduced sweep time by 26.4% and fitting time by 5.4% on an M2.
24/25 complete candidate banks were byte-identical; the remaining Nahm case
had a wireframe difference below 1% of cube diameter. These are sampled cached
workloads, not complete-video throughput or whole-video accuracy validation.
See `runs/gpu-exhaustive-sweep-mps-v3/README.md` for timings, quality limits,
parallelism and reproduction. The cached geometry renderer accepts the same flag.

Experimental batched Metal refinement is available with
`--geometry-refinement-device mps`, together with `--geometry-sweep-device mps`.
A clean five-repeat benchmark of the same 25 fitting workloads across all five
references measured 6.71× faster refinement and 41.2% less fitting time relative
to GPU sweeping with CPU refinement. All sampled Metal banks retained a geometry
within 1% of cube diameter of the saved selected reference. Banks and cutoff ties
can change because the GPU uses a projected LM solver in float32; CPU final
scoring and the five-plus-ties policy remain in use. The shown Xuanyi Metal fit
was visually confirmed correct by the user. CPU refinement remains the default.
See `runs/gpu-refine-mps-metal-final/README.md` for quality limits, comparisons,
timing scope and reproduction. Static holds and depth recovery remain on CPU.

A separate buffered partial-first experiment is implemented in
`scripts/independent_keypoint_deferred_pose.py`. It generates rigid candidates
from subsets of current detections, uses a one-frame buffer to rank compatible
poses, and runs one all-axis articulated GPU stage with a global refinement
budget. The default fitter is unchanged. A 604-frame five-reference comparison
reduced measured pose work by 56.4%, with mixed quality: best-scored reference
wire agreement improved overall, but retained-bank reference coverage lost two
frames. See `runs/deferred-pose-five-reference-v1/README.md` for the common-input
control, timing limits and regression-only wireframe review. This is experimental.

For a pose-only video, omit `--recognize-moves` and add `--wireframe-only`.
To recompute joint poses from previously filtered observations without running
inference or move search:

```bash
python scripts/render_independent_keypoint_geometry.py \
  --log runs/previous-run/observations.jsonl \
  --output-dir runs/joint-pose-wireframe
```

The log's `.meta.json` supplies the source video, scramble, FPS, and static-frame
boundary. Add `--render-refined` when the input is already a joint-fit log to
render its saved geometries without fitting again.

An experimental previous-pose benchmark reuses already filtered observations
from the saved Matty, Nahm, Xuanyi, Tommy, and Yiheng runs:

```bash
.venv/bin/python scripts/benchmark_temporal_pose_reuse.py \
  --adaptive --rollout --samples 40 \
  --output-dir runs/temporal-pose-experiment
```

It narrows numerical pose/angle bounds after low-error fits, then expands through
wider local fitting, a 49-row-per-axis outer-angle search, and the existing
exhaustive fitter. Early acceptance uses the displayed `REPROJ fit` metric
(`median + 0.5*p90` landmark error divided by projected cube diameter), with a
default limit of `0.025`, plus a separate balanced landmark/sticker geometry-cost
limit of `0.15`. Override these with `--reprojection-threshold` and
`--geometry-threshold`. These are fit-error gates, not detector confidence gates.
Numerical rotation-vector bounds are not physical angular-travel limits;
quarter-turn angle wrapping and cube symmetry still need physical bookkeeping.

One-step trials start from the saved previous-frame pose; rollouts bootstrap once
and then feed back their own outputs. Current saved poses are evaluation references
only. The JSON report includes timing, search stages, unresolved quality gates,
and differences from the saved geometry costs, which are not ground-truth accuracy.
Detection, initialization, move search, and future-frame smoothing are excluded.
The production renderer's default fitting path is unchanged. Matty stops at frame
560 to omit the solved tail. Use `--cases yiheng` to run a single clip.

To examine the core-motion prior on conservative middle-of-solve windows:

```bash
MPLCONFIGDIR=/private/tmp/live-recognition-mpl .venv/bin/python scripts/analyze_core_motion.py \
  --settle-seconds .75 --tail-seconds .5 \
  --output-dir runs/core-motion-analysis
```

The analysis removes startup/ending transitions, compares stationary, velocity,
and mean-reverting predictions, and chooses prediction parameters on the other
four clips for each held-out clip. These trimmed windows are a sensitivity test,
not annotated solve boundaries. Core translations come from fitted references,
so their short-term noise must not be interpreted as measured physical velocity.

The separate five-geometry physics prototype keeps an independent causal core
history for every retained hypothesis and shares projection models across them:

```bash
.venv/bin/python scripts/benchmark_physics_pose_tracker.py \
  --iterations 3 --reversion 8 --samples 20 --rollout \
  --output-dir runs/physics-five-bank-experiment
```

It learns a recent working-position anchor, rather than pulling toward the
pickup position. A weak exponential return prior, rotation/angle prediction,
and current-observation PnP proposals initialize short analytic-Jacobian fits.
The existing exact geometry score and diversity test admit five distinct
geometries, with a hard beam cap of five (no extra score ties). The report checks
the complete retained bank against saved reference projections as well as the
best geometry cost. Use `--reversion 0` for the otherwise identical no-return
control. No current-frame saved poses enter tracker updates, and only the
initial five-hypothesis bootstrap reuses saved reference fits. This prototype
does not enable a new renderer mode or perform move search.

To test continuous next-frame motion prediction with an adaptive local search:

```bash
.venv/bin/python scripts/benchmark_physics_pose_tracker.py \
  --motion-prediction --iterations 2 --max-update-fps 30 --samples 4 --rollout \
  --output-dir runs/motion-prediction-repeat
```

`scripts/independent_keypoint_motion_prediction.py` maintains a separate damped
angular velocity, near-stationary core prediction, layer-turn velocity and
recent prediction-error scales for each of the five hypotheses. Rotation uses
continuous SO(3) increments; it has no preferred quarter-turn orientations.
The angular predictor caps speed at 2700 degrees/second (90 degrees at 30 fps).
Its recent prediction errors determine translation, rotation and layer-angle
search bounds. Current landmarks can propose a bounded PnP correction. Poor
geometry scores or reprojection fits trigger ordinary wider recovery probes.
New layer-turn axes remain possible, and all outputs pass the existing exact
geometry score and diversity filter with a hard cap of five.

`--prediction-damping 0` disables velocity extrapolation while keeping local
search, recent-error bounds and the weak core-position anchor. This ablation
separates the value of angular momentum from the value of restricting the
search. `--max-update-fps 30` samples actual observations and preserves their
original timestamps; zero keeps all available observations. The tracker uses
neither current saved reference poses nor future frames. Like the other
prototype, it bootstraps from a cached reference geometry bank and does not
evaluate cold-start acquisition or full move reconstruction.

Reports are in `runs/motion-prediction-v1`,
`runs/motion-prediction-control-2iter`, `runs/motion-prediction-no-velocity`,
`runs/motion-prediction-30fps-2iter`, and
`runs/motion-prediction-30fps-control`. The first three use every available
reference observation; the final pair use approximately 30 updates/second
(Xuanyi remains at its native 24). These are opt-in experiments. Read
`runs/motion-prediction-30fps-2iter/README.md` for results and limitations.

To test a separate static best-slab search followed by fresh articulation:

**Failed quality experiment:** visual inspection shows severe startup/recovery
failures and wrong orientation/articulation even when core displacement is
accurate. Partial point-projection scores overstated complete wireframe quality.
This prototype should not replace the established pose fitter. Failure audits
are in `runs/slab-failure-audit-full` and `runs/slab-failure-audit-mid`; they
separate translation errors from complete wireframe errors and use explicitly
reference-only interventions to diagnose premature axis/anchor selection.

The opt-in `--slab-all-axes` correctness control removes the premature turn-axis
restriction and refines every available zero-angle anchor of retained coarse
geometries. It does not predict articulation. On the same mid-solve windows,
whole-wireframe agreement improves from 179/417 to 252/417 frames under a 2%
symmetric p90 edge-distance diagnostic. This remains inadequate and costs about
235–446 ms/update. It is not a replacement or a real-time solution. Results are
in `runs/slab-hierarchy-v5-all-axes`, with whole-wireframe audits in
`runs/slab-failure-audit-v5` and exact replay videos in
`runs/slab-hierarchy-v5-wireframes`. These changes have not established reliable
full-start solve tracking. Future benchmark reports now include complete
wireframe reference comparisons after update timing, separately from detected
landmark reprojection and from best-of-five projection coverage.

```bash
.venv/bin/python scripts/benchmark_physics_pose_tracker.py \
  --slab-hierarchy --static-beam 8 --iterations 3 --max-update-fps 30 \
  --samples 4 --rollout --output-dir runs/slab-hierarchy-repeat
```

`scripts/independent_keypoint_slab_search.py` interprets a useful static pose as
one supported rigid slab, on any of the three axes. It scores that slab against
the current observations without requiring the other slabs to remain static.
Mutual one-to-one matches, landmark classes, sticker support and noncollinear
image support prevent a few duplicate detections from claiming a good slab.
Coarse orientation probes surround the previous hypotheses' predicted cameras;
12-degree and 4-degree local stencils plus static-only PnP refine a diverse
shortlist. Nearly identical prior cameras share a static search.

Only after static ranking does the tracker independently search the two other
slab angles on a 15-degree grid. It never predicts articulation or reads the
previous angle values to seed/restrict this search. Continuous angle refinement
keeps the supported slab's rigid pose fixed; folding the middle-slab angle into
the output camera is a coordinate conversion. Complete articulated scores rank
the candidates, and the existing exact geometry/diversity policy retains five
distinct outputs. The full coarse articulation pool supplies alternatives if
continuous fits converge to the same solution.

A good partial slab score alone never marks an update successful. The final
full-observation robust reprojection fit must be at most 0.035 cube diameters,
and the balanced geometry cost at most 0.2. Poor full fits may remain uncertain
proposals for the next search, but cannot train angular/translation velocity
or the trusted working-position anchor. This avoids freezing the search after
a temporary failure while distinguishing proposals from accepted poses.

The final experiment and earlier variants are recorded under
`runs/slab-hierarchy-v4-uncertain-proposals` and `runs/slab-hierarchy-v1-complete`.
The intermediate `v2-ranked` variant is rejected because pruning sometimes left
fewer than five geometries; `v3-gated` retained five but holding only old accepted
poses harmed recovery. The final experiment has a dedicated regression check
for five distinct outputs with only one supported static camera. These modes
remain opt-in and do not change the default renderer or perform move search.

Render all five solve windows with the final reference on the left and a fresh
slab-first rollout on the right:

```bash
.venv/bin/python scripts/render_slab_hierarchy_wireframes.py \
  --output-dir runs/slab-hierarchy-wireframes-repeat
```

Use `--report runs/slab-hierarchy-v4-uncertain-proposals/report.json` to replay
the exact saved benchmark windows instead of rerunning full solves. The outputs
include five individual MP4s, a combined video, previews and per-frame banks.
Green passes the full-fit gate; orange fails it. These are fit-status colors,
not detector confidence. Full-start videos in
`runs/slab-hierarchy-full-solve-wireframes` expose much weaker startup/recovery
on Matty, Nahm and Xuanyi than the later-initialized benchmark. The benchmark
results therefore do not establish reliable complete-solve tracking.

To test color-supported orientation tracking on the same five-solve benchmark:

```bash
.venv/bin/python scripts/benchmark_physics_pose_tracker.py \
  --color-tracking --iterations 3 --reversion 8 --samples 8 --rollout \
  --output-dir runs/color-supported-pose-experiment
```

Detected sticker colors gate one-to-one matches between consecutive frames.
Sparse image motion in a common ROI must pass a forward/backward consistency
check; ambiguous same-color matches are dropped. Occlusion ends a track, and
reacquisition creates a new ID. New face-center labels additionally require an
observed four-inner-landmark profile around the sticker and a good geometry fit;
these checks can still admit incorrect labels. Confirmed face-center tracks keep their physical
association separately for each of the five pose hypotheses. Ordinary sticker
tracks provide robust motion proposals, while center tracks can add soft
orientation constraints without treating outer face turns as core motion.

The tracker uses a smaller fitting budget only when enough nonplanar inliers
support the core across slabs and agree with two tracked centers. Poor geometric
fits restore ordinary proposals without tracked-color constraints. Use
`--no-color-fast-lane` to retain the normal fitting budget throughout. The
quarter-turn prior is independently opt-in and is off in this color experiment.

The benchmark defaults to final selected solution paths. It evaluates sticker
associations and the physical center-color orientation after the update; final
reference identities and future poses never enter the tracker. Reported match
precision includes only observations assigned an evaluable reference sticker.
Color-oriented estimates require two nonparallel, distinct center colors and
are marked unavailable otherwise. These are supporting cues, not guaranteed
identities or calibrated detector confidence scores. Video decoding and initial
bootstrap are excluded from timings; image tracking, motion proposals, fitting,
and maintaining five distinct geometries are included.

The five-reference color experiment is recorded in
`runs/color-tracks-validated-centers-v6/report.json`. Its sequential rollouts
retained five geometries on every frame, with about 3–7 color matches per frame
and about 99% agreement on evaluable sticker pairs. This is association accuracy,
not pose accuracy. Geometry agreement improved on Nahm and Tommy, regressed on
Matty and Yiheng, and was unchanged on Xuanyi. Pose updates took about 60–67 ms,
including about 3–5 ms for tracking and color proposals. Center-label errors on
Yiheng could still produce almost 180-degree orientation errors. The prototype
is opt-in; these results do not establish a reliable orientation anchor or a
30-fps fitting path. See that run's `README.md` for the comparison and limits.

Analyze orientation modes from completed, color-matched solution paths with:

```bash
MPLCONFIGDIR=/private/tmp/live-recognition-mpl .venv/bin/python scripts/analyze_orientation_motion.py \
  --output-dir runs/orientation-analysis
```

The default reads the final selected geometry and saved orientation correction
from each solved path, rather than assuming pose-bank option zero was selected.
Recorded color-frame rotations remain intact. A separate quarter-turn mode
analysis measures offsets from 24 preferred orientations without treating a
change between those modes as a pose error. It also tests causal forecasts with
parameters selected on the other four clips. The left plot shows rotation
between consecutive selected poses, not departure from a presumed fixed target.
Use `--source bank-rank0` only for comparison with the earlier candidate-bank
analysis; its poses are not the final selected reference trajectory.

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

Whole-cube orientation is selected during pose preparation among the 24 proper
cube symmetries. The score combines angular change from the previous accepted
pose with 10 times the mismatch fraction of observed center colors. A 90°
change over a 1/30-second interval contributes one point to the angular score,
with the scale adjusted for elapsed time. The first pose uses center colors to
anchor its orientation, and frames without observed centers use continuity.
When the committed cube state is available, expected center locations follow
its slice and wide turns. Orientation continuity runs in frame order after
pose refinement and trajectory smoothing. Move search scores states using
each frame's supplied pose.
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

With `--recognize-moves`, the cached video renderer calls the same
full-solve search as the standalone scripts. It admits 3, then 5, then 8
children per parent, keeps the eight-frame color mismatch history, and
handles turn handoffs. It searches bidirectionally when it finds a flat solved
anchor, and uses forward full search when it cannot find one. Override the
automatic anchor with `--solved-anchor-frame N`.

`--turning-start-frame N` keeps frames **0 through N inclusive** static.
By default, the renderer tries both parities at 50 FPS or higher and uses all
frames below 50 FPS. Set `--frame-parity` to `all`, `even`, `odd`, or `both`
to override. Each pass has a 300-second search budget. Use
`--search-seconds`, `--max-explored-states`, or `--branch-stage-expansions`
to tune the limits. `--turn-search-state-budget` is an alias for the per-pass
`--max-explored-states` limit. The accepted path replays on the original video clock.
Unsampled frames hold the last committed state, and the solved tail stays solved.

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
including camera inputs, retained turn geometry, and steps from the accepted
full-search path. Records use schema version 3 and retain the binary
`turn_color_matches` matrix indexed by physical sticker identity and observed
color keypoint. Selected steps also record the cost of allowed color confusions.
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

To search the even and odd source frames independently:

```bash
.venv/bin/python scripts/search_independent_keypoint_frame.py \
  --source /path/to/solve.mp4 \
  --initial-scramble "R U2 F'" \
  --turning-start-frame 380 \
  --output-dir runs/full-solve-frame/my_solve
```

The two passes run sequentially at half the input FPS and share a single
observation cache. With a full-rate cache, their paths retain the original
source frame numbers. Use
`--frame-parity odd` or `--frame-parity even` for a single pass, or
`--frame-parity all` for a full-rate pass. Supply `--log PATH` in place of
`--source` to reuse a nine-class observation cache. Older parity logs with
`original_source_frame` fields are already sampled and keep their stored FPS.

Each pass takes the lowest current-frame score from its priority queue.
The score is `0.5 * normalized_geometry + 0.5 * color_error`. Geometry uses
the cached landmark and sticker-position fit across the projected cube.
Color error averages the best detection cost for each observed projected
sticker. Exact matches cost zero and mismatches cost one. For `26m-pose`,
white → yellow, yellow → orange or white, orange → red, and red → orange
cost 0.5. The arrows go from expected color to detected color. Pose and partial
turn options compete using the same score. An identical cube state at the
same frame keeps its best fit. Equal scores prefer deeper paths and then
insertion order. The solved state is accepted when popped from the queue.
`--queue-geometry-weight` changes the geometry coefficient for queue ordering.
The queue's color coefficient stays at 0.5, and candidate fitting uses the
50/50 score above. `--ignore-queue-geometry` sets the queue geometry coefficient
to zero.

The video pass uses the nine-class `26m-pose` model and ROI padding
`0.4` by default. The last static frame is inclusive. Frames through that
boundary hold the supplied scramble, and subsequent sampled frames each
offer a hold or a scored turn. `--start-frame` and `--end-frame` select an
interval before parity sampling. Observation caches with stale probability
or geometry scoring must be regenerated.

`--mismatch-window-weight 1` adds the pooled rate of out-of-table sticker
color mismatches from the last eight sampled frames to the queue score.
The same `26m-pose` pairs are excluded from this window term.

The independent-keypoint renderer defaults to a fixed `--branch-widths 3`.
The prepared search scores M, E, and S and simultaneous turns of parallel
layers. It combines the three layer angles on a 15° grid from −45° to +45°
using six distinct surface geometries per slab. The ±45° endpoints share
their surface projections while retaining their physical sticker identities.
Ray intersections, pairwise slab occlusion, landmark distances, and sticker
containment are cached before combining the slabs. The endpoint landmark
fit preserves the existing seam tie-breaking. Single and wide fits use
their finer grid.

Before color scoring, the prepared frame and video searches select the five
best distinct misalignment geometries. A flat hold counts once. Fits less than
0.01 above the fifth-best geometry score are included as ties.
The prepared search uses `--max-tps 40` for ordinary turns and
`--max-double-tps 50` for 180-degree turns, including doubles spanning multiple
frames. At 60 fps these allow 60° and 75° per interval respectively. The actual
source-frame spacing determines each interval, including skipped observations.
For each pair of shortlisted geometries, the search precomputes routes that
continue a parallel rotation or resolve it to an aligned boundary before
switching axes. Every outer-slab angle is measured relative to the middle slab,
where the axes are anchored; common rigid rotation consumes no turn budget.
Simultaneous slabs have independent limits, so two ordinary turns each traveling
50° fit a 60° interval. A slice also checks its two relative angles independently.
Sequential turns on different axes share the interval's time budget: finishing
45° on one axis and entering a 15° residual on another uses 60° at the same rate.
A geometry-only reachability pass identifies monotone 180-degree spans and
marks their intervals, including motion before the second quarter commit.
Ambiguous intervals that also fit an ordinary 90-degree span keep the 40 TPS
limit; merely having a possible double in the bank cannot raise a single’s rate.
An isolated single does not qualify, and a simultaneous ordinary turn keeps
its own 40 TPS limit even when the other slab is part of a double. Hidden moves
are added only when their full time fits after the residual travel.
Crossing ±45° uses the unwrapped angular distance.
This replaces handoff thresholds and bonuses. No geometry, visibility, or route
enumeration runs during state expansion: legal routes use cached sticker
permutations and batched color scoring. Temporal contexts retain the selected
residual geometry so the next interval uses the correct starting angles.
The default top-three child cutoff still ranks the resulting transitions by
the combined score, with equal geometry and color weights.

`--branch-widths 3 5 8` admits the top 3 distinct children per parent by
candidate rank, then widens to 5 and 8. Each cutoff includes all children
tied on the full ranking key. Use `--branch-widths 3` for a single stage.
Widening releases saved children from previously expanded parents into
the same priority queue. Each stage widens
when the queue empties or after 100 expansions per sampled frame in the
solve interval. The interval starts after the last static frame and ends at
the selected end frame. Use `--branch-stage-expansions` for a fixed budget.

The bidirectional script accepts the same width schedule. Its two queues
have separate expansion budgets and mismatch histories. A run joins when
both sides reach the same cube state and compatible turn handoff at the
same frame boundary. Supply a flat solved frame with `--end-frame` as the
backward anchor.

The prepared scorer keeps candidate scoring and ranking in compact numeric
rows. Bidirectional widening applies its existing unique-move cutoff and tie
rules to those rows before constructing states and diagnostic records.
Optional native kernels accelerate grouped ranking, state permutations and
color scoring, including batches of independent states. Build them locally with:

```bash
.venv/bin/python setup.py build_ext --inplace
```

Installation also attempts to build the optional extension. The Python/NumPy
fallback preserves the same candidates and scores; set
`CUBE_SEARCH_PYTHON_KERNELS=1` to exercise it explicitly. `evaluate_many` scores
independent states sharing a frame and restriction in a single color batch.
The priority queues still expand one node at a time; batching does not reorder
the frontier or change pruning, TPS limits, scoring or branch widths.

To compare mechanical throughput, first capture real expansion inputs with
`scripts/benchmark_independent_keypoint_expansion.py --config CONFIG_JSON
--output-dir CAPTURE_DIR`, then replay its `workload.json` using `--workload`
and a fresh output directory. Replay verifies the complete candidate records
outside the timed region. `--compare-admission BENCHMARK_JSON` additionally
checks that the admitted continuations match a previous replay exactly.

`--search-seconds 300` sets a search-only time limit per pass.
`--max-explored-states 20000` sets an exact expansion limit per pass.
A two-pass run writes a `summary.json` and separate `even/` and `odd/`
directories. Each pass saves `config.json`, `progress.jsonl`, `status.json`,
`result.json`, and its SQLite search queue. A single pass writes directly
to the output directory. Create an `OUTPUT_DIR/STOP` file to stop the run.

Add `--save-final-video runs/my_solve.mp4` or `--display` to replay the first
solved pass with its pose overlays and virtual cube. The replay uses the
original video clock and holds the last committed state on unsampled frames.
To render a saved solution later, pass the solved directory to the renderer:

```bash
.venv/bin/python scripts/render_independent_keypoint_search.py \
  --result runs/full-solve-frame/my_solve/even \
  --save-final-video runs/full-solve-frame/my_solve/solution.mp4 \
  --turn-candidate-log runs/full-solve-frame/my_solve/rendered_turn_candidate_scores.jsonl
```

The original video path comes from the input metadata. Override it with
`--source PATH` for the renderer or `--render-source PATH` for the search.
The turn log can be opened with `review_turn_candidates.py`.
`--no-show-keypoints`, `--no-show-orientation-axes`, and
`--no-show-virtual-cube` control the replay overlays.

For ground-truth overlays on the prepared dataset, use
`scripts/render_independent_keypoints.py`:

```bash
.venv/bin/python scripts/render_independent_keypoints.py \
  --dataset data/independent_keypoints_dataset_v2 \
  --stems 000000 005000 \
  --output-dir runs/independent_keypoint_ground_truth
```

## Direct pose and articulation head

This section describes the earlier **single-stage** experiment. For the reviewed
epoch45 model with separate static and image-context articulation heads, use
[`train_yolo26_staged_cube_pose.py`](scripts/train_yolo26_staged_cube_pose.py) and
the [staged CUDA training instructions](#staged-cube-pose-with-joint-cuda-training).
The two trainers have different architectures and incompatible custom checkpoints.

The additional targets in `data/direct_cube_pose_dataset/` support a structured
head on the existing nine-class independent YOLO pose detector. Six centers in
`U,D,F,B,R,L` order (white, yellow, green, blue, red, orange) supervise the rigid
pose. The branch also predicts six clockwise face offsets in degrees, normalized
image center/scale/perspective/principal point, and a continuous 6D rotation
representation converted to a proper rotation matrix.

```bash
.venv/bin/python scripts/train_yolo26_direct_cube_pose.py \
  --dataset data/direct_cube_pose_dataset \
  --model runs/pose/runs/26m-pose/weights/best.pt \
  --device 0 --imgsz 160 --batch 16 --epochs 100 \
  --output-dir runs/direct_cube_pose
```

The default checkpoint is the trained YOLO26m independent detector. A stock
pose `.pt` or `.yaml` can also initialize the detector; its classification and
keypoint shapes are adapted to the nine classes and one keypoint per object.
The original detector retains the installed Ultralytics default pose loss.
The new branch first warms up for five epochs with that detector frozen, then
jointly fine-tunes with a detector learning rate one tenth of the new head's.
Use `--freeze-epochs 0` for joint training from the first epoch.

`--imgsz` (also `--resolution`) rounds up to the model's stride of 32, with a
minimum of 64; for example, `--imgsz 144` trains at 160. `--batch -1` estimates
the full model's CUDA memory use, including backward through the new head.
On CPU or MPS, automatic batch sizing uses the standard fallback of 16. Resuming
with `--batch -1` restores the checkpoint's effective batch size. The resolved
image and batch sizes are printed at startup.

Progress prints on the first batch and every 50 batches: total loss, independent
YOLO loss, custom pose loss, center/visible-keypoint error in pixels, rotation and mean
face-angle error in degrees, moving-face angle error, and predicted pose quality
versus its target. Batch updates refresh one console line; epoch summaries
remain on separate lines.
Epoch summaries report train/validation losses and validation pose metrics.

Inputs to the branch are shared multiscale features and multiple predicted
landmark/sticker points per class. Point weights come from the independent
objects' detection/class logits, never their keypoint-presence channel.
Ground-truth points are used only for supervision. Auxiliary center predictions
and projected articulated landmarks guide the structured pose. The loss chooses
one allowed symmetry jointly for rotation, ordered centers, turn channels, and
visible keypoint reprojection. It cannot choose a different symmetry for
each term. Historical sticker omissions make the allowed sets conservative.
Slice targets use the verified frame rebase; corner cuts use the dominant angle.

The geometric term (default weight 10) fits the saved visible landmark
observations and labeled center-sticker observations, using their existing
physical-anchor correspondences. Both retained visibility 1 and 2 contribute;
absent labels and hidden ideal points do not. It averages radial Smooth L1
reprojection error in normalized image coordinates (transition at 0.02 image
width). Every visible match contributes: there is no fixed-fraction trimming
or hard distance cutoff. Large residuals have bounded gradient magnitude
instead of the increasing influence of squared error. With no visible matches the
geometric term is zero, with no fallback to hidden targets. Geometry is averaged
over images with visible supervision, so empty sets cannot dilute its scale.
No dataset regeneration is needed. Other rotation, projection-parameter,
center-anchor and balanced face-angle losses retain ground-truth supervision.

`visible` is the untrimmed RMSE over all landmark and center-sticker observations
used by this geometric loss. Epoch `landmarks` remains the full 56-point ideal
target RMSE for diagnostics; `observed` remains untrimmed visible-landmark RMSE.
Colored sticker points without a matched physical anchor retain their original
independent YOLO supervision.

The angle loss averages errors within moving and stationary face groups, then
weights the two present groups equally for each image and symmetry candidate.
Aligned cubes still supervise all six zero angles. A face is moving when its
target offset exceeds `0.0001` degrees in magnitude. This avoids letting the
roughly 85% zero-offset targets dominate articulation training. `turns` retains
the mean over all six faces for comparison with earlier logs; `moving_turns`
reports only target moving faces. JSON metrics additionally include
`stationary_angle_mae_deg`. Epoch and group moving/stationary metrics are
weighted by the number of corresponding supervised faces, excluding masked
images. Epoch JSON metrics also record the moving/stationary face counts; a
zero count means that metric has no contributing faces. These diagnostics use
saved target motion, not predicted visibility.

Empty-label images retain their independent detection supervision and have all
direct geometric losses masked. Their pose-quality target is zero. The learned
`quality` score estimates visible reprojection accuracy relative to the best
joint allowed target: `exp(-0.5 * (normalized_visible_RMSE / 0.02)^2)` by
default. It uses all visible matches, is
detached from the pose gradients, and has target zero without visible matches.
Hidden ideal landmarks do not determine this quality target.
This score is distinct from object detection and keypoint-presence scores and
does not measure the probability that an orientation is unique or calibrated.

`split.json` fixes the stratified train/validation split. Labels are cached once;
training augmentation changes brightness, contrast, and blur while preserving
color identities and geometry. `metrics.jsonl` records losses, center/landmark
projection errors at the training resolution (both target and observed landmarks), rotation error, and face-angle
error, including validation breakdowns by turn mode and allowed symmetry count.
`weights/best.pt` and `weights/last.pt` contain the full detector and head,
optimizer/scheduler/scaler state, random states, and configuration. Resume with
`--resume runs/direct_cube_pose/weights/last.pt` and the same model/data/objective
options; `--epochs` may be increased. Reuse the same output directory on resume.
The objective version is recorded in checkpoints and metric records. Resuming
an older objective's checkpoint keeps learned weights and optimizer progress,
prints the objective change, and resets best-loss/early-stopping history because
the old and new loss scales are not comparable.

A short CPU wiring check, which does not establish pose accuracy:

```bash
.venv/bin/python scripts/train_yolo26_direct_cube_pose.py \
  --model yolo26n-pose.yaml --limit 32 --epochs 1 --freeze-epochs 0 \
  --imgsz 64 --batch 2 --workers 0 --device cpu --max-batches 2 \
  --output-dir runs/direct_cube_pose_smoke
```

For inference, `live_recognition.direct_cube_pose.load_direct_pose_checkpoint`
returns `(model, checkpoint)`. Pass square RGB `[B,3,H,W]` floats in `[0,1]` at
the saved training size. `model(images)["pose"]` returns the structured pose,
auxiliary `center_keypoints`, `projected_centers`, and `quality`;
`model(images)["independent"]` preserves the raw YOLO outputs. No original base
checkpoint is required to reload the saved model. This architecture still needs
an accuracy comparison against the earlier six-point model on matching data and
training budgets, especially for blur and occlusion.

Observation tokens follow the checkpoint's configured inference branch:
one-to-many when end-to-end inference is disabled, otherwise one-to-one.
New custom-pose checkpoints record `observation_version` and
`observation_token_branch`. Earlier observation caches that always selected
one-to-one outputs for a non-end-to-end checkpoint must be regenerated;
custom heads trained on those inputs need fresh validation or retraining.

The current single-state objective does **not** yet handle hidden-layer
articulation uncertainty. It still penalizes hidden face-angle and geometry
differences even when the visible observations are identical. Reproduce this
limitation before treating an overfit/smoke run as evidence of correctness:

```bash
.venv/bin/python scripts/verify_hidden_layer_ambiguity.py
```

The report in `runs/hidden_layer_ambiguity_verification/report.json` deliberately
records `ambiguity_handled_by_current_loss: false`. This synthetic diagnostic
uses a crop filled by an opaque front face; it does not discard silhouette/color
evidence from existing images or infer hidden layers from missing old labels.
Supporting the requested multiple explanations requires a joint multimodal
pose/articulation model and uncertainty for continuously unconstrained angles.

For the first experiment, only the existing cube-frame symmetries are handled.
To verify that the current head can memorize its supplied targets, run:

```bash
.venv/bin/python scripts/overfit_direct_cube_pose.py \
  --samples 24 --steps 600 --output-dir runs/my_direct_pose_overfit
```

This samples parts 1/3/5/7, all five turn modes, and 1/4/24-symmetry cases.
Augmentation and weight decay are disabled. The pretrained detector is frozen,
so its predicted inputs can be cached without changing the head's training
objective. The final result is replayed through the full image-to-pose path and
again after checkpoint reload. Reports, target/before/after wireframe overlays,
and an inference-only `overfit.pt` checkpoint are written to the output folder.
This diagnostic is training-set memorization, not held-out evaluation.

Checkpoint v2 explicitly preserves the detector's runtime `end2end` flag, which
can differ from its YAML architecture setting. The independent loss uses the
shipped `PoseLoss26` for a `Pose26` head even in non-end-to-end mode: in the
tested Ultralytics version, the generic non-end-to-end criterion otherwise uses
the older XY decoder. Box/class/keypoint loss formulas and gains remain the
provided YOLO26 defaults. Legacy prototype checkpoints lacking the detection
mode are rejected instead of silently reloading different prediction branches.
The independent criterion also checks each forward result's actual structure:
flat dictionaries use the stock single-branch pose loss, and paired
`one2many`/`one2one` dictionaries use the stock E2E loss. This prevents a
`KeyError: 'kpts'` when a runtime prediction layout differs from the model flag
seen at criterion construction. Package versions and selected loss layout are
printed in the training log.

For a nonfinite first-batch loss, run the standalone diagnostic on the same
CUDA device and with the same model/image-size options:

```bash
.venv/bin/python scripts/check_direct_pose_batch_numerics.py \
  --run-dir runs/direct_cube_pose --device 0
```

Use the failing run's output directory for `--run-dir` to reuse its saved model,
image size, seed, augmentation and loss settings. The diagnostic defaults to
the four reported first-batch samples and compares AMP loss math,
FP32 loss math on identical AMP network outputs, and a full FP32 forward/loss.
The report under `RUN_DIR/numeric_probe/report.json` includes individual
loss components and assignment/keypoint/RLE stages. It performs no optimizer
updates. Add `--batch-file RUN_DIR/diagnostics/train_epoch_1_batch_000001_batch.pt`
to replay the exact saved images and targets, including their original size and
augmentation. The probe also reports invalid model parameters/buffers and the
first network layer producing nonfinite outputs in AMP and FP32.
Normal training has no additional scans, snapshots or retries; detailed
diagnostics and the failed batch are saved only if its existing finite-loss
check fails. These checks do not assume that AMP is the cause.

The trainer applies Ultralytics' stock AMP compatibility check once at startup,
before automatic batch sizing, and prints the effective training precision.
Ultralytics disables AMP on the GTX 1660 family because FP16 convolution can
produce NaNs. A CUDA report on the GTX 1660 SUPER confirmed finite inputs and
weights entering a nonfinite FP16 detector convolution, while the FP32 forward
and both losses passed. This hardware therefore trains in FP32, matching the
standard YOLO trainer. Compatible CUDA devices retain AMP. No compatibility
checks or retry passes run inside the training loop.

## Ordered cube pose labels

The ordered 26-point pose target represents one rigid cube transform. When
`scripts/apply_blender_cube_moves.py` is run with `--enable-mid-turns`,
mid-turn and corner-cut images receive layer-aware keypoints: each point stays
attached to its owning cubie while that cubie is transformed by the partial
turn. Empty pose files remain supported for older or deliberately excluded
samples, and the pose trainer skips those files when building training pairs.
Pass `--pose-label-mode ordered` to the Blender generator when creating this
26-point target.

## Synthetic dataset lighting

The Blender generator samples saturated room lighting by default. Yellow and
blue are the most common casts, with occasional red, magenta, cyan, and green.
By default, 15% of lighting selections use a white key with a separately
colored fill and a matching wash on the solve-station wall when available.
The mixed profiles pair warm white with blue-violet or teal, cool white with
amber, and neutral white with magenta.

```bash
blender --background syntheticAssets/cube.blend --python scripts/apply_blender_cube_moves.py -- \
  --count 1000 \
  --output-dir data/synthetic_cube_dataset
```

Each completed image gets its own atomically written `metadata/<image-stem>.json`
file with its lighting profile, sources, and labels. The generator exports the
combined `metadata.json` array at the end of the run for existing readers.
Generated images and visualization masks are 144 × 144 pixels.
Set `--mixed-lighting-probability 1` to generate only mixed-lighting samples or
`--mixed-lighting-probability 0` to use only the room-wide cast profiles.
Rotational blur and post-render camera degradation use their regular probability
settings for all lighting profiles.

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

## Staged cube pose with joint CUDA training

`scripts/train_yolo26_staged_cube_pose.py` trains the architecture used by
`robust_handoff_context60/epoch_045.pt`: one independent YOLO26m encoder,
a static orientation/projection head, and the canonical geometry plus
image-context transformer articulation head. The second head predicts layer
turns; it does not revise the first head's static pose. There is no PnP or
iterative pose fitter in inference. The geometric frame is defined modulo the
24 proper cube symmetries; a separate color-frame assignment is not added here.

Use the already exported `data/direct_cube_pose_dataset` (17,857 images),
including `manifest.jsonl`, `symmetries.json`, `images/`, `pose/` and `targets/`.
The original nine independent classes and visibility 1/2 annotations are
preserved. No re-export or Blender generation is needed. For auxiliary
early reprojection and layer-support targets, retained landmark metadata provides
physical slots, and sticker labels use known centers or unique nearby model
projections. Uncertain sticker bindings are omitted only from this auxiliary
loss; all original YOLO rows remain in the default detector loss. RGB-only
handoff/joint geometry uses the original visible label rows directly, without
requiring those physical-slot bindings. Dominant-angle
corner-cut handling is retained.

From the repository root in the Windows CUDA Python environment, copy
`runs/staged_cube_pose_epoch45_init.pt` from this workspace to the same relative
path, then run this single-line PowerShell-compatible command:

```powershell
python scripts/train_yolo26_staged_cube_pose.py --dataset data/direct_cube_pose_dataset --init-checkpoint runs/staged_cube_pose_epoch45_init.pt --device 0 --batch 16
```

The approximately 100 MiB initialization file contains the reviewed static/source
epoch30 and complete articulation epoch45 weights; no investigation caches or
other `runs/` folders are needed. It also records their original training keys,
which are kept out of the new validation split. With seed17 and the current
dataset, the split is 16,083 training / 1,774 validation images. This is validation
for the new custom-head training; the existing detector's earlier keypoint
pretraining may already have used these source images.

The default128px schedule automatically runs:

| Phase | Epochs | Updated components | Articulation conditioning |
|---|---:|---|---|
| Static | 30 | Detector and static head | Turn head frozen |
| Articulation | 40 | Complete turn head | 50% correct, 50% perturbed static poses |
| Handoff | 30 | Complete turn head | 100% actual RGB estimates |
| Joint | 60 | Detector, static head and complete turn head | 100% actual RGB estimates; no ground-truth pose replay |

Thus 90 of the default 160 epochs run the full prediction path without
ground-truth pose, camera, or articulation inputs. Ground truth is used only
after prediction to calculate losses and metrics. The earlier static and
known-static phases still train the components individually.

During handoff/joint, the geometry objective projects all three network-predicted
axis hypotheses, each with its predicted rotation, camera, and continuous layer
angles. Visible label rows (both visibility 1 and 2) are matched to the nearest
visible, class-compatible projected landmark or sticker. All observations are
retained with radial Huber influence; no fixed fraction is discarded. Geometry
ignores sticker identities/colors when matching centers, as in the manual fitter.
It is invariant to equivalent cube frames and layer quarter-turn permutations.
Half the objective scores the deployed hard axis choice; half scores the
axis-probability-weighted losses of the three complete shapes. This supplies
gradients to axis selection without averaging incompatible shapes. No
ground-truth axis is forced into these projections. The shared encoder runs
once and articulation runs once per batch in these phases, including diagnostics.

The RGB-only objective is `geometry + 0.25 * (static + angle) + 0.1 * YOLO`.
Geometry uses a 2px Huber transition and is expressed in pixel-loss units divided
by 4, balancing landmark/sticker groups. `--geometry-weight` (default 1) and
`--pose-aux-weight` (default 0.25) adjust the first two weights. Pose/angle labels
remain auxiliary loss targets; they never condition the handoff/joint forward
pass. The detector retains its installed default YOLO pose loss.

Each phase hands its best validation checkpoint to the next phase. Static is
selected by static rotation error, articulation by explicitly teacher-conditioned
observed-turn error, and handoff/joint by actual RGB-only class-nearest visible
fit RMSE (`observed_fit_rmse_px`, displayed as `fit=...px`).
All stages also report actual static/moving/observed-moving errors and visible
reprojection RMSE. The older bound-point metric remains
`visible_keypoint_rmse_px`; its correspondence policy differs from the new
class-nearest fit metric. Teacher-conditioned metrics are labeled separately
and are omitted (`null`) during handoff/joint, which perform no GT-conditioned
diagnostic forward pass. Metrics
and stage losses are saved in `metrics.jsonl`; batch progress overwrites one
console line and fits within the current terminal width to prevent wrapping.
Redirected/captured output receives one final batch summary per train/validation
pass. The static/source copy and both turn branches are unfrozen during
joint training. New runs start joint training at `1e-4` for the static head,
`3e-4` for the articulation head, and `1e-5` for the shared backbone/detector.
These are 10x, 6x, and 5x the previous joint defaults, allowing substantial
head adaptation to the new objective. Override them with `--joint-static-lr`,
`--joint-articulation-lr`, and `--joint-backbone-lr`. Each group follows the
existing cosine schedule toward `1e-6` over the joint phase. Actual rates are
printed at phase entry and each epoch, and the rates used for each epoch are
recorded under `learning_rates` in `metrics.jsonl`. CUDA batch calibration uses
the same configured optimizer as training. Detector batch-normalization
statistics remain fixed, as in the reviewed model. The unused one-to-one branch
and DFL integral are frozen; the configured one-to-many path uses the installed
default YOLO pose loss. Structured geometry stays FP32; supported CUDA hardware
can use AMP for the detector after its stock compatibility check.

Internal point-selection NMS runs in FP32 on CPU, as in the reviewed experiment,
and processes every image without a whole-batch wall-clock cutoff. This avoids
slow Python-loop GPU NMS and prevents timeout-truncated batches from fabricating
empty point inputs. Neural points/class scores retain gradients on their original
device; the discrete CPU selection mask is transferred back once per batch.
Checkpoints record the observation version. When resuming a pre-fix run, weights,
optimizer and schedule are kept, but the current-stage validation best is reset
so old timeout-affected scores do not control the new stage selection.

Training pose diagnostics run every10batches by default; all batches still
receive their original full losses/backward/optimizer updates. Validation metrics
always use every validation batch. `--train-metrics-every 1` restores full training
diagnostics, including on resume. Logged `pose_metric_batches`/`pose_metric_images`
identify the diagnostic sample size. Articulation image descriptors are sampled
only when the turn head or its diagnostics need them. Console progress includes
images/second and an epoch ETA.

`--static-epochs`, `--articulation-epochs`, `--handoff-epochs` and `--joint-epochs`
change stage lengths; zero skips a phase. `--batch -1` calibrates a separate
CUDA batch size at the start of each phase, using that phase's freeze mask and
actual loss, metrics, backward pass and AdamW step. Calibration runs on a
disposable model copy and preserves the training weights and random states.
It includes optimizer-state memory and avoids requesting image gradients
through a frozen encoder. It runs once per phase, with results recorded in
`batch_profiles.jsonl`; it adds no per-batch profiling during training.

To continue the completed model using only the new RGB-only joint objective,
keep the original held-out split and start a separate run:

```powershell
python scripts/train_yolo26_staged_cube_pose.py --dataset data/direct_cube_pose_dataset --init-checkpoint runs/staged_cube_pose/weights/best.pt --split runs/staged_cube_pose/split.json --static-epochs 0 --articulation-epochs 0 --handoff-epochs 0 --joint-epochs 60 --device 0 --batch -1 --output-dir runs/staged_cube_pose_rgb_only
```

`--split` preserves the exact listed train/validation memberships and rejects
validation images previously used to train the checkpoint. It cannot be
combined with `--limit` or `--resume`. Older checkpoints remain valid for
initialization and rendering. Since the objective and selection metric changed,
exact `--resume` requires a checkpoint with the new objective version; use
`--init-checkpoint` and a new output directory to continue an older model.

Resume preserves the saved learning-rate plan and optimizer/scheduler state.
Checkpoints from before configurable joint rates preserve their old conservative
rates. Explicit joint LR flags on resume override only those groups' base rates,
preserving AdamW moments and the current cosine epoch. Thus overriding halfway
through a phase applies the halfway-decayed value of the new base rate; it does
not restart the schedule. To apply all three new base rates to such a run:

```powershell
python scripts/train_yolo26_staged_cube_pose.py --resume runs/staged_cube_pose_rgb_only/weights/last.pt --device 0 --joint-static-lr 1e-4 --joint-articulation-lr 3e-4 --joint-backbone-lr 1e-5
```

To automatically use larger head-only batches and fit joint training separately:

```powershell
python scripts/train_yolo26_staged_cube_pose.py --dataset data/direct_cube_pose_dataset --init-checkpoint runs/staged_cube_pose_epoch45_init.pt --device 0 --batch -1
```

`--static-batch`, `--articulation-batch`, `--handoff-batch`, and `--joint-batch`
override the global batch for individual phases; each accepts a positive size
or `-1` for automatic sizing. For example, a manually selected plan could use
`--static-batch 32 --articulation-batch 128 --handoff-batch 128 --joint-batch 16`.
The CUDA calibration defaults to an85% free-memory budget and a1024-image cap,
adjustable with `--batch-memory-fraction` and `--max-auto-batch`. Explicit sizes
are used directly; CPU/MPS automatic sizing falls back to16 (or dataset size).
Learning rates remain unchanged when batch sizes change; larger batches give
fewer optimizer updates per epoch.

Resolved stage batch sizes are saved in every training checkpoint and printed
at phase entry. Resume normally restores them. Explicit batch arguments on
resume override the saved sizes without resetting weights, optimizer or schedule.
For example, `--resume runs/staged_cube_pose/weights/last.pt --device 0 --batch -1`
recalibrates the current and remaining phases that inherit the global batch;
`--joint-batch -1` recalibrates just joint training. Individual phase overrides
take precedence over the global batch.

`--resolution` aliases `--imgsz` (128 by default), and
`--source-dir` aliases `--dataset` for the exported dataset. No spatial
augmentation is introduced. Windows loaders use spawn, top-level functions and
the multiprocessing main guard. `--workers 0` is available for loader diagnostics.

To check CMD progress without loading weights/data or starting training:

```powershell
python scripts/train_yolo26_staged_cube_pose.py --progress-test
```

On an attached Windows console this reports `Batch progress: Windows console cursor`
and counts twenty updates on one row. It uses native console cursor positioning
and the visible window width; redirected output reports its capture mode and
prints one final summary. Restart/resume the Python process after updating the
trainer and `src/live_recognition/training_progress.py` to load the new writer.

To continue an interrupted run, using its saved split, optimizer, RNG and schedule:

```powershell
python scripts/train_yolo26_staged_cube_pose.py --resume runs/staged_cube_pose/weights/last.pt --device 0
```

Checkpoints are self-contained. Load for inference with
`live_recognition.staged_pose.model.model_from_checkpoint(path, device)`;
`model(images)["pose"]` uses only RGB. The joint stage writes
`weights/best.pt`; `weights/last.pt` and per-stage best checkpoints are retained.

To reproduce the portable initialization from individual experimental weights:

```powershell
python scripts/train_yolo26_staged_cube_pose.py --static-checkpoint runs/direct_cube_pose_1k_investigation/continued/static_first_rgb_adapt30/epoch_030.pt --articulation-checkpoint runs/direct_cube_pose_1k_investigation/continued/robust_handoff_context60/epoch_045.pt --initial-split runs/direct_cube_pose_1k_generalization_128_mps/split.json --export-init runs/staged_cube_pose_epoch45_init.pt
```

For a short pipeline check instead of full training, add
`--limit 16 --max-batches 1 --static-epochs 1 --articulation-epochs 1 --handoff-epochs 1 --joint-epochs 1 --output-dir runs/staged_pose_smoke`.
