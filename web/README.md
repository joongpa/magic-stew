# Live Recognition web detector

Svelte 5 + TypeScript + Vite with both ONNX models running in a worker through
WebGPU or WASM (CPU). This follows the native renderer's two-part flow:

**640px full-frame cube detection → padded square ROI → 256px pose inference →
pose-point ROI tracking.** The cube detector runs again after tracking loses its
usable points. The pose input defaults to 256px, matching the search scripts; 160px and 288px are also available.

## Run

From the repository root (Node 22.3+ recommended for cubing.js):

```sh
cd web
npm ci
npm run dev
```

Open the printed localhost URL. Choose **WebGPU (GPU)** or **WASM (CPU)** under
**Inference backend**. Click **Try a sample
image**, **Start camera**, or **Open file**. You can also drop an image/video on
the viewer. Camera access requires browser permission. Use HTTPS when serving
anywhere other than localhost. Camera, images and videos stay on the device;
there is no server-side inference. WASM uses the dedicated CPU runtime with SIMD and up to four threads in the
worker and does not require WebGPU. The viewer reports the actual CPU thread
count. Threading requires cross-origin isolation; without it, CPU mode runs
single-threaded and the viewer reports that limitation. Changing backend finishes the current
frame, retains queued source frames, and restarts both model sessions.

## Cube timer (testing)

Open **[/timer](/timer)** directly or select the **Cube timer** tab and connect a camera.
The detector stays at `/`; tab navigation supports browser back/forward.
Hold space for 0.3 seconds,
then release to start; **any newly pressed key stops** the solve. The on-screen
button supports the same hold/release interaction, and tapping it stops a solve.
Camera interruption stops the timer and records a DNF.

Scrambles are generated locally by [cubing.js](https://js.cubing.net/cubing/scramble/)
using `randomScrambleForEvent("333")`.
The flat cube net shows the resulting scrambled state. No Java or scramble server
is required. Use Node 22.3+ for dependency installation (as required by cubing.js).

The camera keeps streaming between solves, with detection and pose fitting idle.
An idle preview uses the connected stream directly. The capture reader continues
draining native frames and reporting capture FPS without creating inference
bitmaps; starting a solve resumes inference-frame delivery on the same source
clock. The processed-frame counter counts inference, so it stays at zero before
the first solve even while the preview is streaming.
Starting a solve resumes capture admission and starts fresh tracking/fitting context
while retaining the loaded workers. Frames are admitted by camera source timestamp
in **[timer start, timer stop + 0.3s)**. At the end of that window, an ordered boundary
flushes remaining fitting lookahead, drains admitted frames, and closes search input.
Idle camera frames are neither inferred nor retained. The camera source clock is anchored
once in the capture worker, before bitmap transfer and inference queues; timer
boundaries do not use the delayed viewer position. This mapping uses the first
frame's capture-worker arrival time, so it does not calibrate camera/driver latency.
Camera fitting and search use integer millisecond timestamps (`fps: 1000` is the
numerical clock rate; `source_fps` retains the nominal capture FPS). This preserves
irregular camera timing without rounding two admitted frames onto the same index.
The clock conversion itself does not discard frames. Video keeps its source-frame clock.
Forward move search starts with the first finalized pose inside the timed range
and receives each later finalized pose immediately, while timing/capture/fitting
continue. As with uploaded-video forward search, fitting promises are submitted
in source order and their banks are prepared as soon as they resolve, even if
an earlier fit is holding up display. The ordered display/recording queue does
not delay search preparation. Input closes after the stop + 0.3s tail has finished fitting. A newer
solve pauses older search expansions; camera backpressure spaces expansion chunks
apart only while its input remains open, to leave CPU time for fitting without
starving search. Closed solves have no camera pacing delay. Finalized frames
continue to be prepared immediately, and neither mechanism resets the frontier.
Finding a solution leaves the camera connected; detection pauses after the tail drains. Disconnecting the camera preserves
searches whose complete input window has already been received. Incomplete capture or failed fitting
is reported explicitly. Keep the scrambled/solved cube visible around timer input.
Reconstruction has a 10-second wall-clock deadline starting after the 0.3-second
capture tail. The main thread terminates the search worker at that deadline,
including a worker blocked inside WASM or still waiting for fits. The solve time,
last candidate, diagnostics and pose replay remain available after a timeout.
Detection records are retained only within timed solve windows, including the
0.3-second tail. The detector and fitter do no work on idle frames between solves.

During capture and tail draining, the timer displays the furthest fully processed
camera frame, with the fitted pose drawn on that same frame by default. Once the
tail drains, it returns to the live idle preview. **Show fitted pose** remains
available in the timer tab; disabling overlays still shows processed frames during
a solve.
Its throughput row separates measured capture FPS from processing
FPS, processing delay and skipped frames. A full buffer keeps draining at the rate
the processor can sustain: queued frames are preserved, and newly arriving frames
are skipped until a slot is available. Capture does not stop, and no Resume action
is needed. Browser-discarded frames are counted as well; real timestamp gaps remain
in fitting/search. Processing drains the camera FIFO without an additional 30fps
admission timer. Finalized fit banks/meshes are released from internal pipeline
history once only their continuity core is needed.

Timer diagnostics are enabled by default and create a separate run as soon as
timing starts, including attempts that fail before move search. On local Vite dev
and preview servers, captures are checkpointed during the solve and saved on stop,
failure and completion under `pose-fitter/results/browser-runs/`; `latest.json`
identifies the most recent capture. Each capture contains only that solve's
detections, model/settings and source clock, original per-frame `fittingError`,
and the search journal if search ran. **Download diagnostics** in the solve details
exports the same run while that page session remains open. The visible failure
message includes the first underlying fitting error and its source timestamp.
The Detector tab's **Record search diagnostics** setting also controls timer logs.
Large diagnostic JSON is serialized in a separate worker, with one snapshot in
flight and repeated queued saves coalesced. Ongoing search progress updates memory
without repeatedly rewriting the full detection capture or browser-local session.

The session includes +2/DNF edits, deletion, best time, ao5/ao12 (dropping best and
worst), scramble/reconstruction details and JSON export. Times and compact results
are retained in browser local storage; raw camera images are not stored there.
New solves also retain compressed processed-frame snapshots in page-session memory
for replay. Already processed frames become available immediately when the timer
stops, with remaining solve/tail frames appended as they finish. Replay does not
wait for reconstruction to finish or succeed. Solve details show the same `ProjectedCube` viewer used by uploads,
including reconstructed colors, moves/rotations, smooth motion and adjustable view,
beside the matching processed camera frame and selected pose overlay. Play/pause,
0.25×/0.5×/1× speed, scrubbing, and frame buttons/arrow keys use the original irregular
source timestamps. Deleting a solve releases its replay. Reloading preserves times
and compact results but releases replay images; historical solves from before this
feature have no camera images to replay. Snapshots are encoded asynchronously only
inside solve windows, without waiting before search submission.

This initial version has no inspection countdown or hardware timer input.
The existing Detector tab retains file reconstruction.

## Video reconstruction

Upload opens a preview and a video-editor trim control before inference starts.
Drag the two handles or edit the start/end seconds, enter the initial scramble,
and choose **Reconstruct selected range**. The thumbnail strip shades excluded
footage. Thumbnail/handle previews decode video only and create no inference
records. The selected interval is start-inclusive/end-exclusive on the source
clock; the 30fps selector establishes its origin at the first included frame.

The first pass has no pause/resume button or playback scrubber. Those controls
appear after the selected frames have finished and move search reaches a terminal
status. Clear/replacing the source still cancels the run. When complete, the
projected cube appears automatically in an enlarged panel beside the video.
The panel renders full reconstructed sticker colors from the applied scramble
and recovered moves. It stays synchronized during cached playback and scrubbing.
Its camera, cube origin and scale are fixed: only the selected fit's rotation,
residual layer turns and reconstructed sticker state animate. Source-camera
translation, distance and lens distortion do not affect this separate view.
The frame counter shows the saved-frame ordinal and the zero-based source frame.
After reconstruction, Left/Right or the previous/next buttons pause playback and
step one saved result (two source frames for a 60fps source sampled at 30fps).
Keyboard shortcuts leave text fields and other editing controls alone.
**Smooth cube motion** eases rotations and turns over 100ms at the screen's refresh
rate with `requestAnimationFrame`, independently of detector/video frame cadence.
When a new target arrives, the previous ease curve is cut off immediately and
the new curve starts from the currently displayed pose; curves never queue up.
Animation stores one global rotation and three shared slab angles. Every sticker,
center and solid body on a slab uses the same transform. Committed moves rebase
the shared angles, preserving continuity through quarter-turn resets and reverse
scrubbing. Perpendicular turns meet at an aligned cube before the next axis moves;
there are no per-sticker motion paths. Distant seeks snap to their frame; adjacent steps ease to
the exact saved state. Disable smoothing for immediate frame inspection. The video
and its wireframe always show the exact selected frame without this easing.
The video retains its separate wireframe overlay. Its mesh uses the geometry selected by
search for that frame, including the reserved sixth flat fit; all six meshes
were computed when the finalized fit bank became available. Visible physical-face IDs and camera depths are
saved with every candidate mesh. The side panel uses the fitter's physical geometry,
applies the scramble and recovered moves using the CPU cube permutations.
A small WebGL renderer draws closed, opaque slab bodies and inset colored stickers
with a depth buffer, including the exposed black caps between turned layers.
Surfaces are occluded per pixel rather than sorted by their average depth.
Seeking reads immutable state snapshots; it
does not apply moves again or change the video wireframe overlay. Reviewing the saved
reconstruction runs neither fitting nor search again. **Adjust range** returns
to setup and preserves reusable completed results for the loaded file.

The layer planner and mesh builder are independent TypeScript modules. From
`web`, `npx tsx tools/check-cube-motion.mts recording.json [report.json]` checks
physical sticker continuity at every phase boundary, forwards and backwards,
while simulating 120Hz refreshes interrupting 100ms eases on 30fps input. This
checks geometry; it does not measure browser/GPU frame rate.

## Move search

Move search is enabled by default and requires an initial scramble. The Rust
engine preserves the CPU scorer used by
`search_independent_keypoint_bidirectional.py`. Its forward frontier uses:
54-byte cube states, directional 26m color-confusion costs, unique-sticker duplicate
semantics, balanced geometry/color ranks, geometry contexts, eight-frame mismatch
history, dominance, sibling ties, sparse branching and deferred widening. There
is one shared **40 TPS** move limit; cube rotation remains at **20 TPS**.
No double-turn witness/classification is used. Saved runs retain their recorded
TPS setting for diagnostic replay.
The browser defaults to width 3, sparse threshold 3 and a 1,000,000-state budget.
The optional opening hold is measured from the first included frame.

Videos default to **Bidirectional video search**. Once every selected frame's
pose fit completes, two separate Rust/WASM workers search from the scramble and
a corroborated solved end pose. They exchange admitted states and validate a
joined path in Rust before publishing a solution. Each direction retains its
full configured expansion allowance; backward work does not change the existing
forward queue or consume its allowance. Two workers can still add startup,
messaging and CPU/memory overhead, especially on easy solves. Missing solved
anchors or backward-worker failures fall back to forward search. The optional
geometry-derived one-second TPS gate remains an isolated experiment and is not
enabled here.

Turning the video option off, or using the camera, selects streaming forward
search. A separate CPU WASM worker prepares each immutable fit bank immediately when its
promise resolves. Source ordinals are assigned before completion, so out-of-order
fits can be prepared without skipping an earlier frame. States needing an
unavailable frame remain parked; eligible states continue, and newly arrived
frames wake the frontier. Waiting does not count as exhaustion or consume search
expansions. End-of-input is explicit. Search has no dependency on Svelte, video
pixels or OpenCV. Shared cube/permutation conventions are embedded in the module.

Progress displays prepared/searched frames and expanded states. Unfinished paths
are labeled **Current candidate**, and only a verified solved state is labeled
**Solution**. Downloads include `moveSearches`, its configuration/source range,
chosen geometry indices, moves, state/queue scores and computation timings.
State-limit/exhausted results are explicit, not presented as solved.

The standalone checks in [the fitter/search documentation](../pose-fitter/README.md)
compare both native Rust and WASM with Python on identical saved banks. A streaming
run can explore a different valid path because it works on other eligible states
while the deepest branch waits; it is not claimed to reproduce batch expansion
order. Matty's streamed path is independently checked against Python transitions. The selected-range browser run also validates every path edge against Python;
its fit banks match the standalone fitter. See
[the browser check](../pose-fitter/benchmarks/browser-move-search.json).

## Fitted-pose preview

The video reconstruction computes Rust/WASM poses for move search while keeping
the overlay hidden during its first pass. It automatically enables **Show fitted
pose** when complete. The checkbox remains available for camera/images and for
hiding the completed overlay. Landmark/sticker points remain hidden, and each
review frame is presented with its matching saved projection. Saved detection data also includes `fittedPose`, its
candidate bank, sticker assignments, mesh, and fitting time when enabled.

Detection and fitting overlap through a CPU worker pool (up to four workers).
The app uses the complete validated CPU pipeline: OpenCV sticker/PnP startup,
initial geometry, immutable-neighbor refinement, color/continuity orientation,
and sparse depth/outlier recovery. Initial priors and orientation advance in
source order. Recovery buffers 180 ms of future source time. A twelve-frame
presentation queue applies backpressure without replacing admitted frames.
Each queued frame owns its raw canvas; rendering and saving happen in source
order after that frame's complete fit. Pause/model changes/seek flush the
available tail context; resuming starts a new fitting segment. Source replacement
invalidates old work. ROI reacquisition alone does not reset pose history.

Completed video results are retained for the loaded file. Scrubbing snaps to the
nearest completed selected frame, decodes its pixels, and presents the saved
detections and full fitted pose without rerunning either worker. Playback through
cached frames also reuses results and does not append duplicate recording rows.
The viewer labels these as **Showing saved result**. Fitted results remain reusable
when their overlay is hidden. Cache entries are separated
by model size, inference backend, thresholds, ROI padding and fitting mode;
unprocessed frames still run normally. Only results are retained, not full video
canvases. Clear, replacing the source or reloading the page releases this cache;
the JSON download remains available for saving results to disk.

The sidebar reports CPU computation and draw/presentation time separately. CPU
computation sums all numerical jobs for that frame; it excludes queueing and
lookahead waiting, and can exceed the frame interval because workers overlap.
Downloaded results include per-stage numerical times (`fittedPose.stageMs`), summed
CPU worker queue waits (`fittedPose.workerQueueMs`), and camera bitmap copy/worker
delivery times (`pipeline.camera.copyMs` / `deliveryMs`). These separate a slow
camera source from CPU saturation and delayed UI delivery. Worker elapsed times
include preemption by concurrent work; they are not exclusive CPU measurements.

Downloaded results include main-thread `pipeline` timestamps, `fittedPose.stage`
(`cpu_pipeline`), the content-addressed WASM version, source frame index and
fitting segment/configuration for standalone replay. FPS measures presentation
throughput over a window so simultaneous completions do not inflate it.

To profile CPU fitting on a saved diagnostic run, using fresh fits from its raw
detections and the shared scheduler:

```sh
cd web
npx tsx scripts/profile-live-pipeline.mts CAPTURE.json REPORT.json
# Optional exact fitted-output comparison against an older WASM asset:
npx tsx scripts/profile-live-pipeline.mts CAPTURE.json REPORT.json --baseline-wasm OLD.wasm
```

The report includes per-stage durations, input/runtime hashes and exact output
parity when a baseline is supplied. This isolates fitting; it does not measure
camera capture, GPU detection, concurrent search or browser worker scheduling.
Only recorded solve frames are replayed, so the unrecorded camera prelude is not
reproduced.

Upload and timer reconstruction share the Rust route generator, motion checks,
scorer and frontier. The timer adds source-window admission and scheduling. Search
candidate rows share immutable cached sticker permutations and move strings;
discrete cubie paths are reused across fitted shapes, with a 4,096-entry cache
bound. The fitted angles, motion limits, candidate ordering and search rules are
unchanged. Cached material matrices use exact-sized storage.

For a search-only comparison using the fitted-input format accepted by
`benchmark-parallel-search.mjs`, run from the repository root:

```sh
node web/scripts/benchmark-search-memory.mjs INPUT.json OLD.wasm NEW.wasm REPORT.json 4096 2
```

It alternates fresh workers, compares every search snapshot, and reports elapsed
search time and WASM allocated-memory high-water marks. These are isolated search
measurements, not live browser FPS or live-object heap sizes.

Build the complete fitter from the repository root before starting/building the
app. See [the fitter build instructions](../pose-fitter/README.md#build-and-run-without-the-app)
for building the OpenCV helper first, using official OpenCV/Emscripten sources.

```sh
rustup target add wasm32-unknown-unknown
cargo build --release --locked --target wasm32-unknown-unknown --lib \
  --features cpu-pipeline --target-dir pose-fitter/target/parity-wasm \
  --manifest-path pose-fitter/Cargo.toml
```

`npm run dev` / `npm run build` copy both WASM modules and cube geometry into
`public/pose/` with content hashes and a `pipeline.json` manifest. There is no
runtime CDN or pose-fitting server. Missing complete-pipeline assets produce a
visible error. The worker does not fall back to the older quad preview.

The pipeline uses the scripts' 55 mm cube and centered pinhole camera
(`focal = 1.2 * max(width, height)`). The app supplies the actual ROI pixel scale,
filtered detections, separate landmark-presence values and sticker color names.
The app has no configured static-hold boundary; all frames use articulated
search. Diagnostics can specify a boundary. Move search is outside the overlay.

Numerical stages live in Rust; scheduling lives in the framework-independent
`pose-fitter/runtime/buffered-pipeline.mjs`. Both run without Svelte. To verify
streaming against batch fitting or replay a downloaded browser recording:

```sh
node pose-fitter/tools/check-buffered-pipeline.mjs \
  pose-fitter/fixtures/full-pipeline/matty.json pose-fitter/results/matty-stream-check.json
web/node_modules/.bin/tsx pose-fitter/tools/check-browser-recording.mts \
  /absolute/path/cube-detections.json pose-fitter/results/browser-recording-check.json
```

The buffered scheduler matches the complete batch banks and recovery flags on
all 1,247 saved frames across five solves. The batch pipeline's six accepted
Python winner differences remain documented in the fitter report; this is not a
claim that fresh WebGPU detections equal cached Python detections. The earlier
initial-only preview's ~29 fps timing does not describe this complete pipeline.

## Export both models

From the repository root, in the existing Python environment:

```sh
.venv/bin/python -m pip install 'onnx>=1.16,<2' 'onnxruntime>=1.20,<2' 'onnxslim>=0.1.71'
.venv/bin/python scripts/export_web_detector.py
```

| Stage                 | Checkpoint                                | Export                             | Input                  | Raw output    |
| --------------------- | ----------------------------------------- | ---------------------------------- | ---------------------- | ------------- |
| Cube acquisition      | `runs/500ExamplesWeights.pt`              | `public/models/cube-bbox-640.onnx` | Up to 640, rectangular | `[1,5,N]`     |
| Pose (fast)           | `runs/pose/runs/26m-pose/weights/best.pt` | `public/models/cube-pose-160.onnx` | 160 × 160              | `[1,16,525]`  |
| Pose (default/search) | same                                      | `public/models/cube-pose-256.onnx` | 256 × 256              | `[1,16,1344]` |
| Pose (comparison)     | same                                      | `public/models/cube-pose-288.onnx` | 288 × 288              | `[1,16,1701]` |

`public/models/detector.json` records both stages, tensor contracts, thresholds,
tracking defaults, file sizes and source-checkpoint SHA-256 values.
`public/models/verification.json` records numerical parity checks. ONNX files
and the generated verification report are ignored by Git.

The cube model is about 10 MiB; each pose model is about 81 MiB. Both stages load
at startup, but only the selected pose resolution is fetched. The application
shell is separate from the model weights and the ONNX Runtime
bootstrap (roughly 24 MiB for WebGPU or 11 MiB for CPU). Only the selected runtime
is imported and initialized. Lower resolution reduces compute but
does not substantially reduce weight download size.

Use `--sizes 256` to export only the default pose size. `--model`, `--bbox-model`,
`--bbox-size`, and `--output-dir` override individual inputs. 256px becomes the default when exported; otherwise the first requested pose size
becomes the default. Sizes must be multiples of the model's 32px stride; 144px
is not an exported option. The bbox model keeps the native renderer's 640px input.

Exports are FP32, batch 1, opset 17, with embedded NMS disabled and
`end2end=False`. Pose exports have fixed square shapes. The cube model has dynamic
spatial dimensions so full-frame acquisition uses the native predictor’s
`rect=True` minimal stride-32 padding (for example, 1280×720 → 640×384). The exporter validates both schemas, checks each ONNX graph,
and compares raw output with PyTorch on deterministic inputs before updating
the manifest. Source checkpoints remain untouched.

The checked exports used Ultralytics 8.4.51, PyTorch 2.12.0, ONNX 1.23.2,
ONNX Runtime 1.32.0 and onnxslim 0.1.99. Maximum absolute raw-output differences
were 0.00067 (bbox), 0.00026 (160px pose), 0.00030 (256px pose), and 0.00046 (288px pose). These verify
conversion, not held-out detection accuracy.

## Acquisition and tracking

The worker ports `src/live_recognition/roi_tracking.py` and the independent
renderer's `tracking_bounds()` logic:

1. Without an ROI, letterbox the full frame to 640px and run the cube detector.
   Pick the highest-scoring valid cube above its separate 0.25 bbox threshold.
2. Make a square around that cube with padding 0.4 on each side, clipped and
   shifted inside the source frame. The initial padded side becomes the size floor.
3. Crop original source pixels; resize with area interpolation to the native
   renderer’s 256px `view_size`, then linear interpolation to the selected pose
   input. Normalize RGB to `[0,1]` and use FP32 NCHW. Resizing uses explicit tested
   pixel kernels rather than browser canvas scaling.
4. Decode the nine pose classes and one `x,y,presence` keypoint per detected object.
   Apply the global object-score prefilter (lowest class threshold), class-aware
   IoU NMS (0.7), and the 128-object limit, **then** the per-class object-score and
   presence filters. Match native keypoint clipping and FP32 scaling before
   mapping coordinates back to the frame.
5. Bounds of usable pose points move the next ROI with gain 1.4. Its side is the
   larger of the original padded side and the observed point extent padded by
   0.18 on each side. The cube model is skipped on tracked frames.
6. No usable tracking points resets the ROI. The next frame runs cube acquisition
   again. If no cube is found, pose inference is skipped and acquisition continues.

Tracking uses the renderer's default presence threshold of 0.05, with a 0.25
sticker floor, separately from the configurable display-presence threshold.
Both share the configurable pose object/class-score thresholds.

Changing source, seeking a video, changing resolution, changing bbox threshold
or initial padding, and clicking **Reacquire cube** reset tracking. Still images
reacquire on each evaluation so settings comparisons use the same two-stage path.
The acquisition score shown while tracking is explicitly labeled **Last cube
acquisition**; it is not a fresh bbox prediction on each frame.

## Frame processing (up to 30fps)

Frame selection uses source timestamps, independently of inference completion:

- 30fps or slower: process every source frame.
- 60fps: process frames 0, 2, 4, … (a 30fps stream).
- Other rates above 30fps: select at most one source frame per 1/30-second slot.

Uploaded videos use Mediabunny/WebCodecs sequential decoding instead of native
real-time playback callbacks. The decoder prepares one selected frame ahead
while inference processes the current frame.
A slow frame delays playback; it does not cause selected frames to be skipped.
Playback is paced to the source frame rate, capped at 30fps, and the viewer shows
an actual processed-frame count. Mediabunny loads only when opening a video file.
Seeking resets the processing count. Completed frames use their saved results;
unprocessed regions start a new fitting segment.

Camera capture runs independently in a worker through MediaStreamTrackProcessor.
The camera is requested at no more than 30fps. Selected frames are copied out of
the camera's native buffer pool and retained in arrival order, even while the
inference worker is busy. The viewer shows the remaining FIFO queue length.
The app never replaces an older buffered frame with the newest frame.

Live capture cannot slow physical time. A 60-frame buffer accommodates transient
inference stalls. If that buffer fills, processing continues and new incoming
frames are skipped until space opens. Every admitted buffered frame finishes in
order. Browser-reported discards are counted rather than stopping the stream.
The timer preview advances with completed processing and reports skipped frames.
This bounds raw-frame memory use. Cameras require a browser supporting frame
streams; unsupported browsers report the limitation rather than falling back to
presentation callbacks. Video files remain available through sequential decoding.

One frame is sent to inference at a time. The visible canvas holds the last
completed frame while the next frame runs inference. Point markers, object boxes
and score labels are not rendered. The optional tracked ROI outline is off by
default. Without that outline, `FramePresenter` presents the completed source
directly in one draw, avoiding the extra compositing copy. With the outline on,
it composes offscreen first. Toggling the outline uses the last completed raw
snapshot, including while another frame is pending; pending input cannot
clear/resize the display or appear before its inference completes.
Changing models finishes the in-flight frame and retains the file
cursor/camera FIFO. Overlay toggles redraw the current snapshot without running
inference or consuming another source frame. Clearing/leaving closes buffered
bitmaps, decoders and camera tracks. Pausing retains at most one prepared video
frame and stops camera admission; resuming drains admitted camera frames first. No
CPU-only session is automatically substituted when WebGPU is unavailable.

## Detection records

All processed camera/video source frames and evaluated still images retain their
complete `FrameResult` in this tab, including point coordinates, pose object/class
scores, keypoint-presence scores, object boxes, last cube acquisition, ROI/tracking
state and worker timings. Records include source identity, source timestamp, seek
segment, inference backend and the settings actually applied to that frame.
Switching or clearing the media does not discard earlier records in the tab.

Use **Download detection data** to save `cube-detections.json`. The export includes the model manifest/class mapping and labels
the pose object score, keypoint-presence score and separate cube bbox score
explicitly. Results are retained by reference without per-frame JSON serialization
or point copying; JSON is produced only when downloading. Records are held in
memory for this tab and do not survive a reload unless downloaded.

The app implements detection and ROI tracking. It does not yet port the native
renderer’s geometry recovery, orientation, move search or solve reconstruction.
The explicit area/linear kernels agree with the OpenCV pixel fixtures within one
8-bit level. Native-tensor fixtures match displayed detections and ROI updates
at both 160px and 256px. Browser video decoding and floating-point execution can
still produce small differences near thresholds; these are not bit-exact
end-to-end guarantees. To compare with the independent search command using only
`--roi-padding 0.4`, select **256 × 256 · Search script** and keep default thresholds.
The app starts at 256px; 160px remains available as the faster option. Camera capture now requests
640×480 at up to 30fps, matching the renderer defaults.

## Scores

These three values have separate meanings:

- **Cube bbox score:** the full-frame cube detector's object score, default 0.25.
- **Pose object/class score:** `result.boxes.conf` in the pose model. Thresholds
  are 0.25 / 0.40 / 0.25 for `corner`, `corner-edge`, `corner-edge-center`, and
  0.90 for each sticker color. Overlay decimals are these scores.
- **Keypoint presence:** `result.keypoints.conf`. The display default is 0.05
  for landmarks with a 0.25 floor for stickers. Presence can saturate near one;
  it is not localization confidence.

## Validate and build

```sh
npm test
npm run build
npm run preview
```

Tests cover input normalization, coordinate restoration into offset ROIs,
independent score filtering, class-aware NMS, cube selection and rejection,
acquisition/movement/miss/reacquisition examples from the native tracker,
30/60/120fps selection, FIFO ordering and slow-consumer preservation.
The build includes Svelte and TypeScript diagnostics. Native parity fixtures can
be regenerated from the repository root with
`.venv/bin/python scripts/verify_web_detector_parity.py`; this calls the actual
independent renderer’s preprocessing, prediction, filtering and tracker functions. With the dev server running,
`/tests/capture-check.html` runs a synthetic camera-stream check without requesting
camera permission: 60 uniquely numbered generated frames are processed through
the real capture worker with an 80ms consumer delay, checking complete FIFO order.
This developer test page is not included in the production build.
`/tests/frame-presentation.html` checks the shared presenter across deliberately
delayed detection results and multiple browser paints, overlay toggles, source
size changes, transparency and clearing. It verifies that pending raw frames
never replace the last annotated frame before presentation.

Deploy `dist/` to static HTTPS hosting, including `models/` and `runtime/` and
serving WASM with its normal MIME type. For CPU multithreading, serve every app
page with these response headers (already enabled for Vite dev and preview):

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Reload the page after enabling headers. The inference worker must report
`crossOriginIsolated` and SharedArrayBuffer/thread support must be available;
otherwise CPU mode falls back to one thread. No deployment is configured yet.

The viewer reports pose session time, total worker processing time, and processed
FPS. Worker time includes bbox preprocessing/inference when acquisition runs,
plus pose preprocessing/inference and decode. FPS also includes capture and main
thread scheduling. Download, compilation and initial warmup are excluded. Compare
resolutions using the same source and settings; point count alone is not accuracy.

The **Detector worker** metric excludes the rest of the pipeline. Expand **Frame
timing** beneath the metrics for the last frame's start-to-start interval, source
wait, source drawing, worker round trip, overlay drawing, and pacing wait, plus
worker resize/pixel-read times. Pacing wait includes timer scheduling latency;
phase timings span adjacent frame cycles and are not a breakdown that sums to the
current worker time. Processed FPS remains visible when playback finishes.

Video decoding now prepares exactly one selected frame ahead while inference is
running, with sequential consumption and no eviction. Seek/close dispose pending
frames, including reads that complete after cancellation. Frame pacing starts
before source acquisition/drawing, so these costs count toward the interval;
overdue frames dispatch without another zero-delay timer task. The display canvas
is resized only when its dimensions change.

A local full-UI run of `tommySolve3.mp4` (1920×1080, 59.94fps) at 160px WebGPU
processed all 161 selected frames. The final displayed rate was 28.1fps, with
34.3ms worker time and 35.3ms frame interval. That frame spent 26.4ms in pose
inference, 4.3ms resizing and 3.3ms in pixel reads; source wait was below 0.1ms.
This real-video result remains slower than 30fps despite the faster resize and
source overlap. A 30.5ms static-scene worker benchmark does not guarantee sustained
30fps through the full app.
After removing the extra timer task for overdue frames, another run finished at
29.5fps, with all 161 selected frames processed, 32.1ms last-frame worker time and
a 33.8ms start-to-start frame interval. These separate runs are not a controlled
inference-speed comparison; they demonstrate full-app throughput and the narrow
remaining frame budget on this hardware.
The same clip at 256px finished at 24.7fps with all 161 selected frames processed.
Its last frame took 39.6ms in the worker (30.1ms pose inference, 3.9ms resize,
5.0ms pixel read) with a 39.8ms frame interval and no deliberate pacing wait.
Larger-input inference plus pixel preprocessing exceeded the 33.3ms budget even
with video lookahead; this was not a frame-rate-limiter bottleneck.

## CPU performance comparison

The original CPU option explicitly disabled threading and used the WebGPU runtime
build. CPU mode now imports `onnxruntime-web/wasm`, loads its matching SIMD/pthread
binary, and uses up to four threads when isolation is available. WebGPU retains
its separate runtime and one WASM orchestration thread. Both CPU sessions warm
up three times before the app reports ready.

For a repeatable local comparison, run from the repository root:

```sh
.venv/bin/python scripts/benchmark_web_cpu.py
```

Then open `/tests/cpu-benchmark.html` on the dev server and click **Run CPU
comparison**. It uses the exact same exported models and input tensors as the
native benchmark. Each browser case uses its own worker/runtime, three warmups,
and nine measured runs; medians measure session inference, excluding download,
preprocessing, NMS and drawing. This developer page is excluded from production.

One local run produced these warmed median milliseconds (timings vary by device
and concurrent load):

| Pose input | Old WASM, 1 thread | CPU WASM, 4 threads | PyTorch, 4 threads |
| ---------- | -----------------: | ------------------: | -----------------: |
| 160px      |              125.0 |                41.9 |               25.2 |
| 256px      |              290.9 |                82.9 |               58.3 |

The CPU-specific runtime with one thread measured 120.4/281.9ms. Thus the thread
pool explains most of the improvement; the runtime-build change alone was small.
Browser WASM and native PyTorch still use different compiled CPU implementations.
This does not guarantee 30fps CPU throughput. Selected frames remain sequential
and the frame-buffer policy applies when inference takes longer.

## GPU frame-processing profile

Open `/tests/gpu-frame-profile.html` on the dev server to profile the actual app
worker using a 1280×720 scene with a 360px cube image. Each model size runs one
acquisition frame, three warmup tracked frames and ten measured tracked frames.
The page reports component medians and the raw per-frame results. This developer
page is excluded from the production build.

For CPU preprocessing, `FrameResult.profile` separates each model's canvas draw/pixel read, CPU resize,
RGBA-to-NCHW tensor preparation, `session.run`, and output copy/tensor cleanup.
It also separates bbox decoding, pose decoding, tracking decoding, and ROI update.
`nonInferenceMs = totalMs - bboxMs - poseMs`. Session time includes ONNX Runtime's
input/output transfers and synchronization; it is not a shader-only GPU measurement.
Capture, source video decoding, queue wait, message transfer, and UI drawing remain
outside the worker's frame timer.
For GPU preprocessing, `gpuSubmitMs` measures CPU command submission, not GPU
execution. Kernels are submitted on ONNX Runtime's own device/queue with no
intermediate CPU readback or extra fence. Session elapsed time includes remaining
queued preprocessing work, so the UI labels this **Pose + GPU preprocess** rather
than treating it as inference-only timing. The total worker timer includes the
complete critical path. Use the isolated benchmark below for synchronized
preprocessing timings.

Before GPU preprocessing, one local run with an approximately 608px tracked ROI measured 13.1/13.3ms median
non-inference overhead at 160/256px pose input. Resizing accounted for about
9.5–10.3ms, canvas/pixel reads for 2.4–2.8ms, and decoding/tracking for 0.4–0.6ms.
Tracked total frame medians were 37.6/45.9ms, including 24.4/32.1ms pose session
medians. Component medians need not sum to the median total. Initial acquisition
was more expensive because the full-frame cube model and its preprocessing also
ran. Values depend on input/crop size, browser, GPU, and current system load.

## WebGPU optimization comparison

Open `/tests/webgpu-options.html` and click **Compare WebGPU options** to compare
the current CPU-backed tensors, reusable GPU input/output buffers, and reusable
GPU buffers with graph capture. Each case uses its own worker, the same FP32
input, four warmups, and eleven measured runs. Timings include input upload and
output readback so GPU-buffer cases do not hide synchronization costs. This is
an inference-only developer benchmark, excluded from the production build.

One local run measured these median times:

| Pose input | CPU-backed tensors | GPU buffers | GPU buffers + graph capture |
| ---------- | -----------------: | ----------: | --------------------------: |
| 160        |            24.52ms |     24.47ms |                     24.96ms |
| 256        |            30.67ms |     30.66ms |                     31.52ms |

All six cases returned identical outputs for this test input. This short run
shows no useful improvement from changing I/O allocation or enabling graph
capture alone; production inference settings remain unchanged. GPU buffers can
still matter when preprocessing writes model inputs directly on the GPU.

That comparison identified CPU image preprocessing as the largest target. The shared native
tracker and browser tracker both use square ROIs rounded to whole pixels; neither
snaps ROI sides to integer multiples of the 256px intermediate view. Such snapping
could permit specialized integer-ratio area resizing, but changes the crop and
cube scale and needs recognition validation. Merely rounding to a multiple of 32
does not guarantee an integer reduction into 256px. The new GPU preprocessing path
preserves the native area-to-view, then linear-to-model resize and intermediate
8-bit rounding. Tensor preparation and postprocessing were much smaller targets.
Capture, queue latency, and UI rendering need separate end-to-end measurements.

## Optimized CPU resizing

The production resize kernels cache coefficient plans in bounded 32-entry caches,
reuse internal float32 scratch buffers, share source indexing across RGB channels,
and accumulate the vertical area pass over contiguous rows. Returned image buffers
remain independently owned. ROI dimensions, interpolation order, float32
accumulation order, intermediate 8-bit rounding, and model inputs are preserved;
there is no new integer-ratio shortcut or ROI size snapping.

Open `/tests/resize-benchmark.html` for an alternating comparison against the
frozen original kernels in `tests/resize-reference.ts`. It uses eight warmups and
21 measurements per implementation and checks every output byte. One local run:

| Resize path                    |  Before |  After | Speedup |
| ------------------------------ | ------: | -----: | ------: |
| 608px ROI → 256 → 160          |  8.83ms | 3.22ms |   2.74× |
| 591px ROI → 256 → 160          |  8.54ms | 3.22ms |   2.65× |
| 608px ROI → 256                |  8.31ms | 2.88ms |   2.88× |
| 721×719 ROI → 256 → 288        | 11.94ms | 4.85ms |   2.46× |
| Bbox linear 1280×720 → 640×360 |  3.83ms | 2.92ms |   1.31× |

Maximum pixel difference was zero in every case. Regression tests also cover
upsampling, single-pixel axes, cache eviction, scratch reuse, and native fixtures.
The actual app worker's GPU profile on the same scene measured tracked 160px
resize/total medians of 3.66/30.46ms (previous run 10.27/37.64ms); at 256px these
were 3.84/39.33ms (previously 9.48/45.93ms). All detections and ROI sequences matched
the previous profile exactly. These full-worker results are separate local runs,
so inference and other timing variation also contribute to the total difference.
Capture, queueing, and UI rendering are excluded; this is not an end-to-end 30fps
guarantee.

## GPU preprocessing and parity

`src/lib/gpu-preprocess.ts` copies the native-resolution ROI directly from the
frame bitmap to a GPU texture. Pose preprocessing runs separable area filtering
into the 256px view, performs the existing 8-bit rounding, then uses the same
11-bit fixed-point linear coefficients and final rounding as the CPU/native
path. RGB normalization uses an FP32 lookup table and writes NCHW directly into
an ORT GPU input buffer. Cube acquisition uses the same linear resize and
rectangular 114-valued padding as before. Crop geometry, checkpoints, precision,
thresholds, NMS and tracking are unchanged; ROI dimensions are not snapped.

Float32 materialization between area multiply/add operations prevents compiler
FMA contraction from moving pixels across rounding boundaries. Coefficients and
GPU buffers are reused with bounded plan caches. Tensor wrappers are refreshed
on shape changes, including landscape/portrait shapes with equal element counts.
The preprocessor owns its GPU buffers; ORT wrappers never dispose them. WebGPU
preprocessing is the default for camera/video; `preprocessing: "cpu"` on worker
initialization is a developer benchmark override, not a second inference backend.

Browser checks (developer pages, excluded from the production build):

- `/tests/gpu-preprocess.html`: raw input tensor parity and isolated preprocessing
  timing, including upload and a completion fence. Opaque cases cover 608/591/721px
  crops, 512→256 rounding, identity/upscale cases, all three pose sizes, and
  landscape/portrait bbox padding. All opaque test tensors matched the CPU path
  exactly on the tested browser/GPU. Transparent bitmap conversion can differ by
  one 8-bit level, so production still-image requests use the original CPU path.
- `/tests/gpu-state-check.html`: alternating landscape/portrait frames, alpha
  preservation via CPU preprocessing, then resuming GPU preprocessing/tracking.
  All six output/ROI comparisons matched the CPU reference exactly.
- `/tests/gpu-video-benchmark.html`: choose the same video used in the app. Runs
  CPU versus GPU preprocessing with WebGPU inference at 160 and 256, compares
  exact per-frame detections/ROIs and selected frame counts, and measures uncapped
  processing capacity. Includes decode wait, snapshot/display draws, worker round
  trip and overlay drawing; excludes four initial frames from timing. This is
  not paced UI playback; normal playback is still capped at 30fps.

On `tommySolve3.mp4` (1920×1080, 59.94fps), all four passes processed the same 161
selected frames. GPU and CPU preprocessing returned identical detections and ROI
sequences at both resolutions, with zero changed frames:

| Pose input | CPU preprocessing capacity | GPU preprocessing capacity | CPU worker median | GPU worker median |
| ---------- | -------------------------: | -------------------------: | ----------------: | ----------------: |
| 160        |                   28.44fps |                   37.46fps |           32.68ms |           26.05ms |
| 256        |                   24.23fps |                   31.17fps |           39.86ms |           31.59ms |

Isolated preprocessing for a 608px ROI → 256 → 160 measured about 5.02ms CPU
versus 0.82ms GPU, including bitmap upload and synchronization. These are local
measurements, not a guarantee across hardware, browsers or footage; 256px still
has a narrow 30fps margin. No model precision reduction or frame dropping was
used to obtain these results.

Normal Svelte app playback of the same clip finished with displayed rates of
29.1fps at 160px and 28.3fps at 256px, with all 161 selected frames processed in
both runs. Last-frame worker times were 24.6ms and 33.1ms respectively. These UI
runs include pacing and Svelte/overlay/rendering work that the uncapped developer
benchmark does not reproduce exactly; 256px is improved but not locked to 30fps.
