# CPU pose fitter: Rust and WASM

## Sparse recovery reprojection weighting

Depth-recovery proposals now rank with four times the capped observation-loss
weight. Pose priors and the existing per-active-layer penalty remain unchanged;
the camera optimizer and number of fitting passes are unchanged. This intentional
ranking change follows the reviewed Jeff1551 pose at 8.687 seconds, where the
correct articulated candidate previously ranked behind a rigid candidate.
`bank[].recovery_score_components` records the observation loss, pose-prior loss,
active-layer count, observation count and reprojection weight separately. Ordinary
geometry scores are unaffected. The original Python reference retains its old
recovery ranking for comparison.

The acceptance check preserves sticker assignments in the five regular geometry
options, after aligning equivalent cube frames; the sixth flat fallback does not
count toward that check. Every changed previous primary retains its assignments
in the top five across the eight paired raw-observation datasets. Tommy2 frame
710's reviewed correct old fit remains candidate 2. Matty, Nahm and Tommy2 also
still reconstruct after fresh browser-input fitting with the current search
settings, with unchanged state counts. See
[`benchmarks/recovery-reprojection-weight.json`](benchmarks/recovery-reprojection-weight.json)
for scope and results. Full timestamp-matched wireframe videos for the six new
failed browser captures are in `results/recovery-weight/videos/`.

## Incremental move search

`move_search::Engine` supports streaming forward search and post-fitting
bidirectional search over the same scorer and mechanical transitions. The
standalone Python implementation is `search_independent_keypoint_bidirectional.py`.
Both rate limits are fixed at 54 TPS; the separate double-turn classification is
removed. Input banks retain the five assignment-distinct fits plus reserved flat
sixth option. Identical shape-and-assignment contexts are collapsed by the same
rule as the CPU scorer, preserving their original bank indices.

- `move_search/cube.rs`: immutable cube conventions and permutation composition.
- `move_search/routes.rs`: relative-slab motion, shared angular budgets, seam
  crossings, hidden moves, face/wide/slice notation; no future double witnesses.
- `move_search/prepared.rs`: state-independent bank validation, context selection,
  per-sticker color-cost aggregation and adjacent-frame transition compilation.
- `move_search/frontier.rs`: state scoring, stable rank/tie rules, dominance,
  shared paths, mismatch history, parked frontier and deferred widening.
- `move_search/mod.rs`: typed session input and independent `push`, `advance`,
  `end`, `snapshot` and `probe` operations. Out-of-order input prepares immediately
  and waits for its missing earlier sequence before linking transitions.

The WASM ABI `move_search_json` accepts `init`, `frame`, `advance`, `end`, `status`
and `probe`. Waiting for input consumes no expansion budget. `advance` is bounded
for cooperative workers. Configuration, full input banks and transitions are
serializable; diagnostics run with neither the app nor OpenCV.

Set `bidirectional: true` in the search config to buffer every fitted frame and
start both frontiers only after `end`. The browser's **Bidirectional video
search** option enables this by default for videos; turning it off retains the
streaming forward search. Camera search remains forward. `MoveSearch.end()`
waits for its fitting promises and complete frame sequence before closing the
worker input, and the worker schedules no expansions before that closure.
Existing serialized configs default to `bidirectional: false`.

`move_search/bidirectional.rs` starts from the scramble and from its inverse in
all 24 solved cube orientations. It chooses the start of the last corroborated
cluster of flat solved views, requiring at least two frames with at least 12
assigned sticker observations and color error at most 0.1. These are color-fit
criteria, not detector confidence thresholds. Without a corroborated anchor it
falls back to forward search after fitting and reports that fallback in the UI.
Special handling of timer-stop drops or long unreliable trailing pose runs is
not part of this change.

Pose-less opening frames are skipped: the initial motion clock advances through
them, so the first usable pose gets only the interval since the last skipped
frame. They cannot accumulate a large hidden-turn allowance before the cube is
observed. Missing poses after the first usable pose retain the full elapsed
motion allowance. This rule applies to both forward and bidirectional search,
including out-of-order fit completion. Search reports `running` as soon as its
workers finish preparation, before the first expansion batch returns.

The browser runs two Rust/WASM instances in separate workers. Direction zero
calls the original forward frontier: its candidate admission, dominance, queue
order and full state allowance are unchanged. Direction one independently
inverts the existing forward mechanical rows, retaining the 54 TPS turn/combined
limits, configured slice limit and rigid-rotation limit. Reverse ties use the
predecessor frame's color agreement. Each direction has the configured state
limit, so total expansions can reach twice that limit. Workers still compete
for CPU and memory, and preparing a second instance and exchanging meetings
adds overhead; independent queues do not guarantee lower elapsed time.

Workers exchange admitted states as fixed 64-byte records. Meetings require the
same frame depth, sticker state and geometry cue after cube-orientation
canonicalization. Rust rebases the backward suffix into the forward cube basis,
validates every suffix edge against the existing mechanical compiler and checks
the final solved cube. A rebased edge can lie outside the forward scorer's
orientation shortlist; validating that edge does not add it to the forward
frontier. Histories and best-first priorities remain separate. This remains a
heuristic search, without an optimality guarantee. A backward-worker failure
leaves forward search running. Cancellation terminates both workers.

The parallel API adds `advance_side`, `path` and `complete_join`; the action
journal records directional chunks and the selected join for replay. The serial
`advance` API remains available. Result statistics include `forward_expanded`,
`backward_expanded`, `solved_anchor_frame`, `join_frame_index` and
`search_direction`. Tests check unchanged forward progress while backward work
runs, independent budget exhaustion, cube-basis rebasing, final goals and
worker cancellation/failure. Diagnostics-off retains the essential solution.

The geometry-derived one-second gate is **experimental and disabled in
production**. The isolated comparison checks forward, reverse and join-crossing
windows using `min(54, 1.25 * geometry_TPS + 2)`. It is not consistently beneficial
on the cached cohort; see `results/bidirectional/parallel-benchmarks/` for inputs,
raw trials, CSV results and the temporary gate builder. Its geometry feature
uses the previously evaluated strict greater-than-six-degree commit estimate.
The earlier Kyle comparison remains a strong positive result under cached
travel accounting: 57,642 to 13,051 expansions, preserving exact steps. A control
restoring only that travel calculation reproduces 13,051 in the new experiment.
That result must be distinguished from trials using the current shared-motion
travel calculation; the change predates bidirectional search.

The fresh 97-frame Yiheng277 capture reconstructs with approximately 21,000
combined expansions using the parallel coordinator; forward alone reaches its
50,000-state cap. Parallel totals and elapsed times vary with worker completion
order. Independent final cube checks pass. A mechanically legal solved path is
not a full manual review of move timing. Reproduce a production parallel trial
with already fitted input banks:

```sh
node pose-fitter/tools/benchmark-parallel-search.mjs \
  pose-fitter/results/geometry-tps/search-experiment/yiheng277-fresh/input.json \
  pose-fitter/results/bidirectional/yiheng277-parallel.json parallel --states 50000
node pose-fitter/tools/replay-parallel-journal.mjs \
  pose-fitter/results/bidirectional/yiheng277-parallel.json \
  pose-fitter/results/bidirectional/yiheng277-replayed.json
```

```sh
# Immutable conventions only; never includes fitted poses or scored rows:
.venv/bin/python scripts/export_move_search_tables.py

# Generate Python forward-search oracles using identical saved CPU fit banks:
.venv/bin/python scripts/export_move_search_fixtures.py --states 25000
cargo build --release --locked --features cpu-pipeline \
  --bin move-search-benchmark --manifest-path pose-fitter/Cargo.toml
.venv/bin/python scripts/check_move_search_parity.py
.venv/bin/python scripts/check_move_search_parity.py --wasm

# Same engine, incremental arrivals and independently checked Python edges:
node pose-fitter/tools/run-move-search.mjs \
  pose-fitter/fixtures/move-search/matty.json pose-fitter/results/matty-move-search-stream.json stream
.venv/bin/python scripts/validate_streaming_move_search.py
.venv/bin/python scripts/validate_streaming_move_search.py --browser-recording /absolute/path/cube-detections.json
```

The fixed-input tests check the same chosen geometries, moves, candidate scores,
queue priorities, solved boundary and expansion/pruning counts on five solves.
Matty, Nahm and Tommy solve within the 25,000-state benchmark budget; Xuanyi and
Yiheng hit that same budget in both runtimes. They are not claimed solved. See
[WASM timings/parity](benchmarks/move-search-wasm.json),
[native timings/parity](benchmarks/move-search-native.json) and
[the streamed Matty check](benchmarks/matty-move-search-stream.json).
The small checked-in CPU oracle also exercises deferred widening with widths
1/2/3 and a short expansion budget. Native tests cover explicit EOF, parked
frontiers, out-of-order preparation and six separate projection meshes.

Streaming retains alternatives but, by default, pauses exploration once an
admitted path reaches the newest contiguous searchable frame. New frames resume
exploration; EOF releases the pause for remaining backtracking. Set
`pause_at_live_frontier: false` to continue exploring while waiting for frames.
The app also enables `expand_flagged_frames`. The read-only reliability check
uses existing robust geometry error, inlier counts and observation counts; it
does not fit additional poses. Flagged frames admit every legal move/geometry
continuation under the existing 54/54 TPS rules, including shared move sequences
with distinct geometry contexts. Their scores do not affect queue priority or
the mismatch window; the next unflagged frame resumes normal queuing. No undo
filter is applied. Standalone parity fixtures leave this option disabled unless
explicitly testing it. A frame can provide `landmark_count` and optionally
override automatic classification with `flagged`.
Its candidate scores and legality remain the reference rules. Reported CPU/WASM
times exclude detection, fitting, module loading and file I/O; they are local
single-pass diagnostic measurements, not whole-app FPS promises.

## Complete standalone CPU pipeline

### Flag-aware PnP continuity selection

The browser and `pipeline_json` use the same 25% image-plane core-displacement
rule. Among finite raw PnP roots, prefer the existing <=0.1 reprojection-metric
pool when it is nonempty. Keep its visual winner unless its projected core jump
exceeds 25% of its projected cube diameter **plus a motion allowance**, and the
closest eligible root reduces the jump by more than half. No depth/world-space
displacement participates and no extra PnP solves or geometry fits are run.

The reference is the most recent eligible **finalized, unflagged** fitted core.
Eligibility has a fixed source-time boundary: the frame is more than 180 ms old
and its following initial neighbor precedes the current frame. This avoids a
dependency cycle with recovery and makes the result independent of worker speed.
Flags use the same existing residual/inlier/count rule as move search. Missing
or flagged poses cannot enter the reference history. Two unflagged finalized
cores are required; their screen-plane speed times the reference age supplies
the motion allowance. Until then, keep the visual ranking. The allowance avoids
mistaking ordinary motion during the older reference's lag for a PnP jump.

`src/pipeline/sequence.rs` interleaves the unchanged numerical stages once per
frame for synchronous replay; `runtime/buffered-pipeline.mjs` schedules those
same operations across workers. Saved poses expose `pose_flag_reason`,
`continuity_core`, `startup_reference_core`, and `startup_motion_allowance` for
diagnostics. Batch stage checkpoints retain the per-frame reference cores and
allowances. The isolated `startup` stage requires explicit reference cores;
use `pipeline_json` for the automatic finalized-reference policy. The pure
threshold/root-selection oracle remains separately reproducible:

```sh
node pose-fitter/tools/check-quarter-pnp.mjs
node pose-fitter/tools/check-guarded-pnp.mjs
node pose-fitter/tools/check-buffered-pipeline.mjs INPUT.json OUTPUT.json
```

The `cpu-pipeline` feature adds the complete standard joint-pose pipeline outside
the browser app: per-frame sticker grouping and OpenCV camera hypotheses,
initial geometry, immutable-neighbor refinement, color/continuity orientation,
and sparse depth/outlier recovery. Search grids, retained candidates, seed
budgets, solver bounds, losses, iteration limits and stopping criteria are kept.
OpenCV 5.0.0 IPPE/SQPnP/P3P/AP3P and pivot LM kernels are compiled to a separate
WASM library; their algorithms are not replaced by the earlier quad initializer.

The final saved-input replay covers **1,247 selected frames in five solves**.
Rank-zero sticker assignments agree on 1,241 frames. The remaining six frames
were reviewed and accepted by the user, including top-five overlays. There are
no additional differing winners, all recovery decisions agree, and every bank
passes the five-unique-assignment-plus-reserved-flat policy. See
[the full report](benchmarks/full-pipeline-parity.json) and
[the explicit accepted cases](benchmarks/accepted-pipeline-differences.json).
This is an accepted close match, not exact or bitwise equality. Complete
candidate-bank equality is not claimed: ordered banks differ on 92 frames;
assignment-pattern sets differ on 87, including frames whose winner agrees.

This is a controlled fitter comparison, separate from the historical render
audit below. Python is re-run on the *same* source-time-selected frames as WASM,
including the available clip prefix and the configured static boundary. Both
receive the same filtered detections, colors and ancillary inputs. Historical
logs omit landmark presence values and the detector ROI: presence is explicitly
set to one in these fixtures, and the saved-diameter pixel-scale fallback is
passed to both implementations as a scalar. No reference camera/PnP seed or
fitted pose is passed into the autonomous WASM rollout. Fresh detector/ROI
equivalence and final move-search selection are outside this comparison.

The Svelte checkbox now uses this complete pipeline, including OpenCV startup,
neighbor refinement, orientation and depth recovery. A framework-independent
buffered scheduler shares the numerical stage entry points with the standalone
batch runner. It matches complete batch banks and recovery flags on all 1,247
saved frames across the five solves. This scheduling check uses identical
inputs; fresh browser detection/ROI differences remain a separate concern.

The app uses up to four compute workers, an ordered twelve-frame presentation
queue, and 180 ms of future source context for recovery. The causal initial and
orientation passes preserve source order; neighbor jobs use immutable initial
winners. Pause, seek and settings boundaries flush the available tail. See
[the app documentation](../web/README.md#fitted-pose-preview) and
[the browser integration check](benchmarks/browser-integration-check.json).

### Module boundaries and tests

- `src/startup.rs`: sticker grouping, class-consistent correspondence sequences,
  camera hypotheses and ranking. `landmark_presence` explicitly means the pose
  model's keypoint-presence channel; detections have already been filtered by
  their separate per-object class scores before reaching this module.
- `src/opencv.rs`, `tools/opencv-pnp.cpp`: camera-kernel ABI; no hypothesis
  selection or sequence policy is hidden in the bridge.
- `src/search.rs`, `src/projection.rs`, `src/solver.rs`: geometry search,
  reprojection/assignments and bounded TRF solver. The same TRF engine serves
  ordinary soft-L1 fits and the linear-loss, two-stage depth recovery.
- `src/pipeline/mod.rs`: explicit input, configuration and checkpoint types,
  plus independent startup, initial, neighbor, orientation and recovery passes.
- `src/pipeline/orientation.rs`, `depth.rs`, `assignment.rs`: orientation,
  depth trend/recovery and padded one-to-one matching.
- `src/pipeline/jobs.rs`: individual CPU stage operations for buffered hosts.
- `runtime/buffered-pipeline.mjs`: framework-independent ordering, neighbor
  lookahead, recovery windows and flush/cancel semantics.
- `runtime/wasm-host.mjs`: shared OpenCV/Rust ABI used by Node and browser workers.

None of these modules imports Svelte, a detector, a video decoder or browser UI
state. Rust unit tests use saved CPU oracles for grouping, orientation and
fixed-input depth solves, alongside projection/search/TRF tests. WASM startup
tests exercise the actual OpenCV kernels. A full replay checks the documented
acceptance gate; alternative-bank differences remain visible in its report.
The complete camera-backed pipeline currently executes through the WASM host;
native tests exercise the pure Rust stages without installing OpenCV bindings.

### Build and run without the app

Use the official OpenCV 5.0.0 source tree and Emscripten 4.0.20. The task-local
copies are under `/private/tmp/live-recognition-opencv/`. Third-party notices
and provenance are in [vendor/opencv](vendor/opencv/README.md).

```sh
.venv/bin/python pose-fitter/tools/build-opencv-wasm.py \
  --emsdk /private/tmp/live-recognition-opencv/emsdk-main \
  --source /private/tmp/live-recognition-opencv/opencv-5.0.0 \
  --build-dir /private/tmp/live-recognition-opencv/build-wasm

cargo build --release --locked --manifest-path pose-fitter/Cargo.toml \
  --features cpu-pipeline --target wasm32-unknown-unknown --lib \
  --target-dir pose-fitter/target/parity-wasm
cargo test --release --locked --features cpu-pipeline --manifest-path pose-fitter/Cargo.toml

node pose-fitter/tools/check-startup.mjs \
  pose-fitter/tests/fixtures/startup-golden.json pose-fitter/results/startup-golden-check.json

# Recompute the Python reference, with independent stage oracles if desired:
.venv/bin/python scripts/export_pose_pipeline_fixtures.py --trace
# Replay all five solves and fail on a new winner/policy/recovery regression:
.venv/bin/python scripts/check_pose_pipeline_parity.py
```

The complete pipeline build has its own target directory. The app's runtime-sync
step publishes it under a content hash; the earlier preview module remains
separate for historical diagnostics. To capture and resume a stage:

```sh
node pose-fitter/tools/run-pipeline.mjs \
  pose-fitter/fixtures/full-pipeline/matty.json /private/tmp/matty-stages.json --capture-stages
node pose-fitter/tools/run-pipeline.mjs \
  pose-fitter/fixtures/full-pipeline/matty.json /private/tmp/matty-orientation.json \
  --stage orientation --checkpoint /private/tmp/matty-stages.json --checkpoint-stage neighbors
```

Other stage names are `startup`, `depth_targets`, `initial`, `neighbors` and
`recovery`. Checkpoints retain prerequisites and are checked against the input
fixture hash. Resuming orientation from a captured neighbor checkpoint produces
the same complete banks as the uninterrupted run. Renderers consume saved JSON
only and run neither inference nor fitting:

```sh
MPLCONFIGDIR=/private/tmp/live-recognition-mpl .venv/bin/python scripts/render_pose_pipeline_differences.py
MPLCONFIGDIR=/private/tmp/live-recognition-mpl .venv/bin/python scripts/render_pose_pipeline_top_five.py
```

### Standalone timing

One sequential Node/V8 WASM worker, single pass, module loading excluded;
timings include JSON exchange and the complete camera/geometry/orientation/
recovery computation. These are diagnostic batch timings, not app throughput.

| Solve | Selected frames including prefix | Complete pipeline seconds |
| --- | ---: | ---: |
| Matty | 289 | 9.69 |
| Nahm | 384 | 10.90 |
| Xuanyi | 235 | 5.37 |
| Tommy | 221 | 15.06 |
| Yiheng | 118 | 9.40 |

Neighbor refinement remains the largest stage. The report records per-stage
times and binary/input hashes. Camera WASM is approximately 1.77 MB and Rust
pipeline WASM approximately 0.81 MB, uncompressed. The browser budgets the
complete pipeline; the earlier initial-only preview's FPS is not a timing
claim for this implementation.

---

The sections below document the original numerical port, historical tests and
the earlier browser preview. Their measurements have the stated narrower scope.

Standalone port of `scripts/independent_keypoint_pose_geometry.py::fit_frame` and
its CPU geometry/scoring dependencies. The original **Show fitted pose**
preview used only the initial-fit stage; the integration above supersedes it.
The core entry point is `cube_pose_fitter::fit_frame(&Geometry, &Workload)`;
both inputs are serializable. The standalone WASM ABI loads a fixture set once
and runs indexed workloads repeatedly in a persistent worker.

**Full-window parity is not yet achieved.** The later comparison across Matty,
Yiheng and Tommy found candidate-bank differences and some winning-fit
differences that the original 20-workload sample missed. See
[the full-window summary](benchmarks/sequence-parity.json) and the generated
overlays in `results/sequence-parity/`. The port remains experimental.

## Current bank selection

The final bank keeps the best representative of up to five distinct sticker
assignment patterns, then appends the best available flat (zero slab-angle)
fit as a reserved fallback. Validity flags are part of the assignment pattern;
invalid placeholder indices are ignored. Cube-symmetry relabelings are aligned
to the best fit's frame before comparing patterns. Fewer than five available
patterns are kept without padding. The flat slot is outside this quota and may
duplicate a pattern or pose in the first five. Static-only calls return one
flat fallback. Python orientation/depth-recovery banks and downstream replay
selection preserve this policy and the reserved flat slot.

The earlier timing/parity/refit-gate reports below describe the previous
camera/angle-diversity policy. In particular, the earlier unanimous-bank skip
estimate does not transfer unchanged to an intentionally assignment-diverse bank.

## Preserved numerical behavior

- All three articulation axes, the complete −45°…45° grid at 15° intervals,
  the separate shared-zero proposal, and both signed 45° interpretations.
- Middle-slab folding, cube symmetries, class-compatible nearest landmarks,
  seam duplication, front-face tests and slab-ray occlusion.
- CPU sticker-center scoring, uncertainty weights, assignment rays and polygon
  containment. Already filtered observations are supplied unchanged; the fitter
  does not reinterpret detection or keypoint-presence scores.
- Diverse per-axis/per-active-slab seed selection, the same seed budget,
  retained unrefined sweep measurements, and the original 0.01 cutoff ties
  for internal seed/sweep selection. Final bank selection is described above.
- The bounded exact-SVD **TRF** solver used by SciPy, including the analytic
  Jacobian, `soft_l1` loss with scale 3, bounds, variable scales, priors,
  fixed/free angle masks, 18-evaluation budget and default 1e-8 stopping tests.
- Static six-variable solves and temporal calls that retain the original bank
  and sweep the new neighboring cameras.

The numerical implementation uses float64, with the CPU's float32 projected
corner/enclosing-circle normalization. It uses no search pruning, smaller
iteration budget, approximate solver, GPU fitter, or relaxed floating point.
The Rust implementation uses nalgebra's SVD instead of SciPy's LAPACK SVD;
numerical equivalence is tested, rather than assuming bitwise equality.
Algorithm notices are in [THIRD_PARTY.md](THIRD_PARTY.md).

## Historical measured results

Local ARM64 macOS run, Rust 1.100.0 release/LTO, SciPy 1.17.1, PyTorch CPU with
two threads, one native solver thread / one browser WASM worker. Five measured
passes follow a full warm-up for each workload. CPU/native/Node WASM runs are
interleaved with rotating execution order; Chromium 156 is measured separately.
The table shows the median of the per-workload medians, in milliseconds.

| Fit call | Python CPU | Native Rust | Browser WASM |
| --- | ---: | ---: | ---: |
| Initial geometry (10 workloads) | 241.50 | 17.17 | 22.25 |
| Temporal neighbors (5 workloads) | 537.29 | 36.70 | 46.60 |
| Static geometry (5 workloads) | 4.26 | 0.26 | 0.30 |

Browser initial fits ranged from 18.0–28.1 ms and temporal fits from 43.6–55.9 ms
across the workload medians. The WASM module is 480,434 bytes uncompressed.
Instantiation plus fixture loading took 10.9 ms; JSON result encoding/copying/
decoding averaged 0.157 ms per call, measured separately from fitting.
Browser timers are quantized, so sub-millisecond static results are approximate.

The five saved reference runs are Matty, Nahm, Xuanyi, Tommy and Yiheng, selected
using the mechanical CPU benchmark's sampling policy. Both implementations
receive identical cached cameras, filtered detections, priors, neighboring
cameras and retained banks. These are **geometry-call timings**, not cold-start
or full video processing timings. Detector inference, PnP seed generation,
orientation matching, sequence depth recovery and move search are outside this
port and benchmark boundary. Agreement with the CPU fitter is not a separate
ground-truth accuracy measurement.

The tracked [benchmark summary](benchmarks/cpu-port.json) records versions,
hashes, ranges and numerical comparisons. Full generated reports remain in
`results/interleaved.json` and `results/browser.json` (ignored by Git).

## Sampled numerical validation

All 20 complete candidate banks pass in native Rust, Node WASM and browser
WASM: bank length/order, axis, proposal source, nearest sticker assignments,
validity masks, assigned landmark classes and integer support metrics agree.
Floating values use explicit tolerances: 1e-4 per pose/angle or projected-point
component and 1e-6 for normalized scores/metrics.

Browser observations:

- 20,580 coarse proposals checked across 20 camera sweeps; no row-count
  differences, maximum normalized score difference 6.03e-7.
- 485 solves on the **identical CPU seeds, priors and free-variable masks**:
  identical evaluation counts and termination statuses; maximum parameter
  difference 9.06e-12 and maximum cost difference 4.89e-12. These forced-seed
  checks are separate from normal search and are never used to seed its fits.
- 45 projection/Jacobian probes cover all axes, zero camera rotation and
  nonzero radial/tangential/thin-prism distortion. Maximum residual/Jacobian
  differences are 6.38e-6 / 1.52e-5. Float32 enclosing-circle diameter differs
  by at most 6.11e-5 pixels on these probes.
- Maximum final bank projected-point difference: 3.36e-5 pixels. Maximum
  pose/angle component difference: 1.20e-5; maximum score difference: 1.29e-7.

Search outputs are not bit-identical. Six of the 485 normal browser refinement
starts choose a different tied coarse representative: five involve the signed
45° endpoints (score gaps below 1e-15), and one a symmetric middle-frame camera
(gap 2.18e-8). The full details are recorded in the summary. The final retained
banks still agree under the checks above. Whole-sequence move/orientation
equivalence remains to be evaluated when those stages are ported.

## Reproduce

Run from the repository root with the existing Python environment and a Rust
toolchain supporting `wasm32-unknown-unknown`:

```sh
rustup target add wasm32-unknown-unknown
cargo build --release --locked --manifest-path pose-fitter/Cargo.toml
cargo build --release --locked --target wasm32-unknown-unknown --lib --manifest-path pose-fitter/Cargo.toml
cargo test --release --locked --manifest-path pose-fitter/Cargo.toml

.venv/bin/python scripts/export_cpu_pose_fixtures.py
.venv/bin/python scripts/benchmark_cpu_pose_port.py --repeats 5
.venv/bin/python scripts/serve_pose_port_benchmark.py
```

Open `http://127.0.0.1:4177/bench/` and click **Run benchmark**. The server saves
the browser report locally. Then run:

```sh
.venv/bin/python scripts/summarize_cpu_pose_port.py
```

The default exporter requires the saved logs referenced by
`scripts/analyze_orientation_motion.py::REFERENCE_SOLUTIONS`. The checked-in
`tests/fixtures/cpu-golden.json` supports `cargo test` without those logs; it
includes initial, temporal and static cases plus all 45 projection probes.
To explicitly regenerate that golden fixture from the reference CPU code:

```sh
.venv/bin/python scripts/export_cpu_pose_fixtures.py \
  --golden-output pose-fitter/tests/fixtures/cpu-golden.json
```

For an isolated Node WASM run:

```sh
node pose-fitter/tools/benchmark-wasm.mjs \
  pose-fitter/fixtures/cpu-reference.json pose-fitter/results/wasm.json 5
```

The toolchain used for this initial experiment was installed under
`/private/tmp/live-recognition-rust`; a normal user Rust installation also works.
To use that temporary installation in a shell before the commands above:

```sh
export RUSTUP_HOME=/private/tmp/live-recognition-rust/rustup
export CARGO_HOME=/private/tmp/live-recognition-rust/cargo
export PATH="$CARGO_HOME/bin:$PATH"
```

## Browser preview

`src/browser.rs` adds live-frame input and camera initialization from
class-compatible four-landmark sticker quads. It refines each planar camera
against its fixed quad correspondences, ranks hypotheses against the current
landmarks/sticker centers, and runs the same initial `fit_frame` geometry stage.
Subsequent frames use the previous fitted camera as their seed. WASM exports
`fit_browser_frame` alongside the offline fixture benchmark ABI. The returned
bank carries `flat_fallback`; the returned wireframe belongs to its best fit.

This adapter uses the scripts' centered pinhole camera and 55 mm cube. It has
not ported all native P3P/SQPnP startup branches, nor connected the offline
neighbor refit, depth recovery, color orientation or move search to the browser.
The preview is an inspection option; the historical fixed-camera parity results
do not validate the new startup initializer or a full tracking rollout.

### Saved-detection browser rollout audit

The browser preview does **not** preserve the saved Python geometry's sticker
assignments. A clip-start rollout on all five reference solves (source-time
sampling at a 30 fps cap, with the available pre-solve prefix retained) found:

| Solve | Evaluated frames | Browser winner differs | Isolated neighbor pass differs |
| --- | ---: | ---: | ---: |
| Matty | 99 | 21 | 6 |
| Nahm | 137 | 32 | 17 |
| Xuanyi | 59 | 25 | 18 |
| Tommy | 196 | 37 | 19 |
| Yiheng | 118 | 26 | 17 |

These compare each observed sticker's assigned/unassigned status and projected
face ID after closest global cube-symmetry alignment. Raw IDs and pixel-distance
tolerances are not the acceptance criterion. The reference is logged geometry
option zero, not ground truth or the final move-search selection. Saved filtered
detections are replayed; detector inference and ROI reacquisition resets are
outside this audit. Yiheng frame 0 lacks a saved input pose/observations and is
reported as unavailable; no available solve-window replay returned an empty fit.

The reference assignment is absent from the entire current browser bank on
117 of 609 evaluated frames. Initializer/camera rollout differences and the
omitted neighbor pass are both material; the bank-display choice alone cannot
explain the discrepancy. The former fixed-camera numerical benchmark cannot
validate this integration.

`fit_browser_neighbors` now exposes the CPU temporal geometry pass for isolated
evaluation, using immutable initial neighbor cameras and the symmetry-aligned
mean rotation/translation prior from `process_pose_geometry`. Its saved Tommy
regression matches Python assignments and native/Node WASM projections. It is
**not wired into the app**: it leaves 77 of 609 rollout frames differing and is
not sufficient to restore the CPU behavior. The CPU initialization/trajectory,
orientation and recovery stages still need a faithful port and rollout checks.

Run the full audit and render the worst example per solve:

```sh
.venv/bin/python scripts/audit_browser_pose_rollout.py --render
# Reuse the saved WASM outputs when regenerating the report/overlays:
.venv/bin/python scripts/audit_browser_pose_rollout.py --reuse --render
```

The compact [audit summary](benchmarks/browser-rollout-parity.json) records the
counts and hashes. Full per-frame results and `examples.png` are generated in
`results/browser-rollout-audit/`. This is a behavioral regression, not an accepted
numerical tolerance difference; the app remains an incomplete CPU integration.

## Buffering and later integration

`bench/worker.mjs` keeps one WASM instance alive and drains every queued saved
workload in FIFO order. It neither drops work nor applies a latency deadline.
Fixture parsing and output transfer are outside the timed numerical function.
This is a benchmark worker, not the production detector-worker protocol.

At approximately 47 ms for a temporal call, one worker cannot sustain 30 such
calls per second indefinitely. A buffer permits lookahead and absorbs bursts;
it does not raise sustained throughput. Initial and temporal phases may both
be required for a frame, so their costs must be budgeted together. The next
integration design should schedule buffered initial fits and neighbor-dependent
temporal fits across a worker pool, then release results in frame order. Keep
the existing search/solver policy while measuring the complete pipeline.

## Full-window comparison

Reproduce the broader check and its ranked video overlays with:

```sh
.venv/bin/python scripts/validate_cpu_pose_port_sequences.py
MPLCONFIGDIR=/private/tmp/live-recognition-mpl .venv/bin/python scripts/visualize_cpu_pose_port_differences.py
```

This checks 708 eligible saved frames across the three solve windows, with an
initial and a temporal fit per frame (1,416 calls). Yiheng frame 0 is excluded
because it lacks the required saved pose/observations. Both ports receive the
CPU's exact inputs and retained banks, isolating the fitter boundary. This is
not a rollout driven by the port's own previous outputs or a move-search test.
The full-window WASM comparisons run the same module under Node/V8; the initial
20-workload browser check produced exactly the same banks and refinement
outputs as Node.

The ranked distance is the Euclidean distance between CPU/WASM projections
assigned to the same observed landmark, in original video pixels. It is not a
ground-truth pose error. Raw angle differences are not used for ranking because
quarter-turn aliases can describe the same surface. Both video overlays use
the existing CPU renderer with the returned poses and angles, without refitting.

| Solve | Fit calls | Ordered banks differing | Rank-0 fits differing | Largest rank-0 point distance |
| --- | ---: | ---: | ---: | ---: |
| Matty | 394 | 36 | 4 | 6.4003 px (frame 575, initial) |
| Yiheng | 236 | 17 | 0 | 0.0000326 px (frame 111, temporal) |
| Tommy | 786 | 47 | 10 | 3.0682 px (frame 307, temporal) |

Eight fit calls have a rank-0 point distance above 0.1 px; six exceed 1 px.
The broader test therefore does **not** justify claiming behavior equivalence.
On three discrepant Matty frames (380, 484, 575), 72 identical-seed CPU/native
solves still match to 6.04e-12 with identical evaluation counts and stopping
statuses; their normal searches choose different coarse seeds. That diagnoses
those examples but does not establish the cause of every full-window difference.

## Skipping temporal refits

`scripts/analyze_pose_refit_gate.py` compares each saved initial rank-0 fit
with its CPU temporal result, using sticker-to-face assignments and
assigned/unassigned status as the acceptance criterion. A closest cube-symmetry
mapping puts the two assignment vectors in a common coordinate frame; raw ID
counts are also saved. Placeholder face indices for two unassigned detections
are ignored. This tests the cached assignments used by subsequent search.

The three-pass Node WASM replay now saves every call's times in
`results/full-wasm-timings.json`. Total fitting costs (sums of per-call medians)
are 16.39 s for Matty, 9.15 s for Yiheng and 28.88 s for Tommy. These are serial
geometry costs for the saved solve windows, with loading/JSON outside timing.

| Modeled skip rule | Temporal calls skipped / 708 | Total fitting time saved | Skipped frames with changed sticker assignments |
| --- | ---: | ---: | ---: |
| Initial mean landmark error ≤ 1% of projected cube diameter | 277 | 27.0% | 3 |
| Same error cutoff, plus top two initial fits agree on assignments | 232 | 22.6% | 0 |
| All retained initial fits agree on assignments | 246 | 23.8% | 0 |

The unanimous-bank policy models a reduction from 54.41 s to approximately
41.48 s across all three replays. It retains the complete initial bank when
skipping. No gate has been enabled in the fitter or app. These are counterfactual
results on the same three runs, not guarantees on unseen data or a downstream
orientation/move-search rollout. After common-frame alignment, zero changes
were observed in the unanimous-bank group; raw indices differ on two frames
due to equivalent cube-coordinate relabelings.

Details and per-frame data are in [refit-gate.json](benchmarks/refit-gate.json)
and `benchmarks/refit-gate.csv`. Reproduce with:

```sh
.venv/bin/python scripts/analyze_pose_refit_gate.py
# Recompute assignment policies from the saved timing replay:
.venv/bin/python scripts/analyze_pose_refit_gate.py --reuse-timings
```

## Cube rotation and layer motion

The production search enables `resolve_orientation_ambiguity` and
`max_rotation_tps: 20`, alongside the existing 54/54 layer-turn limit. One TPS
unit is 90 degrees per second. The rotation limit comes from the selected
Python reference paths: Matty's maximum is 14.34, Nahm's is 13.69, and the
largest across all five references is 15.63. Twenty gives at least 28% margin.
Calibration details are in `benchmarks/cube-rotation-budget.json`; regenerate
the interval measurements with:

```sh
.venv/bin/python scripts/calibrate_cube_rotation_budget.py \
  --output pose-fitter/results/search-failure-audit/rotation-budget-reference.json
```

`move_search/motion.rs` composes the fitted cameras, residual slab angles and
committed turns before measuring material-cubie motion. The least-moving
cubie bounds common rotation; the fastest bounds combined motion. Both caps
must pass, in addition to the original relative layer-turn budget. A rotation
therefore reduces available turn speed when their physical motions reinforce.
Same-axis cancellation is applied first: `R x'` has the same cost as `Lw`.
A middle-slab fit's rotation absorbed into its camera is composed with its
outer angles, so an M turn is not automatically charged as whole-cube rotation.
These are endpoint constraints; they do not infer hidden full revolutions.

The six numerical fit banks stay unchanged. Search derives immutable cube
symmetry permutations, keeps the original orientation, and checks its nearest
camera continuation after a large change. The initial frame can also use its
best full-sticker color orientation because it has no motion reference. Normal
search scoring uses all classified sticker observations and the current cube
state, including moved centers after slices. Geometry aliases share a physical
search key. Mechanical routes and their material rotations are prepared once
per shape pair and reused; no PnP, projection or numerical refit is added.
Rotation/motion gates are omitted when either endpoint has a finalized bad-pose
flag. Flagged frames retain the existing unrestricted legal-move expansion.

The browser and independent runner share `runtime/reconstruction-policy.mjs`,
`runtime/buffered-pipeline.mjs` and `runtime/incremental-search.mjs`. Source-frame
numbering, opening holds, admission capacity, worker limits, search chunking and
configuration defaults are defined once.

Search scoring caches first-permutation row deduplication during preparation,
scores observed colors directly through each permutation, and stably sorts
compact candidate descriptors. Full successor states are created for sibling
deduplication; step telemetry is created only after branch admission. Raw probe
rows, first-row selection, tied ranks, queue priorities and search counters are
preserved.

**Record search diagnostics** controls the next browser run (`diagnostics` in
the search configuration, default `true`). When disabled, search stores only
essential path data: frame, geometry/orientation, moves and pose flag. It skips
optional step telemetry, the worker action journal, timing telemetry, runtime
fingerprint collection and automatic development-run diagnostic exports. Search
costs, history windows, pruning and progress counters remain active and unchanged.
Reconstruction and cached review still work. Explicit `probe` requests continue
to return full diagnostics. Runs without journals cannot be exact action-order
replays; use diagnostics when collecting a regression capture. This setting
does not disable detector/fitter output or the in-memory detection recording.

The fixed-action WASM comparison in
[`benchmarks/move-search-optimization.json`](benchmarks/move-search-optimization.json)
uses three repetitions, 128-expansion requests, saved fitted banks and a
10,000-expansion cap. With diagnostics enabled, median batch search times changed
from 779 to 597 ms (Matty), 4,236 to 2,638 ms (Kyle trim90), and 5,096 to 2,585 ms
(Yiheng277). Every response, including probes and counters, matches the earlier
build. Disabling diagnostics preserves every essential response and further
reduces median streamed Matty search time from 691 to 586 ms. These are search
ABI timings, not fresh-fitting or end-to-end browser reconstruction timings.
Saved build/input artifacts are under `results/move-search-optimization/`.
To compare another pair of builds with the same input and action order:

```sh
node pose-fitter/tools/compare-move-search.mjs \
  BEFORE.wasm AFTER.wasm REPORT.json INPUT.json \
  --repeats=3 --states=10000 --mode=stream
```

Development runs automatically save raw detections, exact run settings, file and
loaded ONNX hashes, runtime fingerprints and the search action journal under
`results/browser-runs/`. `latest.json` points to the newest run. Captures are scoped
to one reconstruction, including explicit links when the app reuses saved raw
frames. No automatic upload exists in a production build.

The default replay consumes that recorded configuration unchanged, recomputes
every fit and replays the recorded frame-arrival/expansion order without video
playback or wall-clock waits:

```sh
web/node_modules/.bin/tsx pose-fitter/tools/replay-browser-flow.mts \
  RECORDING.json OUTPUT.json
```

It explicitly discards recorded fits, cameras and startup seeds. Recorded banks
are used only for an output comparison. A resolution, source clock, settings,
runtime or shared-driver mismatch fails before fitting. Legacy captures and any
configuration overrides require `--diagnostic`; those outputs are explicitly
marked ablations, not exact browser reproductions. `--action-journal FLOW.json`
can verify a standalone driver journal with another completely fresh fit pass.

Reference trajectories are used only to calibrate/diagnose, never as replay
inputs. Earlier timing examples in `benchmarks/cube-rotation-search.json` describe
the captured in-app-browser runs only. They do not establish success for a
different browser/configuration. In particular, the old Tommy3 raw capture was
160px and is not a valid regression comparison with the current 256px browser.

For pose-only investigations, `replay-browser-flow.mts RECORD OUTPUT
--diagnostic --fit-only` recomputes every bank from raw observations without
running search, and reports differences from captured banks. These fresh results
include `search_frames` for the native `move-search-benchmark` tool. Its `journal`
mode accepts recorded `actions`, replays them against the fresh frames, then
appends any messages still queued at the EOF snapshot before closing input and
continuing search. An incomplete journal continuation is diagnostic rather than
an exact completed-browser reproduction. The WASM replay tool also supports
`--continue-after-journal` for this case.

`scripts/render_browser_run_review.py` renders fresh-fit review sheets from a
JSON list of `{fits, video, frames}` specifications. It verifies source hashes
and associates each captured timestamp with actual video presentation timestamps,
then decodes sequentially. A browser `sourceFrame` is a time-derived clock and
must not be used as a decoder index for variable-rate video. The JSON beside each
image records the selected decoder indices and timestamp differences.

The earlier reviewed Jeff annotations remain in
`results/search-failure-audit/jeff-review/user-labels.json`. They refer to
`jeffSolve.mov`, independently of the newer `jeff1551.MOV` browser capture. The
fresh current-runtime check in
`results/search-failure-audit/current-browser/jeff-earlier-reviewed-current-check.json`
preserves the reviewed primary sticker assignments at all twelve labeled frames.
All nine labeled bad poses remain flagged. The newer contact sheet's original
Jeff overlays had an incorrect timestamp association; the corrected review
accepted frames 468/469 and candidate 2 at 471 independently of the older file.

## Pose support flags

Two additive checks use fitting results already available. Sparse observations
(at most ten landmarks and five stickers) are flagged when mean landmark
reprojection error exceeds 0.025 of the projected cube diameter. A second check
flags combined mean error above 0.03 when at most 70% of observed landmarks have
distinct assigned 3D landmarks and at most 70% of stickers have valid assignments.
Neither check uses detection scores or keypoint-presence scores, and neither
adds inference, PnP, projections, fitting passes or future buffering.

These checks flag the reviewed Jeff1551 frame 379 and Azdaba1304 frame 133, plus
the previously reviewed bad pose at frame 410 of the earlier Jeff video. Fresh
paired fitting of twelve datasets leaves every geometry bank unchanged. Three
paired raw-browser controls also preserve solved outcomes and state counts.
Finalized geometry flags exclude those poses from the startup-core reference and are sent
to search as authoritative per-frame flags; subsequent unflagged frames resume
normal search queuing. Legacy inputs without an explicit flag retain metric-based
derivation. Yiheng305 and Xuanyi308 are excluded from further acceptance checks.

See [`benchmarks/pose-support-rollout.json`](benchmarks/pose-support-rollout.json)
and `scripts/validate_pose_support_rollout.py`. Jeff1551 and Azdaba1304 wireframe
videos in `results/recovery-weight/videos/` include the new flags. Other renders
retain their recorded fitter version, shown in each video's caption and JSON.

An additional, separate warning identifies insufficient assigned sticker
evidence: every usable five-plus-flat option has at most three distinct valid
projected sticker IDs. This follows the Python search's assigned-count meaning;
it is not a raw detection-count or color-accuracy test. One usable option with
more evidence prevents the frame-wide warning. Seven independently assigned
stickers are sufficient. Empty geometry banks remain unavailable.

`pose_flag_reason` covers either geometric unreliability or insufficient color
evidence; `pose_geometry_flag_reason` distinguishes unreliable geometry. Color-
only warnings preserve the fitted camera as a finalized geometry reference and
retain current/previous geometry in motion constraints. Both frame fields are
forwarded to search. Older untyped explicit flags retain their previous bad-
geometry meaning. No pose, bank, correspondence or saved result is erased.

For a particular search state/cue, the existing <=3 assigned-sticker predicate
can also find insufficient evidence when the whole frame's bank is not sparse.
With the flag feature enabled, such a node retains every mechanically legal
context, inherits its incoming queue priority and omits color mismatch history;
the next adequately observed node resumes normal queuing. Static opening holds
are excluded because their synthetic hold row deliberately contains no color
scoring. Reliable geometry still limits moves. All-zero assigned-color rows get
a neutral placeholder only if no mechanically legal candidate has adequate
color evidence; a supported candidate excludes zero-assignment artifacts.
Disabled flag mode retains Python's original branch-only sparse handling and
zero-assignment rejection. Rates, branching thresholds and normal width remain
unchanged.

See [`benchmarks/sticker-evidence-rollout.json`](benchmarks/sticker-evidence-rollout.json)
and `scripts/validate_sticker_evidence_rollout.py`. The four included browser
wireframe videos (Jeff1551, Azdaba1304, Kyle and Yiheng277) hide warned overlays
while preserving source images, timestamp captions and saved geometry. Their
JSON companions record the exact runtime and warning frames.

## Independent search-divergence diagnostics

`tools/lib/move-path-comparison.mjs` compares saved move paths as exact sticker
permutations modulo the 24 whole-cube rotations. It separates temporary
excursions that rejoin the reference from persistent departures. Its word
normalizer cancels adjacent inverse turns after transporting explicit cube
rotations; it never cancels inverses across noncommuting intervening moves.
Run its standalone behavioral checks with:

```sh
node --test pose-fitter/tools/lib/move-path-comparison.test.mjs
```

`tools/compare-saved-move-paths.mjs ORDINARY REFERENCE FLOW OUTPUT.json`
produces JSON and Markdown, including same-parent transition probes. A chosen
partial path differing from the reference does not imply that every reference
branch was pruned. Witness-bank/cue compatibility must be explicitly asserted
with `--compatible-witness-banks` before replaying those parents; a trimmed
restart can change cube-coordinate labels while preserving projected geometry.
Clock frame/FPS is labeled separately from the actual recorded source timestamp.

`tools/diagnose-local-followthrough.mjs FLOW WITNESS OUTPUT FRAME HORIZON`
uses published WASM probes to compare the first 25 distinct local choices and
ordinary admitted successors. The reference only selects the incoming state
and labels results; it never affects successor scores. Exact raw state/cue
keys avoid assumptions about canonical basis aliases. Output records any
context truncation. JSON probe timings are diagnostic overhead, not app speed.

The current investigation is indexed in
[`benchmarks/search-history-investigation.json`](benchmarks/search-history-investigation.json).
Source/expected-color review panels use recorded timestamps and sequential PTS
association, and independently reproduce native color costs before rendering.
These tools do not perform inference or fitting. Fresh fitting from raw
observations remains the responsibility of the replay tools described above.

## Final weak-layer alternatives

`final_bank::alternatives`/`select` and Python `final_assignment_bank` run only
when producing the final output bank. Initial fits, immutable neighbor cameras,
orientation references and finalized camera-continuity anchors retain the
original score-first selection. The final alternatives cannot seed later fits.

Only a nonzero outer angle whose existing `layer_support` is at most one is
eligible. This counter measures geometric proximity support, not detector
confidence, keypoint presence or unique assigned observations. A supported
outer layer stays fixed. Each eligible mask zeros one or both weak angles at
the parent's exact camera; it performs one projection/assignment evaluation,
without a sweep, optimizer or PnP solve. There are at most three masks per fit
and usually no attempts at all.

A variant qualifies if balanced reprojection cost is no worse (1e-9 numerical
tolerance), **or** valid sticker assignments are unchanged. Invalid placeholder
IDs do not count as assignments. Qualified duplicates prefer fewer active
layers, then smaller absolute angles. The bank retains at most five distinct
regular assignment patterns and its original reserved whole-flat fallback.
`final_bank_diagnostics` counts work; `continuity_source_pose` documents the
original fit used for continuity separately from the output alternatives.

The rejected broad flatter-first experiment is archived under
`results/search-failure-audit/flatter-dedup-controls`; it altered intermediate
camera seeds. Final-only raw-input regression results and before/after pictures
are under `results/search-failure-audit/final-flat-controls` and
`results/search-failure-audit/final-flat-control-visuals`.

## Recent motion cost

New web runs explicitly record `motion_cost_weight: 0.025`. Legacy/native inputs
that omit it keep zero weight. The 54 TPS turn limit and separate rigid-rotation
limit remain unchanged. This is a queue preference, not an additional gate.

Each candidate path carries the latest eight processed-frame requirements
(including the current step). A requirement is existing route angular travel
divided by that step's existing maximum-TPS travel budget. Its weighted rolling
sum is added to color/history priority. These are endpoint lower bounds on
required motion, not measured physical TPS. At 30 processed FPS the eight motion
intervals cover about 0.27 seconds; a 60 FPS source still supplies alternate
frames. No future-frame wait, projection, fitting or PnP pass is added.

Color-only inheritance on flagged frames is tracked separately from motion
priority, so old motion cost is not repeatedly added. Zero weight retains the
legacy order and omits the new step telemetry. Saved enabled steps expose
`required_tps_ratio` and `recent_motion_penalty` for diagnostics. The pure
`move_search::motion_history` tests cover expiration and normalization; native
and WASM raw-flow replays cover complete reconstruction behavior.

## Geometry-only TPS curve experiment

`scripts/analyze_geometry_tps.py` compares geometry-only estimates with ten saved
successful 256px captures from nine solve families (the two Yiheng crops are one
family). It independently applies each recorded scramble and recovered path to
verify a solved cube. Predictions read the primary geometry's angles/axis and
existing geometry flags, without selected search geometry, colors, inferred
state, inference, projection or fitting. The target is selected-path angular
travel divided by 90 and actual elapsed source time. It is inferred endpoint
travel, not manually labeled physical TPS. Committed-token TPS is also logged;
equivalent wide/slice spellings can change that counter.

The requested strict `>6°` version counts one assumed quarter-turn per processed
frame if either outer-to-middle residual exceeds six degrees from the nearest
quarter-turn alignment. Other variants count layers, anticipated commit events,
changes in raw/wrapped angles, or changes in a three-state angle signal. The
three-state variant maps `[-6°, +6°]` to zero and residuals outside that interval
to signed 45°. It measures changes using the existing shared-layer travel
formula, with exit-to-flat plus entry-from-flat when axes differ. This is a
geometry activity proxy, not a mechanically reconstructed route.

For median within-solve curve correlation, the three-state variant is more
consistent than per-frame counting: 0.85 versus 0.76 at 0.25s, 0.86 versus 0.79
at 0.5s, 0.84 versus 0.78 at 1s, and 0.87 versus 0.72 at 2s. The worst 1s case
improves from -0.30 to 0.69. A synthetic 2x/4x slowdown repeats the real poses at
the original processing cadence: angle-change estimates scale down with speed,
while per-frame occupancy estimates remain unchanged. This stress test is not
an additional real slow-solver recording.

`scripts/analyze_geometry_tps_curves.py` records zero-lag correlation, absolute
error, bias, overestimate frequency and diagnostic lag sweeps. It also tests
nonnegative blends with held-solve-family-out coefficients and margins. These
can improve median error while degrading individual solves; they are not an
accepted production estimator. Simple positive padding preserves the
three-state curve shape: `1.25 * phase_tps + 2` exceeds the angular reference in
99.8% of the observed full 1s windows, averaged equally across solve families,
but still undershoots some shorter windows. This is empirical coverage on
overlapping windows, not a guaranteed safe hard cap.

All source hashes, compact input geometries, whole-solve results, trailing
0.25/0.5/1/2-second and eight-processed-frame curves, causal prefixes, CSV logs
and plots are under `results/geometry-tps/`. Full-window summaries exclude
warmup; partial interval boundaries use actual elapsed time. The deployed TPS
constraints are unchanged. Reproduce the saved-input analysis with:

```sh
python3 scripts/analyze_geometry_tps.py \
  pose-fitter/results/geometry-tps/manifest.json \
  pose-fitter/results/geometry-tps/report.json
MPLCONFIGDIR=/private/tmp/geometry-tps-matplotlib \
  .venv/bin/python scripts/analyze_geometry_tps_curves.py \
  pose-fitter/results/geometry-tps/report.json
.venv/bin/python -m pytest tests/test_geometry_tps_estimates.py -q
```

`scripts/evaluate_geometry_tps_limits.py` evaluates those estimates as
counterfactual constraints on saved paths. With `min(54, 1.25*phase_tps+2)`, the
0.25s/eight-frame windows cut nine of ten saved paths, 0.5s cuts six, 1s cuts one
(a single 0.081-TPS shortfall), and 2s cuts none. These are saved-path checks,
not full search reruns or evidence that another valid path would be absent.
A margin fitted to every other solve family still misses held-out paths for
windows up to 1s. Full 2s windows are available over only 56% of these short
solves on average, with the original ceiling retained during warmup.

Applying any of these rolling averages directly as a per-transition cap cuts
transitions in every saved solve. A sustained-speed constraint must instead
compare accumulated angular travel with a budget for the same time window,
while retaining the existing per-transition ceiling for bursts. Hard rolling
constraints also require sufficient motion history in state merging: the
current dominance key omits that history. No production constraint or queue
policy is changed by this evaluation. Results and per-window decisions are in
`results/geometry-tps/window-limits.json` and `.csv`. Reproduce with:

```sh
python3 scripts/evaluate_geometry_tps_limits.py \
  pose-fitter/results/geometry-tps/manifest.json \
  pose-fitter/results/geometry-tps/curves.csv \
  pose-fitter/results/geometry-tps/window-limits.json
```

The follow-up search experiment is indexed by
`results/geometry-tps/search-experiment/wasm-summary.json`. It uses the cached
successful cropped-Yiheng and Kyle banks and recorded action journals. The
cached `c9c124df0f46095e` runtime reproduces 55,797 and 57,642 expansions exactly.
Current source has since changed route-travel accounting from the sum of
absolute outer-to-middle changes to shared physical travel. To avoid mixing
that change into the gate experiment, a temporary WASM build restores only the
cached travel convention; its gate-off stats and every step match the cached
successes exactly. The workstation source and production runtime are not
changed by that compatibility adjustment. Earlier native/current-source pilot
failures are not classified as failures of the cached successful runs.

On those matched baselines, a 1s rolling gate reduces Kyle to 13,051 expansions
and cropped Yiheng to 51,199. A 0.5s gate with two additional quarter-turn travel
units reduces them to 43,199 and 52,306. A 0.25s gate with one additional unit
reduces Kyle to 5,700 but fails to solve cropped Yiheng within 100,000 states;
three additional units retain both solves but save only 7% and 1% of expansions.
Every completed gated variant retains the exact cached steps, including geometry
and scores, and independently passes both final cube and rolling-budget checks.
The travel allowance is added to the window budget, not to each interval's
54-TPS ceiling. Larger allowance improves safety while weakening pruning.

These measurements are isolated diagnostics with the existing dominance
heuristic, not an accepted production policy. Single-pass search times include
WASM JSON responses: Kyle's 1s variant improves from 7.7s to 3.5s; cropped
Yiheng's roughly 8% expansion reduction does not produce a speedup. Limits and
history tracking have overhead, and current-source physical-travel results must
be assessed separately from the cached convention. Source hashes, compatibility
build helper, input preparations, raw outputs and replay scripts are retained
beside the summary.

A broader matched-cache replay is in
`results/geometry-tps/search-experiment/one-second-cohort-summary.json` and `.csv`.
All ten gate-off runs reproduce their cached stats and steps exactly. All ten
one-second gated runs reach independently verified solved cubes and pass 1,434
independent winning-path budget checks. Eight retain every cached step. Tommy2
differs at source frame 902 and rejoins at 904; Teodor finishes at frame 162
instead of 164 with different final move timing. Those differences need video
review before treating the gate as reconstruction-preserving.

Expanded states, baseline to one-second gate:

| Capture | Baseline | One-second gate | Change |
| --- | ---: | ---: | ---: |
| Matty | 1,280 | 1,280 | 0% |
| Nahm | 3,032 | 2,333 | -23.1% |
| Tommy2 | 5,175 | 4,997 | -3.4% |
| Tommy short | 2,313 | 2,303 | -0.4% |
| Tommy3 | 4,314 | 4,301 | -0.3% |
| Charlie | 2,773 | 2,773 | 0% |
| Teodor | 627 | 1,694 | +170.2% |
| Kyle | 57,642 | 13,051 | -77.4% |
| Yiheng cropped | 55,797 | 51,199 | -8.2% |
| Yiheng uncropped | 96,931 | 94,653 | -2.4% |

This is ten captures of nine solve families, since the two Yiheng recordings
are related. Nahm and Tommy2 use newer successful captures with action journals.
Single-pass times improve substantially for Kyle (7.57s to 3.57s) and Nahm
(3.32s to 2.63s), but worsen for uncropped Yiheng (13.62s to 15.96s) and Teodor
(1.27s to 1.55s). Small timing differences are inconclusive; fewer expansions
do not always offset history-check overhead.

`results/geometry-tps/search-experiment/combined-summary.json` tests requiring
both the one-second gate and a half-second gate with two additional quarter-turn
travel units. Compared with one second alone, Kyle expands 14,091 rather than
13,051 states, and cropped Yiheng 50,461 rather than 51,199. Both retain the exact
cached reconstruction and pass independent checks for both window budgets.
There is no clear overall benefit from combining these gates on these two cases.
All these experiments retain the eight-frame soft motion priority penalty and
the separate eight-entry color penalty, as well as the original hard interval
caps. The eight-frame motion calculation is not a hard TPS check to replace.
Production remains unchanged, and the cached-convention/dominance limitations
above still apply.

The separate `results/geometry-tps/search-experiment/yiheng277-fresh/summary.json`
uses 97 new 256px WebGPU detections and new current-runtime fits, captured in the
browser rather than drawn from a historical successful run. Its temporary gate
build retains current shared physical travel; gate-off matches production stats
and essential steps at 100,000 states. Gate-off, 1s and combined 1s/0.5s all hit
one million states without solving. The 1s gate prunes no candidates; combined
prunes 2,620. Parallel single-pass times are not comparable speed benchmarks.

The user-confirmed partial ends at source frame 158. A reference-compatible
diagnostic continuation is L' at 160, F' at 164, R2 at 170, and F at 172/174.
Production probes verify those mechanics and an independent permutation check
verifies its final solved cube. R2 at 170 requires sibling width seven; the
configured width three excludes it. On the same primary fit, hold scores
0.209810808476 and R2 0.232538081203. One detected orange object (object/class
score 0.9941, admitted at threshold 0.90) is assigned to a face that expects
yellow for hold and white after R2. Directional confusion gives those costs
0.5 and 1, respectively, accounting for the 0.022727 score gap. This object/class
score is not a keypoint localization confidence. Both subsequent F transitions
rank first once the R2 state is retained. The reference is used only for this
diagnostic; ordinary search never receives it.

Existing standalone Python bidirectional search on these same fresh banks finds
a solved-cube candidate in 3,556 states with width three (no supplied reference).
It is not policy-equivalent to current Rust. A later reference-constrained
current-Rust audit finds a mechanically legal realization of its timed physical
states, including the solved endpoint, with up to 24 retained contexts. That
audit is not an ordinary Rust bidirectional run or a complete manual video
accuracy review. Original pixels, fit/color overlays, scores and scripts are
saved beside the summary. `geometry-curves.csv` logs geometry estimates over
0.25s/0.5s/1s/2s/eight-frame windows; `diagnostic-route-curves.csv` also logs the
reference-assisted route's minimum travel rates, explicitly not an unguided
successful reconstruction or ground-truth physical TPS.
