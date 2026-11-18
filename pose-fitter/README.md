# CPU pose fitter: Rust and WASM

## Incremental move search

`move_search::Engine` is a forward-only CPU port of the scorer/transition and
best-first widening logic used by `search_independent_keypoint_bidirectional.py`.
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

Streaming deliberately retains alternatives and continues eligible work while
waiting for new frames, so its expansion order can differ from a batch run.
Its candidate scores and legality remain the reference rules. Reported CPU/WASM
times exclude detection, fitting, module loading and file I/O; they are local
single-pass diagnostic measurements, not whole-app FPS promises.

## Complete standalone CPU pipeline

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

The app uses up to three compute workers, an ordered twelve-frame presentation
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
