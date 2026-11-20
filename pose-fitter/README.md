# Cube pose fitter

Rust/WASM pose fitting and cube move search used by the browser application.
This directory is independent of the archived Python implementation.
Everything in `../experiment-records/` is unused by the current runtime, build,
and checks.

## Files

- `src/`: camera geometry, fitting, articulated cube poses, and move search.
- `runtime/`: shared WASM host, buffered pipeline, and search scheduling.
- `tools/`: Node build, replay, validation, and benchmark utilities; the OpenCV C++ bridge.
- `tests/fixtures/`: checked-in reference inputs and expected outputs.
- `tables/`: cube geometry, orientations, and move permutations.
- `vendor/opencv/`: camera solver source and third-party notices.
- `bench/`: standalone browser benchmark harness.

## Native checks

From the repository root, with a current Rust toolchain:

```sh
cargo test --locked --manifest-path pose-fitter/Cargo.toml
cargo test --locked --features cpu-pipeline --manifest-path pose-fitter/Cargo.toml
```

These checks use the committed fixtures and do not require the experimental
Python package, training data, or saved runs.

## Build for the browser

Install the Rust `wasm32-unknown-unknown` target, CMake, Ninja, and Emscripten
4.0.20. Obtain the official OpenCV 5.0.0 source tree. Point `EMSDK` to the
Emscripten SDK directory and `OPENCV_SOURCE` to the OpenCV source directory.
Emscripten uses its own SDK-provided Python; it does not use the archived project.

From the repository root:

```sh
rustup target add wasm32-unknown-unknown
node pose-fitter/tools/build-opencv-wasm.mjs --source "$OPENCV_SOURCE"
cargo build --release --locked --manifest-path pose-fitter/Cargo.toml \
  --features cpu-pipeline --target wasm32-unknown-unknown --lib \
  --target-dir pose-fitter/target/parity-wasm
cargo build --release --locked --manifest-path pose-fitter/Cargo.toml \
  --target wasm32-unknown-unknown --lib
npm --prefix web run sync:runtime
```

The OpenCV wrapper accepts `--emsdk`, `--build-dir`, and `--jobs` overrides.
It writes `target/opencv-pnp.mjs` and `target/opencv-pnp.wasm`; the complete Rust
pipeline is built under `target/parity-wasm/`. The frontend sync step copies and
fingerprints the artifacts. Build output is ignored by Git.

The camera kernels use the pinned OpenCV algorithms and compiler flags from the
existing fitter. Source provenance and notices are in
[`THIRD_PARTY.md`](THIRD_PARTY.md) and [`vendor/opencv/README.md`](vendor/opencv/README.md).

## Check the camera kernels

After building the browser artifacts:

```sh
mkdir -p pose-fitter/results
node pose-fitter/tools/check-startup.mjs \
  pose-fitter/tests/fixtures/startup-golden.json pose-fitter/results/startup-check.json
```

Additional replay utilities accept explicit JSON inputs and report destinations:

- `tools/run-pipeline.mjs`: complete pipeline or individual fitting stages.
- `tools/run-move-search.mjs`: batch or streaming move search.
- `tools/check-buffered-pipeline.mjs`: buffered-versus-batch fitting checks.
- `tools/check-browser-recording.mts`: replay a browser detection recording.

For example, from the repository root:

```sh
web/node_modules/.bin/tsx pose-fitter/tools/check-browser-recording.mts \
  cube-detections.json pose-fitter/results/browser-recording-check.json
```

The browser application and its checks are documented in
[`../web/README.md`](../web/README.md). Earlier performance investigations,
Python parity procedures, and benchmark narratives are preserved in
`../experiment-records/pose-fitter/README.md` as unused historical records.
