# Magic Stew web detector

Svelte, TypeScript, and Vite application for cube recognition and solve
reconstruction. ONNX inference runs locally in WebGPU workers, with pose fitting
and move search provided by the Rust/WASM code in `../pose-fitter/`.

Everything in `../experiment-records/` is unused historical material. The app,
its build, and its tests do not use that directory or its Python package.

## Run

From the repository root, with Node 22.3 or newer:

```sh
cd web
npm ci
npm run dev
```

Open the printed localhost URL in a browser with WebGPU support. Image and video
uploads use the detector at `/`; camera solves use the cube timer at `/timer`.
Camera access requires browser permission and HTTPS or localhost. Hold space
for 0.3 seconds and release to start a solve; any newly pressed key stops it.

## Required model files

Model weights are not committed. Place the ONNX files described by
`public/models/detector.json` under `public/models/` before running inference.
The default application uses these two models:

| Stage | File | Contract |
| --- | --- | --- |
| Cube bounding box | `cube-bbox-640.onnx` | One cube class; dynamic rectangular input; output `[1, 5, N]` |
| Pose | `cube-pose-256.onnx` | Fixed 256 × 256 input; nine classes; one keypoint per object; output `[1, 16, 1344]` |

The manifest also lists optional 160px and 288px pose models for developer
comparisons. Use FP32, batch-one models with embedded NMS disabled. The retained
pose training and Blender generator live in [`../model-support/`](../model-support/README.md).

Pose class scores use thresholds of 0.25/0.40/0.25 for the three landmarks and
0.90 for the six sticker colors. Keypoint-presence scores are separate channels;
they are not per-point localization confidence. The cube bounding-box detector
also has its own score and threshold. `detector.json` specifies the full tensor
layout, thresholds, tracking settings, and model provenance.

## Required WASM artifacts

Build the Rust and OpenCV camera kernels using
[`../pose-fitter/README.md`](../pose-fitter/README.md).
`npm run sync:runtime` copies the generated artifacts and ONNX Runtime bootstrap
into `public/pose/` and `public/runtime/`. It runs automatically before `dev`
and `build`. These generated directories are ignored by Git.

## Validate and build

From `web/`:

```sh
npm run check
npm test
npm run build
npm run preview
```

Tests cover detection and preprocessing, frame/camera timing, ROI tracking,
pose fitting, move search, reconstruction, recording, and timer replay. The
runtime and search tests use the checked-in fixtures in `src/lib/fixtures/` and
`../pose-fitter/tests/fixtures/`.

Serve `dist/` on static HTTPS hosting, including the generated model and runtime
files. Serve WASM as `application/wasm`. Cross-origin isolation headers are
configured for Vite development and preview:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## Developer tools

- `scripts/local-diagnostics.ts`: local diagnostic capture storage for Vite.
- `scripts/profile-live-pipeline.mts`: profile a saved browser recording.
- `scripts/benchmark-search-memory.mjs`: compare WASM search memory behavior.
- `tests/`: browser-only checks for capture, frame presentation, and GPU processing.
- `tools/check-cube-motion.mts`: cube motion checks.

The former detailed experimental guide is preserved in
`../experiment-records/web/README.md` as an unused record.
