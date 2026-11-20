# Magic Stew

Rubik's Cube recognition and solve reconstruction in the browser. The active
application uses Svelte, TypeScript, ONNX Runtime Web, and a Rust/WASM pose fitter.

## Repository layout

- [`web/`](web/README.md): browser application, workers, and frontend tests.
- [`pose-fitter/`](pose-fitter/README.md): Rust/WASM pose fitting and move search.
- [`model-support/`](model-support/README.md): nine-class YOLO pose training,
  the matching Blender dataset generator, and synthetic scene assets.
- `docs/`: generated GitHub Pages site, including its model and WASM files.

**Everything in [`experiment-records/`](experiment-records/README.md) is unused
historical material.** Its Python pipelines, alternative trainers, notebooks,
F-Clip experiments, tests, and old documentation are not part of the current
application, model-support workflow, build, or checks.

## Run the web application

From the repository root:

```sh
cd web
npm ci
npm run dev
```

See [`web/README.md`](web/README.md) for required detector models, WASM artifacts,
and browser requirements.

## Model Weights
https://huggingface.co/joongpa/you-only-cube-once

## Checks

```sh
npm --prefix web run check
npm --prefix web test
cargo test --manifest-path pose-fitter/Cargo.toml
```

Training data, checkpoints, local environments, and local runtime inputs are
ignored by Git. The generated `docs/` deployment is committed.
