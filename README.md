# Magic Stew

Rubik's Cube recognition and solve reconstruction in the browser. The active
application uses Svelte, TypeScript, ONNX Runtime Web, and a Rust/WASM pose fitter.

## Repository layout

- [`web/`](web/README.md): browser application, workers, and frontend tests.
- [`pose-fitter/`](pose-fitter/README.md): Rust/WASM pose fitting and move search.
- [`model-support/`](model-support/README.md): nine-class YOLO pose training,
  the matching Blender dataset generator, and synthetic scene assets.

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

## Checks

```sh
npm --prefix web run check
npm --prefix web test
cargo test --manifest-path pose-fitter/Cargo.toml
```

Training data, checkpoints, local environments, and generated runtime files are
ignored by Git.
