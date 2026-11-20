# Model support

This folder retains the nine-class YOLO pose workflow used by the browser
detector. Each detected object has one keypoint (`kpt_shape: [1, 3]`). The classes
are `corner`, `corner-edge`, `corner-edge-center`, `white`, `yellow`, `orange`,
`red`, `blue`, and `green`, in that order.

- `apply_blender_cube_moves.py`: Blender synthetic dataset generator.
- `train_yolo11n_keypoints_local.py`: YOLO11/YOLO26 pose training and dataset preparation.
- `synthetic-assets/`: scene libraries, hand assets, textures, and printed graphics.

Both scripts were restored together from commit
`4ffae0547430f4cb08cd4fbd5e3bf43ddc6457b8`, immediately before the later sticker
schema experiments. Paths now resolve relative to this folder. The current
21-class versions and the segmentation/alternative trainers are preserved in
[`../experiment-records/scripts/`](../experiment-records/scripts/).

## Environment

From the repository root:

```sh
python3 -m venv model-support/.venv
model-support/.venv/bin/python -m pip install -r model-support/requirements.txt
```

On Windows, use `model-support/.venv/Scripts/python.exe`. The generator searches
this environment first and falls back to an existing repository-root `.venv`.
Blender's embedded Python must be compatible with the installed NumPy/OpenCV
packages. Install `ffmpeg` on `PATH` for video-compression augmentation.

## Generate a dataset

With Blender available as `blender` on `PATH`, run from the repository root:

```sh
blender model-support/synthetic-assets/cube.blend --background \
  --python model-support/apply_blender_cube_moves.py -- \
  --count 100 --gpu
```

Output defaults to `model-support/data/synthetic_cube_dataset/`, with `images/`,
`pose/`, and a class manifest. Use `--output-dir` to change the destination.

## Train or prepare only

```sh
model-support/.venv/bin/python model-support/train_yolo11n_keypoints_local.py \
  --model-family yolo26 --model-size medium --device 0 --batch 12 --epochs 100
```

The trainer reads the generator's default output folder and writes prepared
datasets and training results under `model-support/runs/yolo11n_keypoints_local/`.
Use `--source-dir`, `--workdir`, or `--prepare-only` to override those choices.
It rejects other class schemas, including the archived 18/21-class experiments.
Both scripts support `--help` (after Blender's `--` for the generator).

The browser's required ONNX artifacts, including the separate cube bounding-box
detector, are documented in [`../web/README.md`](../web/README.md).
Nothing in `experiment-records/` is used by this workflow.
