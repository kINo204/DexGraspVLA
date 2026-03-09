# Repository Guidelines

## Project Structure & Module Organization
`train.py` and `train.sh` are the main entrypoints for controller training. Core learning code lives under `controller/`: `model/` for network components, `policy/` for the diffusion policy, `dataset/` for Zarr-backed data loading, `workspace/` for training orchestration, and `config/` for Hydra YAMLs. High-level planning code is in `planner/`, deployment helpers are in `inference_utils/`, and `inference.py` plus `inference.sh` run the hardware-facing inference pipeline. Keep images and paper figures in `assets/`; avoid mixing generated logs or checkpoints into source directories.

## Build, Test, and Development Commands
Install dependencies with `pip install -r requirements.txt` inside a Python 3.9 environment. Run single-GPU training with `python train.py --config-name train_dexgraspvla_controller_workspace`. Run multi-GPU training with `accelerate launch --num_processes=8 train.py --config-name train_dexgraspvla_controller_workspace` or `./train.sh`. Start inference with `python inference.py --save_deployment_data --gen_attn_map` or `./inference.sh`. Update `controller/config/*.yaml` and `inference_utils/config.yaml` instead of hardcoding paths or hardware settings.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and clear, module-scoped imports. Use `snake_case` for functions, variables, and config keys; use `PascalCase` for classes such as `DexGraspVLAPlanner`. Keep new files aligned with existing naming patterns like `train_dexgraspvla_controller_workspace.yaml` and `base_image_policy.py`. This repository does not currently include a dedicated formatter or linter, so keep diffs small and consistent with surrounding code.

## Testing Guidelines
There is no standalone `tests/` suite yet. For code changes, run the narrowest practical validation: controller changes should at least execute a short `python train.py ...` smoke run on sample data, and inference changes should be checked with `python inference.py --help` or a safe dry run on configured hardware. If you add tests, place them in a new `tests/` package and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects such as `Update inference.py` and `Disable tqdm update from non-main process`. Keep commit titles concise, present tense, and scoped to one change. Pull requests should state the affected area (`controller`, `planner`, or inference), summarize config or checkpoint impacts, link related issues, and include logs, screenshots, or attention-map outputs when behavior changes are visible.

## Configuration & Data Tips
Do not commit datasets, checkpoints, API keys, or hardware secrets. Store large artifacts under local `data/`, `logs/`, or external storage, and rely on `.gitignore` for generated outputs.
