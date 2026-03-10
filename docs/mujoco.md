# MuJoCo Simulation Setup

This repository now includes a headless MuJoCo rollout path for DexGrasp-VLA. It is designed for server execution and writes episode MP4s instead of relying on an interactive viewer.

## Optional Python Dependencies

These packages are not installed automatically in this repository. Install them manually in the target environment when ready:

```bash
pip install -r requirements-sim.txt
```

## Required External Assets

The simulator code still needs repository-local or server-local assets that are specific to your robot and scene:

- A MuJoCo XML/MJCF model file referenced by `controller/config/task/grasp_mujoco.yaml`
- Joint names for the 7 arm DoFs and 6 hand DoFs
- Matching actuator names if the model uses position/servo actuators
- Camera names for the wrist and third-person views
- Geom names for segmentation masks and body names for success checks

This repo now includes `assets/mujoco/dexgrasp_shadowhand_arm_scene.xml`, which expects the ShadowHand MJCF from MuJoCo Menagerie to exist at:

`third_party/mujoco_menagerie/shadow_hand/right_hand.xml`

The Menagerie ShadowHand root body is `rh_forearm`, and the weld in `assets/mujoco/dexgrasp_shadowhand_arm_scene.xml` is already set accordingly.

The hand joint/actuator names in `controller/config/task/grasp_mujoco.yaml` now match Menagerie. If you switch to a different ShadowHand source, update those names.

## Server Notes

For Linux servers, headless rendering is typically run with `MUJOCO_GL=egl`. Save videos by enabling the MuJoCo env runner and setting a valid `model_path` in `grasp_mujoco.yaml`.

## Menagerie Assets

Clone MuJoCo Menagerie into `third_party/mujoco_menagerie` so the ShadowHand include path resolves:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
```

## Example Commands

Run a controller checkpoint in simulation:

```bash
python mujoco_inference.py --checkpoint path/to/model.ckpt
```

Use the MuJoCo task config during training-time rollouts:

```bash
python train.py task=grasp_mujoco
```
