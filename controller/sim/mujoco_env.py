from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import gymnasium as _gym
except ImportError:
    _gym = None


def _import_backends():
    try:
        import gymnasium as gym
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "MuJoCo simulation requires the optional packages listed in requirements-sim.txt."
        ) from exc
    return gym, mujoco


@dataclass
class JointBinding:
    qpos_ids: np.ndarray
    ctrl_ids: np.ndarray


class DexGraspMujocoEnv(_gym.Env if _gym is not None else object):
    """Gymnasium-compatible MuJoCo environment for controller rollouts."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        model_path: str,
        robot: dict,
        task: dict,
        render_width: int = 640,
        render_height: int = 480,
        render_mode: str = "rgb_array",
        wrist_camera: str = "wrist",
        third_person_camera: str = "third",
        frame_skip: int = 10,
        action_low: Optional[Sequence[float]] = None,
        action_high: Optional[Sequence[float]] = None,
        mask: Optional[dict] = None,
    ):
        gym, mujoco = _import_backends()
        self.gym = gym
        self.mujoco = mujoco

        if not model_path:
            raise ValueError("A MuJoCo model_path must be provided.")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_height
        self.wrist_camera = wrist_camera
        self.third_person_camera = third_person_camera
        self.frame_skip = frame_skip
        self.robot_cfg = robot
        self.task_cfg = task
        self.mask_cfg = mask or {}
        self.step_count = 0

        self.arm_lower_limits = np.asarray(robot["arm_lower_limits"], dtype=np.float32)
        self.arm_upper_limits = np.asarray(robot["arm_upper_limits"], dtype=np.float32)
        self.hand_action_low = np.asarray(robot["hand_action_low"], dtype=np.float32)
        self.hand_action_high = np.asarray(robot["hand_action_high"], dtype=np.float32)
        self.initial_arm_qpos = np.asarray(robot["initial_arm_qpos"], dtype=np.float32)
        self.initial_hand_qpos = np.asarray(robot["initial_hand_qpos"], dtype=np.float32)

        self.arm_binding = self._bind_joints_and_actuators(
            robot["arm_joint_names"], robot.get("arm_actuator_names", [])
        )
        self.hand_binding = self._bind_joints_and_actuators(
            robot["hand_joint_names"], robot.get("hand_actuator_names", []), allow_tendons=True
        )
        self.mask_geom_ids = self._resolve_ids(
            self.mask_cfg.get("geom_names", []), mujoco.mjtObj.mjOBJ_GEOM
        )
        self.success_body_ids = self._resolve_ids(
            self.task_cfg.get("success_body_names", []), mujoco.mjtObj.mjOBJ_BODY
        )

        act_low = np.asarray(action_low if action_low is not None else [-1.0] * 13, dtype=np.float32)
        act_high = np.asarray(action_high if action_high is not None else [1.0] * 13, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "right_cam_img": gym.spaces.Box(
                    low=0, high=255, shape=(render_height, render_width, 3), dtype=np.uint8
                ),
                "rgbm": gym.spaces.Box(
                    low=0, high=255, shape=(render_height, render_width, 4), dtype=np.uint8
                ),
                "right_state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
                ),
            }
        )

        self.renderers = {
            "wrist": mujoco.Renderer(self.model, render_height, render_width),
            "third": mujoco.Renderer(self.model, render_height, render_width),
        }
        self._segmentation_enabled = False
        self.metadata["render_fps"] = self.task_cfg.get("render_fps", self.metadata["render_fps"])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if _gym is not None:
            super_reset = getattr(super(), "reset", None)
            if callable(super_reset):
                super_reset(seed=seed)
        del options

        self.mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self._set_joint_qpos(self.arm_binding.qpos_ids, self.initial_arm_qpos)
        self._set_joint_qpos(self.hand_binding.qpos_ids, self.initial_hand_qpos)
        # Forward once to get arm pose, then align the hand root to the arm end-effector.
        self.mujoco.mj_forward(self.model, self.data)
        self._align_hand_root()
        self.mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        return obs, {"is_success": False}

    def _align_hand_root(self) -> None:
        """Snap the freejoint root of the hand to the arm end-effector pose."""
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "rh_forearm")
        arm_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "arm_ee")
        if body_id < 0 or arm_id < 0:
            return
        jnt_adr = self.model.body_jntadr[body_id]
        jnt_num = self.model.body_jntnum[body_id]
        if jnt_num <= 0:
            return
        if self.model.jnt_type[jnt_adr] != self.mujoco.mjtJoint.mjJNT_FREE:
            return
        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        self.data.qpos[qpos_adr:qpos_adr + 3] = self.data.xpos[arm_id]
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = self.data.xquat[arm_id]

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        self._apply_action(action)
        self.mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        self.step_count += 1
        obs = self._get_obs()
        success = self._is_success()
        reward = float(self.task_cfg.get("reward_scale", 1.0)) if success else 0.0
        terminated = success
        truncated = self.step_count >= int(self.task_cfg.get("episode_horizon", 200))
        info = {"is_success": success, "step_count": self.step_count}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._render_rgb(self.third_person_camera)

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()

    def _bind_joints_and_actuators(
        self, joint_names: Sequence[str], actuator_names: Sequence[str], allow_tendons: bool = False
    ) -> JointBinding:
        qpos_ids = []
        for joint_name in joint_names:
            if allow_tendons and joint_name.startswith("rh_") and joint_name.endswith("0"):
                # Tendon-driven joints do not map to qpos directly.
                continue
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Unknown MuJoCo joint: {joint_name}")
            qpos_ids.append(self.model.jnt_qposadr[joint_id])

        ctrl_ids = []
        for actuator_name in actuator_names:
            actuator_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator_id < 0:
                raise ValueError(f"Unknown MuJoCo actuator: {actuator_name}")
            ctrl_ids.append(actuator_id)

        return JointBinding(
            qpos_ids=np.asarray(qpos_ids, dtype=np.int32),
            ctrl_ids=np.asarray(ctrl_ids, dtype=np.int32),
        )

    def _resolve_ids(self, names: Sequence[str], obj_type) -> List[int]:
        ids = []
        for name in names:
            obj_id = self.mujoco.mj_name2id(self.model, obj_type, name)
            if obj_id >= 0:
                ids.append(int(obj_id))
        return ids

    def _set_joint_qpos(self, qpos_ids: np.ndarray, qpos_values: np.ndarray) -> None:
        if qpos_ids.size == 0:
            return
        qpos_values = np.asarray(qpos_values, dtype=np.float32)
        if qpos_values.shape[0] < qpos_ids.shape[0]:
            qpos_values = np.pad(qpos_values, (0, qpos_ids.shape[0] - qpos_values.shape[0]))
        self.data.qpos[qpos_ids] = qpos_values[: qpos_ids.shape[0]]

    def _apply_action(self, action: np.ndarray) -> None:
        arm_action = self._scale_arm_action(action[:7])
        hand_action = np.clip(action[7:], self.hand_action_low, self.hand_action_high)
        self._apply_target(self.arm_binding, arm_action)
        self._apply_target(self.hand_binding, hand_action)

    def _apply_target(self, binding: JointBinding, target: np.ndarray) -> None:
        if binding.ctrl_ids.size > 0:
            self.data.ctrl[binding.ctrl_ids] = target
            return

        if binding.qpos_ids.size > 0:
            self.data.qpos[binding.qpos_ids] = target
            self.mujoco.mj_forward(self.model, self.data)

    def _scale_arm_action(self, arm_action: np.ndarray) -> np.ndarray:
        return 0.5 * (arm_action + 1.0) * (self.arm_upper_limits - self.arm_lower_limits) + self.arm_lower_limits

    def _unscale_arm_state(self, arm_qpos: np.ndarray) -> np.ndarray:
        return 2.0 * (arm_qpos - self.arm_lower_limits) / (self.arm_upper_limits - self.arm_lower_limits) - 1.0

    def _get_obs(self) -> Dict[str, np.ndarray]:
        wrist = self._render_rgb(self.wrist_camera)
        head = self._render_rgb(self.third_person_camera)
        mask = self._render_mask(self.third_person_camera)
        rgbm = np.concatenate([head, mask[..., None]], axis=-1)
        hand_qpos = self.data.qpos[self.hand_binding.qpos_ids].astype(np.float32)
        if hand_qpos.shape[0] < 6:
            hand_qpos = np.pad(hand_qpos, (0, 6 - hand_qpos.shape[0]))
        right_state = np.concatenate(
            [self._unscale_arm_state(self.data.qpos[self.arm_binding.qpos_ids]), hand_qpos[:6]],
            axis=0,
        ).astype(np.float32)
        return {"right_cam_img": wrist, "rgbm": rgbm, "right_state": right_state}

    def _render_rgb(self, camera_name: str) -> np.ndarray:
        renderer = self.renderers["wrist" if camera_name == self.wrist_camera else "third"]
        renderer.update_scene(self.data, camera=camera_name)
        return renderer.render().astype(np.uint8)

    def _render_mask(self, camera_name: str) -> np.ndarray:
        if self.mask_cfg.get("source", "none") != "segmentation" or not self.mask_geom_ids:
            return np.zeros((self.render_height, self.render_width), dtype=np.uint8)

        renderer = self.renderers["third" if camera_name == self.third_person_camera else "wrist"]
        if hasattr(renderer, "enable_segmentation_rendering"):
            renderer.enable_segmentation_rendering()
            self._segmentation_enabled = True
        renderer.update_scene(self.data, camera=camera_name)
        segmentation = renderer.render()
        if self._segmentation_enabled and hasattr(renderer, "disable_segmentation_rendering"):
            renderer.disable_segmentation_rendering()

        if segmentation.ndim == 3 and segmentation.shape[-1] >= 2:
            geom_ids = segmentation[..., 0]
        else:
            geom_ids = segmentation

        mask = np.isin(geom_ids, np.asarray(self.mask_geom_ids)).astype(np.uint8)
        return mask

    def _is_success(self) -> bool:
        if not self.success_body_ids:
            return False
        height_threshold = float(self.task_cfg.get("success_height_threshold", 0.15))
        for body_id in self.success_body_ids:
            if self.data.xpos[body_id, 2] >= height_threshold:
                return True
        return False
