import os
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import hydra

from controller.common.pytorch_util import dict_apply
from controller.env_runner.base_image_runner import BaseImageRunner
from controller.policy.base_image_policy import BaseImagePolicy
from controller.sim.obs_adapter import SimObservationAdapter
from controller.sim.video_recorder import EpisodeVideoRecorder


class MuJoCoImageRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        env: dict,
        shape_meta: dict,
        n_obs_steps: int,
        enabled: bool = False,
        max_episodes: int = 1,
        max_steps: int = 200,
        save_video: bool = True,
        save_data: bool = False,
        strict_dependencies: bool = False,
        debug_action: bool = False,
        debug_action_scale: float = 0.2,
    ):
        super().__init__(output_dir)
        self.env_cfg = env
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.enabled = enabled
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.save_video = save_video
        self.save_data = save_data
        self.strict_dependencies = strict_dependencies
        self.debug_action = debug_action
        self.debug_action_scale = debug_action_scale

    def run(self, policy: BaseImagePolicy) -> Dict:
        if not self.enabled:
            return {"mujoco_rollout_skipped": 1.0}

        try:
            from controller.sim.mujoco_env import DexGraspMujocoEnv
        except ImportError:
            if self.strict_dependencies:
                raise
            return {"mujoco_rollout_skipped": 1.0, "mujoco_missing_dependencies": 1.0}

        if not self.env_cfg.get("model_path"):
            return {"mujoco_rollout_skipped": 1.0, "mujoco_missing_model_path": 1.0}
        if not os.path.isabs(self.env_cfg["model_path"]):
            try:
                base_dir = hydra.utils.get_original_cwd()
            except Exception:
                base_dir = os.getcwd()
            self.env_cfg["model_path"] = os.path.abspath(os.path.join(base_dir, self.env_cfg["model_path"]))

        rollout_dir = os.path.join(self.output_dir, "sim_rollouts", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(rollout_dir, exist_ok=True)
        env = DexGraspMujocoEnv(**self.env_cfg)
        adapter = SimObservationAdapter(shape_meta=self.shape_meta, n_obs_steps=self.n_obs_steps)
        device = policy.device

        total_reward = 0.0
        success_count = 0
        executed_steps = 0

        try:
            for episode_idx in range(self.max_episodes):
                obs, _ = env.reset()
                adapter.reset(obs)
                policy.reset()
                episode_reward = 0.0
                episode_frames = []
                recorder = None

                if self.save_video:
                    video_path = os.path.join(rollout_dir, f"episode_{episode_idx:03d}.mp4")
                    recorder = EpisodeVideoRecorder(video_path, fps=int(self.env_cfg.get("task", {}).get("render_fps", 20)))
                    recorder.write(obs)

                for step_idx in range(self.max_steps):
                    model_obs_np = adapter.to_model_input()
                    model_obs = dict_apply(
                        model_obs_np,
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device=device),
                    )
                    if self.debug_action:
                        phase = step_idx * 0.1
                        action = np.zeros((13,), dtype=np.float32)
                        action[:7] = self.debug_action_scale * np.sin(
                            phase + np.arange(7, dtype=np.float32) * 0.5
                        )
                    else:
                        with torch.no_grad():
                            action_pred = policy.predict_action(model_obs)
                        action = action_pred[0, 0].detach().cpu().numpy()

                    next_obs, reward, terminated, truncated, info = env.step(action)
                    adapter.push(next_obs)
                    episode_reward += float(reward)
                    executed_steps += 1

                    if recorder is not None:
                        recorder.write(next_obs)
                    if self.save_data:
                        episode_frames.append(
                            {
                                "right_state": next_obs["right_state"].copy(),
                                "action": np.asarray(action, dtype=np.float32).copy(),
                                "reward": float(reward),
                            }
                        )

                    if terminated or truncated:
                        success_count += int(bool(info.get("is_success", False)))
                        break

                total_reward += episode_reward
                if recorder is not None:
                    recorder.close()
                if self.save_data:
                    np.savez_compressed(
                        os.path.join(rollout_dir, f"episode_{episode_idx:03d}.npz"),
                        frames=np.asarray(episode_frames, dtype=object),
                    )
        finally:
            env.close()

        return {
            "mujoco_rollout_reward": total_reward / max(self.max_episodes, 1),
            "mujoco_rollout_success_rate": success_count / max(self.max_episodes, 1),
            "mujoco_rollout_steps": float(executed_steps),
        }
