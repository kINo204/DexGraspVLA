from collections import deque
from typing import Deque, Dict

import numpy as np
import torch
import torch.nn.functional as F


class SimObservationAdapter:
    """Converts raw simulator observations into the controller input format."""

    def __init__(self, shape_meta: dict, n_obs_steps: int):
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.right_cam_buffer: Deque[np.ndarray] = deque(maxlen=n_obs_steps)
        self.rgbm_buffer: Deque[np.ndarray] = deque(maxlen=n_obs_steps)
        self.state_buffer: Deque[np.ndarray] = deque(maxlen=n_obs_steps)

    def reset(self, obs: Dict[str, np.ndarray]) -> None:
        self.right_cam_buffer.clear()
        self.rgbm_buffer.clear()
        self.state_buffer.clear()
        for _ in range(self.n_obs_steps):
            self.push(obs)

    def push(self, obs: Dict[str, np.ndarray]) -> None:
        self.right_cam_buffer.append(obs["right_cam_img"].copy())
        self.rgbm_buffer.append(obs["rgbm"].copy())
        self.state_buffer.append(obs["right_state"].astype(np.float32).copy())

    def to_model_input(self) -> Dict[str, np.ndarray]:
        obs_dict_np = {}
        raw_obs = {
            "right_cam_img": np.stack(list(self.right_cam_buffer), axis=0),
            "rgbm": np.stack(list(self.rgbm_buffer), axis=0),
            "right_state": np.stack(list(self.state_buffer), axis=0),
        }
        obs_shape_meta = self.shape_meta["obs"]

        for key, attr in obs_shape_meta.items():
            input_type = attr.get("type", "low_dim")
            shape = attr.get("shape")

            if input_type == "rgb":
                rgb = torch.from_numpy(raw_obs[key][..., :3]).float().permute(0, 3, 1, 2)
                rgb = F.interpolate(
                    rgb / 255.0,
                    size=(shape[1], shape[2]),
                    mode="bilinear",
                    align_corners=False,
                )
                obs_dict_np[key] = rgb.numpy()
            elif input_type == "rgbm":
                imgs_in = raw_obs[key]
                rgb = torch.from_numpy(imgs_in[..., :3]).float().permute(0, 3, 1, 2)
                mask = torch.from_numpy(imgs_in[..., 3:]).float().permute(0, 3, 1, 2)
                rgb = F.interpolate(
                    rgb / 255.0,
                    size=(shape[1], shape[2]),
                    mode="bilinear",
                    align_corners=False,
                )
                mask = F.interpolate(mask, size=(shape[1], shape[2]), mode="nearest")
                mask = (mask > 0.5).float()
                obs_dict_np[key] = torch.cat([rgb, mask], dim=1).numpy()
            else:
                obs_dict_np[key] = raw_obs[key].astype(np.float32)

        return obs_dict_np

