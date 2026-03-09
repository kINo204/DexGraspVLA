import os
from typing import Dict, Optional

import cv2
import numpy as np


class EpisodeVideoRecorder:
    """Writes headless rollout videos for remote inspection."""

    def __init__(self, output_path: str, fps: int):
        self.output_path = output_path
        self.fps = fps
        self.writer: Optional[cv2.VideoWriter] = None

    def write(self, obs: Dict[str, np.ndarray]) -> None:
        frame = self._compose_frame(obs)
        if self.writer is None:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        self.writer.write(frame[..., ::-1])

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def _compose_frame(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        wrist = obs["right_cam_img"][..., :3]
        head = obs["rgbm"][..., :3]
        mask = obs["rgbm"][..., 3]
        mask_rgb = np.zeros_like(head)
        mask_rgb[mask > 0] = (0, 255, 0)
        return np.concatenate([wrist, head, mask_rgb], axis=1)

