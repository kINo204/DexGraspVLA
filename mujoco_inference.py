import argparse
import os
from datetime import datetime

import hydra
import torch
from omegaconf import OmegaConf

from inference_utils.utils import load_config


def now_resolver(pattern: str):
    return datetime.now().strftime(pattern)


OmegaConf.register_new_resolver("now", now_resolver, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DexGrasp-VLA in a headless MuJoCo simulation.")
    parser.add_argument(
        "--task-config",
        default=os.path.join("controller", "config", "task", "grasp_mujoco.yaml"),
        help="Task config with MuJoCo env settings.",
    )
    parser.add_argument(
        "--main-config",
        default=os.path.join("controller", "config", "train_dexgraspvla_controller_workspace.yaml"),
        help="Main controller training config.",
    )
    parser.add_argument("--checkpoint", default=None, help="Optional controller checkpoint override.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of rollout episodes.")
    parser.add_argument("--steps", type=int, default=200, help="Maximum environment steps per episode.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(main_config_path=args.main_config, task_config_path=args.task_config)
    if args.checkpoint is not None:
        cfg.policy.start_ckpt_path = args.checkpoint

    if cfg.policy.start_ckpt_path is None:
        raise ValueError("Provide a checkpoint via --checkpoint or policy.start_ckpt_path.")

    cfg.task.env_runner.enabled = True
    cfg.task.env_runner.max_episodes = args.episodes
    cfg.task.env_runner.max_steps = args.steps

    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    policy = workspace.model
    policy.eval().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=workspace.output_dir)
    result = runner.run(policy)
    print(result)


if __name__ == "__main__":
    main()
