from datetime import datetime

import hydra
import torch
from omegaconf import OmegaConf

from inference_utils.utils import load_config


def now_resolver(pattern: str):
    return datetime.now().strftime(pattern)


OmegaConf.register_new_resolver("now", now_resolver, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path="controller/config",
    config_name="train_dexgraspvla_controller_workspace",
)
def main(cfg):
    # Allow overriding the task from CLI, e.g. task=grasp_mujoco
    if "task" not in cfg:
        cfg = load_config(
            main_config_path="controller/config/train_dexgraspvla_controller_workspace.yaml",
            task_config_path="controller/config/task/grasp_mujoco.yaml",
        )

    if cfg.policy.start_ckpt_path is None:
        raise ValueError("Provide a checkpoint via policy.start_ckpt_path.")

    cfg.task.env_runner.enabled = True
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    policy = workspace.model
    policy.eval().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=workspace.output_dir)
    result = runner.run(policy)
    print(result)


if __name__ == "__main__":
    main()
