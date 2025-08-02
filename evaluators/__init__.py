from omegaconf import DictConfig, OmegaConf

from .ultralytics_validator import PoseValidator
from .pose6d_validator import Pose6DValidator


def build_validator(cfg, task=None):
    assert task is not None, "Task must be specified for building validator"
    names = cfg.dataset.names
    if isinstance(names, DictConfig):
        names = OmegaConf.to_container(names, resolve=True)

    if task == "pose":
        validator = PoseValidator(names, cfg.dataset.kpt_type)
    elif task == "pose6d":
        classes = list(names.values())
        validator = Pose6DValidator(classes, use_matches_for_pose=True)
    else:
        raise ValueError(f"Unsupported task: {task}")
    return validator
