"""File for all the config and config registration for dmsp experiments."""

from typing import Any, Dict

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from experiment_lab.core import BaseConfig


class DMSPConfig(BaseConfig):
    """The config for the dmsp experiment."""

    model: Dict[str, Any] = MISSING

    data_loader: Dict[str, Any] = MISSING
    force_redownload_dataset: bool = False

    def __post_init__(self) -> None:
        return super().__post_init__()


def register_configs():
    """Registers the configs created."""
    cs = ConfigStore.instance()
    cs.store(name="dmsp_config", node=DMSPConfig)
