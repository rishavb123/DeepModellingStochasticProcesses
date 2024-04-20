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

    load_model_from_path: bool = False
    model_path_to_load_from: str | None = None

    train_model: bool = True
    eval_model: bool = True

    test_proportion: float = 0.2

    batch_size: int = 64
    num_epochs: int = 10

    def __post_init__(self) -> None:
        assert (
            0 <= self.test_proportion <= 1
        ), "Test proportion must be between 0 and 1."
        return super().__post_init__()


def register_configs():
    """Registers the configs created."""
    cs = ConfigStore.instance()
    cs.store(name="dmsp_config", node=DMSPConfig)
