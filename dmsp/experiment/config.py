"""File for all the config and config registration for dmsp experiments."""

from typing import Any, Dict, List

from omegaconf import MISSING
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from experiment_lab.core import BaseConfig


@dataclass
class VisualizeSamples:
    n_samples: int = 3
    traj_length: int = 50
    sample_from_lookback: int = 10
    plot_subset_features: List[int] | None = None
    fig_size_row_multiplier: int = 10
    fig_size_col_multiplier: int = 4


@dataclass
class DMSPConfig(BaseConfig):
    """The config for the dmsp experiment."""

    data_loader: Dict[str, Any] = MISSING
    force_redownload_dataset: bool = False

    trainer: Dict[str, Any] = MISSING

    load_model_from_path: bool = False
    model_path_to_load_from: str | None = None

    train_model: bool = True
    eval_model: bool = True
    visualize_samples: VisualizeSamples | None = field(default_factory=VisualizeSamples)

    test_proportion: float = 0.2

    batch_size: int = 64
    n_epochs: int = 10
    n_epochs_per_save: int = 10

    def __post_init__(self) -> None:
        assert (
            0 <= self.test_proportion <= 1
        ), "Test proportion must be between 0 and 1."
        return super().__post_init__()


def register_configs():
    """Registers the configs created."""
    cs = ConfigStore.instance()
    cs.store(name="dmsp_config", node=DMSPConfig)
