"""File for all the config and config registration for dmsp experiments."""

from hydra.core.config_store import ConfigStore
from experiment_lab.core import BaseConfig


class DMSPConfig(BaseConfig):
    """The config for the dmsp experiment."""

    pass


def register_configs():
    """Registers the configs created."""
    cs = ConfigStore.instance()
    cs.store(name="dmsp_config", node=DMSPConfig)
