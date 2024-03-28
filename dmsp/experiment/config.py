from hydra.core.config_store import ConfigStore
from experiment_lab.core import BaseConfig


class DMSPConfig(BaseConfig):

    pass


def register_configs():
    """Registers the configs created."""
    cs = ConfigStore.instance()
    cs.store(name="dmsp_config", node=DMSPConfig)
