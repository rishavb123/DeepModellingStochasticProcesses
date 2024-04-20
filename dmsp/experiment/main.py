"""The main experiment entry point."""

from omegaconf import OmegaConf

from experiment_lab.core import run_experiment
from experiment_lab.common.resolvers import register_resolvers

from dmsp.experiment.config import DMSPConfig, register_configs
from dmsp.experiment.experiment import DMSPExperiment


def register_dmsp_resolvers():
    register_resolvers()
    OmegaConf.register_new_resolver("eval", eval)


if __name__ == "__main__":
    run_experiment(
        experiment_cls=DMSPExperiment,
        config_cls=DMSPConfig,
        register_configs=register_configs,
        register_resolvers=register_dmsp_resolvers,
        config_path=f"./configs",
        config_name="dmsp",
    )
