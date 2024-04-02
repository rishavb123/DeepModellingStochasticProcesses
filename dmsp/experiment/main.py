"""The main experiment entry point."""

from experiment_lab.core import run_experiment

from dmsp.experiment.config import DMSPConfig, register_configs
from dmsp.experiment.experiment import DMSPExperiment

if __name__ == "__main__":
    run_experiment(
        experiment_cls=DMSPExperiment,
        config_cls=DMSPConfig,
        register_configs=register_configs,
        config_path=f"./configs",
        config_name="dmsp",
    )
