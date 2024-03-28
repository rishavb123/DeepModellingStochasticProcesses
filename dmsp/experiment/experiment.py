from typing import Any
import logging
from experiment_lab.core import BaseExperiment

from dmsp.experiment.config import DMSPConfig

logger = logging.getLogger(__name__)


class DMSPExperiment(BaseExperiment):
    """The experiment class for deep modelling of stochastic processes."""

    def __init__(self, cfg: DMSPConfig) -> None:
        """The constructor for the experiment class."""
        self.cfg = cfg
        self.initialize_experiment()

    def initialize_experiment(self) -> None:
        """Initializes the experiment that is about to get run."""
        super().initialize_experiment()

    def single_run(
        self, run_id: str, run_output_path: str, seed: int | None = None
    ) -> Any:
        """The entrypoint to the dmsp experiment.

        Args:
            run_id (str): The a unique string id for the run.
            run_output_path (str): The path to output or save anything for the run.
            seed (int | None, optional): The random seed to use for the experiment run. Defaults to None.

        Returns:
            Any: The results from this experiment run.
        """
        logger.info("This is a test!")
