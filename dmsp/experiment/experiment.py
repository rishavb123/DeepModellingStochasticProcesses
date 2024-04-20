"""File for the actual dmsp experiment code."""

from typing import Any, Dict
import logging
import torch.utils.data
import wandb
import hydra
import json
import os
import numpy as np
from pathlib import Path
import glob
import torch

from experiment_lab.core import BaseExperiment

from dmsp.experiment.config import DMSPConfig
from dmsp.models.trainers.base_trainer import BaseTrainer
from dmsp.datasets.base_loader import BaseLoader

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
        self.wandb_mode = (
            self.cfg.wandb["mode"]
            if self.cfg.wandb is not None and "mode" in self.cfg.wandb
            else (None if self.cfg.wandb is not None else "disabled")
        )

        self.data_loader: BaseLoader = hydra.utils.instantiate(self.cfg.data_loader)
        self.data_loader.load(force_redownload=self.cfg.force_redownload_dataset)

    def log_values(self, dict_to_log: Dict[str, Any]) -> None:
        """Function to log a dictionary to stdout and wandb (if applicable)

        Args:
            dict_to_log (Dict[str, Any]): The dictionary to log.
        """
        logger.info(json.dumps(dict_to_log))
        if self.wandb_mode:
            wandb.log(dict_to_log)

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
        trainer: BaseTrainer = hydra.utils.instantiate(self.cfg.trainer)

        start_epoch = 0

        if self.cfg.load_model_from_path:
            if self.cfg.model_path_to_load_from is None:
                root_output_dir = Path(self.output_directory).parent.parent.absolute()
                glob_results = sorted(glob.glob(f"{root_output_dir}/*/*/models/*"))
                if len(glob_results) == 0:
                    raise ValueError(
                        "Cannot load model from output directory since there are no other saved models in the output directory."
                    )
                model_path = Path(glob_results[-1])
            else:
                model_path = self.cfg.model_path_to_load_from
            lst = model_path.split("_")
            if len(lst) > 1 and lst[1].isdigit():
                start_epoch = int(lst[1])
            trainer.load_model(model_path)

        if 0 < self.cfg.test_proportion < 1:
            train_trajs, test_trajs = self.data_loader.split_data(
                (1 - self.cfg.test_proportion, self.cfg.test_proportion)
            )
        elif self.cfg.test_proportion == 0:
            train_trajs, test_trajs = self.data_loader.data, []
        else:
            train_trajs, test_trajs = [], self.data_loader.data

        train_dataset = trainer.preprocess(train_trajs)
        test_dataset = trainer.preprocess(test_trajs)

        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.cfg.batch_size, shuffle=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=len(test_dataset), shuffle=True
        )

        if self.cfg.train_model:
            for epoch in range(start_epoch, start_epoch + self.cfg.num_epochs):
                train_metrics = {}
                for train_batch in train_dataloader:
                    m = trainer.train(train_batch=train_batch)
                    for k in m:
                        if k not in train_metrics:
                            train_metrics[k] = []
                        train_metrics[k].append(m[k])
                train_metrics = {k: np.mean(train_metrics[k]) for k in train_metrics}

                save_model_path = f"{run_output_path}/models/epoch_{epoch}"
                os.makedirs(save_model_path)
                trainer.save_model(save_model_path)

                test_metrics = {}
                for eval_batch in test_dataloader:
                    test_metrics = trainer.eval(eval_batch=eval_batch, visualize=False)
                self.log_values({"epoch": epoch, **train_metrics, **test_metrics})

        if self.cfg.eval_model:
            test_metrics = {}
            for eval_batch in test_dataloader:
                test_metrics = trainer.eval(eval_batch=eval_batch, visualize=True)
            logger.info(f"final eval metrics: {json.dumps(test_metrics)}")
