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
import matplotlib.pyplot as plt

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
        if self.wandb_mode != "disabled":
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
        # Initialize variables
        trainer: BaseTrainer = hydra.utils.instantiate(self.cfg.trainer)
        start_epoch = 1

        # Load model weights
        if self.cfg.load_model_from_path:
            if self.cfg.model_path_to_load_from is None:
                root_output_dir = Path(self.output_directory).parent.parent.absolute()
                glob_results = sorted(
                    glob.glob(
                        f"{root_output_dir}/*/*/{self.cfg.experiment_name}*/models/*.pt"
                    )
                )
                if len(glob_results) == 0:
                    raise ValueError(
                        "Cannot load model from output directory since there are no other saved models in the output directory."
                    )
                model_path = str(Path(glob_results[-1]).absolute())
            else:
                model_path = self.cfg.model_path_to_load_from
            lst = model_path.split("_")
            if len(lst) > 1 and lst[1].split(".")[0].isdigit():
                start_epoch = int(lst[1].split(".")[0])
            trainer.load_model(model_path)

        # Process and prepare the dataset
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

        # Train the model
        if self.cfg.train_model:
            os.makedirs(f"{run_output_path}/models")
            for epoch in range(start_epoch, start_epoch + self.cfg.n_epochs):
                train_metrics = {}
                for train_batch in train_dataloader:
                    m = trainer.train(train_batch=train_batch)
                    cur_batch_size = (
                        train_batch.shape[0]
                        if isinstance(train_batch, torch.Tensor)
                        else train_batch[0].shape[0]
                    )
                    for k in m:
                        if k not in train_metrics:
                            train_metrics[k] = []
                        train_metrics[k].append(m[k] * cur_batch_size)
                train_metrics = {
                    k: np.sum(train_metrics[k]) / len(train_dataset)
                    for k in train_metrics
                }

                if (
                    epoch % self.cfg.n_epochs_per_save == 0
                    or epoch == start_epoch + self.cfg.n_epochs - 1
                ):
                    save_model_path = f"{run_output_path}/models/epoch_{epoch}.pt"
                    trainer.save_model(save_model_path)

                test_metrics = {}
                for eval_batch in test_dataloader:
                    test_metrics = trainer.eval(eval_batch=eval_batch)
                self.log_values({"epoch": epoch, **train_metrics, **test_metrics})

        # Evaluate the model
        if self.cfg.eval_model:
            test_metrics = {}
            for eval_batch in test_dataloader:
                test_metrics = trainer.eval(eval_batch=eval_batch)
            logger.info(f"final eval metrics: {json.dumps(test_metrics)}")

        # Visualize Samples
        if self.cfg.visualize_samples is not None:
            d = test_trajs[0].shape[1]
            cont_trajs = trainer.sample(
                test_trajs,
                n_samples=self.cfg.visualize_samples.n_row_samples
                * self.cfg.visualize_samples.n_col_samples,
                traj_length=self.cfg.visualize_samples.traj_length,
                sample_from_lookback=self.cfg.visualize_samples.sample_from_lookback,
            )

            os.makedirs(f"{run_output_path}/plots")
            for i, (traj, samples) in enumerate(zip(test_trajs, cont_trajs)):
                fig, ax = plt.subplots(
                    self.cfg.visualize_samples.n_row_samples,
                    self.cfg.visualize_samples.n_col_samples,
                    figsize=(
                        self.cfg.visualize_samples.n_row_samples
                        * self.cfg.visualize_samples.fig_size_row_multplier,
                        self.cfg.visualize_samples.n_col_samples
                        * self.cfg.visualize_samples.fig_size_col_multiplier,
                    ),
                )
                handles, labels = None, None
                for r in range(self.cfg.visualize_samples.n_row_samples):
                    for c in range(self.cfg.visualize_samples.n_col_samples):
                        sample_idx = r * self.cfg.visualize_samples.n_row_samples + c
                        for feature_idx in range(d):
                            if (
                                self.cfg.visualize_samples.plot_subset_features is None
                                or feature_idx
                                in self.cfg.visualize_samples.plot_subset_features
                            ):
                                ax[r][c].plot(
                                    range(len(traj)),
                                    traj,
                                    label=f"{feature_idx}",
                                )
                                ax[r][c].plot(
                                    range(
                                        len(traj)
                                        - self.cfg.visualize_samples.sample_from_lookback,
                                        len(traj)
                                        - self.cfg.visualize_samples.sample_from_lookback
                                        + self.cfg.visualize_samples.traj_length,
                                    ),
                                    samples[sample_idx, :, feature_idx],
                                    label=f"pred_{feature_idx}",
                                )
                        ax[r][c].set_title(f"Sample {sample_idx}")
                        ax[r][c].set_xlabel(f"Timesteps")
                        ax[r][c].set_ylabel(f"Value")
                        handles, labels = ax[r][c].get_legend_handles_labels()
                fig.legend(handles, labels)
                fig.suptitle(f"Trajectory {i}")
                plt.savefig(f"{run_output_path}/plots/trajectory_{i}_samples.png")
