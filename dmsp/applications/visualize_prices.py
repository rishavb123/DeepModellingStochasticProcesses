from typing import Any, Dict, List

import hydra
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from dmsp.datasets.finance.yfinance_loader import YFinanceLoader
from dmsp.models.trainers.sad_emilie import SadEmilie
from dmsp.datasets.base_loader import BaseLoader
from experiment_lab.common.resolvers import register_resolvers

output_path = "/Users/rbhagat/Downloads/22-09-15/"
model_path = "/Users/rbhagat/Downloads/22-09-15/yfinance_experiment_1714356555_0_1/models/epoch_510.pt"

save_path = "/Users/rbhagat/Downloads/plots"

START_DAY = "2020-01-01"
END_DAY = "2022-12-31"

DAYS_BEFORE = 200
FORCE_RESAMPLE = False

N_SAMPLES = 1000
PLOT_SAMPLES = 5
PLOT_SAMPLES_OVERLAPPING = 30
TRAJ_LENGTH = 400


def save_samples(
    trainer: SadEmilie,
    test_trajs: List[np.ndarray],
    cont_trajs: List[np.ndarray],
    data_loader: BaseLoader,
) -> Dict[str, Any]:
    test_trajs = trainer.validate_traj_list(
        trajectory_list=test_trajs,
        sample_from_lookback=DAYS_BEFORE,
    )
    d = test_trajs[0].shape[1]

    for i, (traj, samples) in enumerate(zip(test_trajs, cont_trajs)):
        fig, ax = plt.subplots(
            d,
            PLOT_SAMPLES,
            figsize=(
                d * 8,
                PLOT_SAMPLES * 4,
            ),
        )
        for feature_idx in range(d):
            ax_row = ax[feature_idx] if d > 1 else ax
            for sample_idx in range(PLOT_SAMPLES):
                ax_row[sample_idx].plot(
                    range(len(traj)),
                    traj[:, feature_idx],
                    label=f"ground_truth",
                    color="red",
                )
                ax_row[sample_idx].plot(
                    range(
                        len(traj) - DAYS_BEFORE,
                        len(traj) - DAYS_BEFORE + TRAJ_LENGTH,
                    ),
                    samples[sample_idx, :, feature_idx],
                    label=f"pred_sample",
                    color="blue",
                )
                feature_name = (
                    data_loader.feature_names[feature_idx]
                    .replace("_", " ")
                    .replace("-", " ")
                    .title()
                )
                ax_row[sample_idx].set_title(f"Sample {sample_idx}; {feature_name}")
                ax_row[sample_idx].set_xlabel(f"Timesteps")
                ax_row[sample_idx].set_ylabel(f"Value")
                ax_row[sample_idx].set_ylim((0, 2.5))

        fig.suptitle(f"Trajectory {i}")
        fig.tight_layout()
        plt.savefig(f"{save_path}/trajectory_{i}_samples.png")
        plt.close()


def save_samples_means(
    trainer: SadEmilie,
    test_trajs: List[np.ndarray],
    cont_trajs: List[np.ndarray],
    data_loader: BaseLoader,
) -> Dict[str, Any]:
    test_trajs = trainer.validate_traj_list(
        trajectory_list=test_trajs,
        sample_from_lookback=DAYS_BEFORE,
    )
    d = test_trajs[0].shape[1]

    for i, (traj, samples) in enumerate(zip(test_trajs, cont_trajs)):
        fig, ax = plt.subplots(
            d,
            figsize=(4, 16),
        )
        for feature_idx in range(d):
            ax_row = ax[feature_idx] if d > 1 else ax
            mu = samples[:, :, feature_idx].mean(axis=0)
            std = samples[:, :, feature_idx].std(axis=0)
            ax_row.fill_between(
                range(
                    len(traj) - DAYS_BEFORE,
                    len(traj) - DAYS_BEFORE + TRAJ_LENGTH,
                ),
                mu - 2 * std,
                mu + 2 * std,
                color="blue",
                alpha=0.2,
                label="confidence interval",
            )
            ax_row.plot(
                range(
                    len(traj) - DAYS_BEFORE,
                    len(traj) - DAYS_BEFORE + TRAJ_LENGTH,
                ),
                mu,
                label="predicted mean",
                color="blue",
            )
            ax_row.plot(
                range(len(traj)),
                traj[:, feature_idx],
                label=f"ground_truth",
                color="red",
            )
            feature_name = (
                data_loader.feature_names[feature_idx]
                .replace("_", " ")
                .replace("-", " ")
                .title()
            )
            ax_row.set_title(feature_name)
            ax_row.set_xlabel(f"Timesteps")
            ax_row.set_ylabel(f"Value")
            ax_row.set_ylim((0, 2.5))

        fig.suptitle(f"Trajectory {i}")
        fig.tight_layout()
        plt.savefig(f"{save_path}/trajectory_{i}_means.png")
        plt.close()


def save_samples_overlapping(
    trainer: SadEmilie,
    test_trajs: List[np.ndarray],
    cont_trajs: List[np.ndarray],
    data_loader: BaseLoader,
) -> Dict[str, Any]:
    test_trajs = trainer.validate_traj_list(
        trajectory_list=test_trajs,
        sample_from_lookback=DAYS_BEFORE,
    )
    d = test_trajs[0].shape[1]

    for i, (traj, samples) in enumerate(zip(test_trajs, cont_trajs)):
        fig, ax = plt.subplots(
            d,
            figsize=(4, 16),
        )
        for feature_idx in range(d):
            ax_row = ax[feature_idx] if d > 1 else ax
            for j in range(PLOT_SAMPLES_OVERLAPPING):
                ax_row.plot(
                    range(
                        len(traj) - DAYS_BEFORE,
                        len(traj) - DAYS_BEFORE + TRAJ_LENGTH,
                    ),
                    samples[j, :, feature_idx],
                    label="predicted mean",
                    color="gray",
                    alpha=0.4,
                )
            ax_row.plot(
                range(len(traj)),
                traj[:, feature_idx],
                label=f"ground_truth",
                color="red",
            )
            feature_name = (
                data_loader.feature_names[feature_idx]
                .replace("_", " ")
                .replace("-", " ")
                .title()
            )
            ax_row.set_title(feature_name)
            ax_row.set_xlabel(f"Timesteps")
            ax_row.set_ylabel(f"Value")
            ax_row.set_ylim((0, 2.5))

        fig.suptitle(f"Trajectory {i}")
        fig.tight_layout()
        plt.savefig(f"{save_path}/trajectory_{i}_samples_overlapping.png")
        plt.close()

def save_samples_stds(
    trainer: SadEmilie,
    test_trajs: List[np.ndarray],
    cont_trajs: List[np.ndarray],
    data_loader: BaseLoader,
) -> Dict[str, Any]:
    test_trajs = trainer.validate_traj_list(
        trajectory_list=test_trajs,
        sample_from_lookback=DAYS_BEFORE,
    )
    d = test_trajs[0].shape[1]

    for i, (traj, samples) in enumerate(zip(test_trajs, cont_trajs)):
        fig, ax = plt.subplots(
            d,
            figsize=(4, 16),
        )
        for feature_idx in range(d):
            ax_row = ax[feature_idx] if d > 1 else ax
            ax_row.plot(
                range(
                    len(traj) - DAYS_BEFORE,
                    len(traj) - DAYS_BEFORE + TRAJ_LENGTH,
                ),
                samples[:, :, feature_idx].std(axis=0),
                label="predicted mean",
                color="gray",
                alpha=0.4,
            )
            feature_name = (
                data_loader.feature_names[feature_idx]
                .replace("_", " ")
                .replace("-", " ")
                .title()
            )
            ax_row.set_title(feature_name)
            ax_row.set_xlabel(f"Timesteps")
            ax_row.set_ylabel(f"Value")

        fig.suptitle(f"Trajectory {i}")
        fig.tight_layout()
        plt.savefig(f"{save_path}/trajectory_{i}_std.png")
        plt.close()


def main():

    print("Loading the config...")

    register_resolvers()

    config = OmegaConf.load(f"{output_path}/.hydra/config.yaml")

    OmegaConf.resolve(config)

    print("Loading the data...")

    loader = YFinanceLoader(
        symbols=list(config["data_loader"]["symbols"]),
        download_kwargs=[
            {
                "start": START_DAY,
                "end": END_DAY,
                "interval": "1d",
            }
        ],
        columns=list(config["data_loader"]["columns"]),
        normalize_by_first_price=bool(
            config["data_loader"]["normalize_by_first_price"]
        ),
    )
    loader.load()

    print("Sampling the future trajectories...")

    trainer: SadEmilie = hydra.utils.instantiate(config["trainer"])
    trainer.load_model(model_path)

    if os.path.exists(f"{save_path}/trajs.pkl") and not FORCE_RESAMPLE:
        future_traj_list = torch.load(f"{save_path}/trajs.pkl")
    else:
        future_traj_list = trainer.sample(
            loader.data,
            n_samples=N_SAMPLES,
            traj_length=TRAJ_LENGTH,
            sample_from_lookback=DAYS_BEFORE,
        )
        torch.save(future_traj_list, f"{save_path}/trajs.pkl")

    print("Plotting graphs...")

    save_samples(
        trainer=trainer,
        test_trajs=loader.data,
        cont_trajs=future_traj_list,
        data_loader=loader,
    )
    save_samples_means(
        trainer=trainer,
        test_trajs=loader.data,
        cont_trajs=future_traj_list,
        data_loader=loader,
    )
    save_samples_overlapping(
        trainer=trainer,
        test_trajs=loader.data,
        cont_trajs=future_traj_list,
        data_loader=loader,
    )
    save_samples_stds(
        trainer=trainer,
        test_trajs=loader.data,
        cont_trajs=future_traj_list,
        data_loader=loader,
    )

    print("Done!")


if __name__ == "__main__":
    main()
