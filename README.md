# Deep Modelling of Stochastic Processes

The goal of this project is to create a model that can generate samples of the underlying stochastic process that generated a target dataset. This is really useful since this allows for various statistics to be estimated over future trajectories using monte carlo rollouts. See our proposal in `reports/proposal.pdf` for more details on the initial ideas and motivation for this work.

## Installation

```bash
conda create -n dmsp_env python=3.11 -y
conda activate dmsp_env
```

Install the version of pytorch that is compatible with your hardware using the pip or conda command from their [website](https://pytorch.org/get-started/locally/).

Then to install this project and its dependencies use the following commands:

```bash
git clone https://github.com/rishavb123/DeepModellingstochasticProcesses.git
cd DeepModellingStochasticProcesses
pip install -e .
```

## Example Run Commands

Run a conditional VAE on a yfinance experiment with a CNN architecture:

```bash
./scripts/run.sh --config-name yfinance_experiment +trainer=vae +networks/vae/encoder@trainer.vae.encoder=medium_cnn +networks/vae/decoder@trainer.vae.decoder=medium_cnn n_epochs=1000 n_epochs_per_save=30 trainer.lookback=50
```

Run SadEmilie trainer on a yfinance experiment with a CNN architecture:

```bash
./scripts/run.sh --config-name yfinance_experiment +trainer=sad_emilie +networks/sad_emilie@trainer.prediction_model=medium_cnn n_epochs=1000 n_epochs_per_save=30 trainer.lookback=50
```

Run SadEmilie trainer on a yfinance experiment with a S4 architecture with a longer sequence length:

```bash
./scripts/run.sh --config-name yfinance_experiment +trainer=sad_emilie +networks/sad_emilie@trainer.prediction_model=medium_s4 n_epochs=1000 n_epochs_per_save=30 trainer.lookback=300
```
