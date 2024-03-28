# Deep Modelling of Stochastic Processes

The goal of this project is to create a model that can generate samples of the underlying stochastic process that generated a target dataset. This is really useful since this allows for various statistics to be estimated over future trajectories using monte carlo rollouts. See our proposal in `reports/proposal.pdf` for more details on the initial ideas and motivation for this work.

## Installation

```bash
conda create -n dsmp_env python=3.11 -y
conda activate dsmp_env
```

Install the version of pytorch that is compatible with your hardware using the pip or conda command from their [website](https://pytorch.org/get-started/locally/).

Then to install this project and its dependencies use the following commands:

```bash
git clone https://github.com/rishavb123/DeepModellingstochasticProcesses.git
cd DeepModellingStochasticProcesses
pip install -e .
```
