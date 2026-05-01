# FedA2L: Adaptive Layer-wise Learning Rate Adjustment in Decentralized Federated Learning

## Overview

This repository implements **FedA2L**, a **Decentralized Federated Learning (DFL)** framework with **adaptive layer-wise learning rate adjustment** across nodes and neighbors.

The goal of FedA2L is to improve **convergence speed** and **convergence stability** in DFL under non-IID data. FedA2L dynamically assigns layer-wise learning rates using model divergence signals, balancing local adaptation and network consistency, and integrates into existing DFL workflows **without extra communication or coordination overhead**.

> **Note**: This codebase is developed and tested primarily on Linux-based systems (Ubuntu recommended).  
> Windows users may need minor shell/command adjustments.

---

## 📋 Prerequisites

- Git
- Python 3.10+
- pip
- Conda/Miniconda (optional, recommended for reproducibility)

---

## 🚀 Getting Started
### Linux
Follow these steps to set up and run the repository.

### 1) Clone the repository

```bash
git clone https://github.com/nclabteam/FedA2L.git
```



### 2) Create and activate environment
If needed (CUDA 11.8 build of PyTorch), install first:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```bash
python -m venv dfl
source ./dfl/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
### 3) Configure experiment

Set experiment arguments through `options.py` (CLI flags), including:
- Dataset and data partitioning
- Topology and decentralized framework
- Local training hyperparameters
- FedA2L hyperparameters

### 4) Run training

```bash
# Vanilla DFL
python main.py --dataset=cifar100 --num_nodes=10 --times=3 --learning_rate=0.01 --framework=DFedAvg --name=DFedAvg-Vanilla_cifar100_ResNet18 --iterations=500 --model=ResNet18

# Scheduler-based strategy
python main.py --dataset=cifar100 --num_nodes=10 --times=3 --learning_rate=0.01 --framework=DFedAvg_Scheduler --name=DFedAvg-OCLR_cifar100_ResNet18 --iterations=500 --model=ResNet18 --scheduler=OCLR

# FedA2L
python main.py --dataset=cifar100 --num_nodes=10 --times=3 --learning_rate=0.01 --framework=DFedAvg_FedA2L --name=DFedAvg-FedA2L_cifar100_ResNet18 --iterations=500 --model=ResNet18 --swin=10 --start_tunning=40 --kt=0.1
```

---

## 🔧 Configuration Guide (`options.py`)

### Core arguments

| Argument                      | Type    | Description                                 | Typical value(s)                                 |
| ----------------------------- | ------- | ------------------------------------------- | ------------------------------------------------ |
| `--dataset`                   | `str`   | Dataset name                                | `cifar10`, `cifar100`, `tinyimagenet`            |
| `--iid`                       | `bool`  | Use IID split (`True`) or non-IID (`False`) | `True` / `False`                                 |
| `--alpha`                     | `float` | Dirichlet concentration for non-IID split   | `0.1`, `0.3`, `1.0`                              |
| `--model`                     | `str`   | Model architecture                          | `CNN`, `ResNet18`, `ResNet34`                    |
| `--num_nodes` / `--num_nodes` | `int`   | Number of participating nodes               | `10`, `20`, `50`                                 |
| `--iterations`                | `int`   | Total communication rounds                  | `300`, `500`, `1000`                             |
| `--epochs`                    | `int`   | Local epochs per round                      | `1`, `2`, `5`                                    |
| `--batch_size`                | `int`   | Local mini-batch size                       | `32`, `64`, `128`                                |
| `--learning_rate`             | `float` | Initial learning rate                       | `0.01`, `0.001`                                  |
| `--topology`                  | `str`   | Client connectivity topology                | `FullyConnected`, `ring`, `KConnected`           |
| `--framework`                 | `str`   | Training framework/strategy                 | `DFedAvg`, `DFedAvg_Scheduler`, `DFedAvg_FedA2L` |
| `--times`                     | `int`   | Number of repeated runs                     | `1`, `3`, `5`                                    |
| `--name`                      | `str`   | Experiment name for logs/output             | any string                                       |

### FedA2L arguments

| Argument          | Notation            | Type    | Description                                                  | Tested range / default                            |
| ----------------- | ------------------- | ------- | ------------------------------------------------------------ | ------------------------------------------------- |
| `--start_tunning` | \(R_{\text{warm}}\) | `int`   | Warm-up rounds before adaptation starts                      | `{10, 20, 40}` (CNN: `10`, ResNet: `40`)          |
| `--swin`          | \(\rho\)            | `int`   | Temporal window size for divergence statistics               | `{5, 10, 20}` (CNN: `5`, ResNet: `10`)            |
| `--tau`           | \(\tau\)            | `float` | Stability threshold for LR adjustment                        | `{1e-2, 1e-3, 1e-4}` (default: `1e-3`)            |
| `--beta`          | \(\beta\)           | `float` | Balance between local update intensity and network consensus | `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}` (default: `0.6`) |
| `--kt`            | \(\xi\)             | `float` | Global decay constant for adaptation strength                | `{0.05, 0.1, 0.3}` (CNN: `0.3`, ResNet: `0.1`)    |

† For TinyImageNet, \(\xi = 0.05\) is used due to higher class diversity.

---

## 📊 Evaluation Suggestions

To validate FedA2L, compare against decentralized baselines (e.g., decentralized FedAvg variants) on:

- **Convergence speed** (rounds/time to target accuracy)
- **Highest test accuracy**
- **Training stability** (variance across runs)
- **Communication efficiency** (accuracy vs. communication budget)
- **Robustness to heterogeneity** (varying non-IID severity, network scale, sparse topologies)

---

## 🛠 Code Formatting

```bash
bash code_formatting.sh
```
---

## 📬 Contact

For questions, suggestions, or collaboration, please open an issue in this repository.