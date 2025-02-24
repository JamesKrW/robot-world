# Robot World

A world model training framework for robotics.

## Overview

Robot-World predicts future frame conditioned on previous frames and actions, the architeture of which is based on Open OASIS.

## Supported Datasets

- DROID: Primary dataset for current development
- Multiple datasets in OXE (WIP)

Thanks to our implementation using OCTO's data loading interface, the framework can easily mix different RLDS-formatted datasets during training.

## Installation

1. Clone the repository:
```bash
git clone git@github.com:JamesKrW/robot-world.git
conda create -n robot-world python=3.11 -y
conda activate robot-world
cd robot-world 
pip install -e .
```

## Usage

### Preparation

1. Download the [DROID](https://droid-dataset.github.io/) dataset following instructions.
2. Set your dataset dir.
3. Set your wandb account.

### VAE Training

```bash
python robot-world/scripts/train_vae.py
```

- Might start with `droid_100`. The full DROID dataset costs ~30 minutes for initialization on an AMD Ryzen 9 7950X machine.
- You can adjust threading parameters for faster data loading:
  ```python
  parallel_calls: int = 16
  traj_transform_threads: int = 16
  traj_read_threads: int = 16
  ```

### DIT TRAINING (WIP)

```bash
python robot-world/scripts/train_dit.py
```
- Could either freeze VAE weights or jointly train both.

## To-dos
- DiT training pipeline adaptation to OCTO data loading
- Multi-dataset training support
- Performance optimization
- Comprehensive evaluation metrics

## Acknowledgments

This project builds upon several excellent works:
- [Open OASIS](https://github.com/etched-ai/open-oasis)
- [DROID](https://droid-dataset.github.io/)
- [OCTO](https://octo-models.github.io/)
- [OXE](https://robotics-transformer-x.github.io/)
