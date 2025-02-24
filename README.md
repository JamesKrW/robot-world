# Robot World

A world model training framework for robotics.

## Overview

Robot-World predicts future frame conditioned on previous frames and actions, the architeture of which is based on Open OASIS, where:

- A VAE encodes video frames into a latent space representation
- DiT predicts future frames based on the encoded representations

### Key Features

- Flexible dataset integration through OCTO-based data loading
- Support for multiple robotics datasets in RLDS format
- Modular training pipeline for VAE and DiT components
- Easy extension to new datasets and architectures

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

### Dataset Preparation

1. Download the DROID dataset following instructions at [DROID Dataset](https://droid-dataset.github.io/)
2. Configure your dataset directory in the training scripts

### VAE Training

```bash
python robot-world/scripts/train_vae.py
```

#### Configuration Tips:
- Start with `droid_100` subset for initial testing
- For full DROID dataset, it costs ~30 minutes for initialization on an AMD Ryzen 9 7950X machine
- Adjust threading parameters for faster data loading:
  ```python
  parallel_calls: int = 16
  traj_transform_threads: int = 16
  traj_read_threads: int = 16
  ```

### DIT TRAINING (WIP)

```bash
python robot-world/scripts/train_dit.py
```
You can either freeze VAE weights or jointly train both.

## To-dos
- DiT training pipeline adaptation to OCTO data loading
- Multi-dataset training support
- Performance optimization
- Comprehensive evaluation metrics

## Acknowledgments

This project builds upon several excellent works:
- [Open OASIS](https://github.com/etched-ai/open-oasis)
- [DROID Dataset](https://droid-dataset.github.io/)
- [OCTO](https://octo-models.github.io/)
- [OXE](https://robotics-transformer-x.github.io/)
