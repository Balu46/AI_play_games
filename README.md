# AI Play Games

## Overview
This project implements Reinforcement Learning (RL) agents capable of playing various classic environments from the Gymnasium library. It serves as a playground for comparing custom implementations ("Bare Bone") against industry-standard libraries like Stable Baselines3.

## Features
-   **Multiple Environments**: Support for `CartPole-v1`, `LunarLander-v2`, and `CarRacing-v2`.
-   **Dual Approach**:
    -   `src/bare_bone`: Custom, educational implementations of RL algorithms.
    -   `src/stable_baseline`: Robust implementations using Stable Baselines3 (DQN, PPO, A2C).
-   **Hyperparameter Optimization**: Integrated Optuna support for tuning agent performance.
-   **Visualization**: Tools to record and play back agent performance.

## Installation

### Prerequisites
-   Python 3.8+
-   System dependencies for Box2D and rendering (see `setup.sh`)

### Quick Setup
The project includes a setup script for Debian/Ubuntu based systems:

```bash
./setup.sh
```

This script will:
1.  Install system dependencies (swig, cmake, ffmpeg, etc.).
2.  Create a Python virtual environment (`.venv`).
3.  Install Python dependencies from `requirements.txt`.

### Manual Setup
If you prefer to set it up manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

All scripts assume you are in the project root.

### 1. Training Agents
To train an agent using the configuration specified in `config.json`:

```bash
./run.sh
```
Or manually:
```bash
python src/stable_baseline/main.py --mode train
```

### 2. hyperparameter Optimization
To find the best hyperparameters for a specific algorithm and environment:

```bash
./run_optimize.sh -e lunar_lander -a ppo -n 20
```
**Arguments:**
-   `-e`: Environment (`cart_pole`, `lunar_lander`, `car_racing`, or `all`)
-   `-a`: Algorithm (`dqn`, `ppo`, `a2c`, or `all`)
-   `-n`: Number of trails (default: 10)
-   `-t`: Timesteps per trial (default: 30000)

### 3. Visualization
To watch a trained agent play:

```bash
./run_visualisation.sh ppo lunar_lander
```

## Project Structure

```
├── articles/               # Documentation and articles
├── car_racing/             # Component specific data
├── cart_pole/              # Component specific data
├── lunar_lander/           # Component specific data
├── src/
│   ├── bare_bone/          # Custom RL algorithm implementations
│   └── stable_baseline/    # SB3 based implementations and main entry point
│       ├── main.py         # Main training/inference script
│       ├── optimize.py     # Optuna optimization logic
│       └── algorithms/     # Wrapper classes for SB3 algos
├── run.sh                  # One-click training script
├── run_optimize.sh         # Optimization wrapper script
├── run_visualisation.sh    # Visualization wrapper script
└── config.json             # Configuration for training
```
