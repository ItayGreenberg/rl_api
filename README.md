# rl_api

`rl_api` is a modular reinforcement learning library for general RL. It is designed around clean separation between agents, environments, networks, schedulers, and logging, so you can iterate quickly while keeping code organized.

## Features

- PPO agent (implemented)
- PPG agent (implemented)
- Training APIs:
  - `train(train_cfg)` - full training loop driven by a `TrainConfig`
  - `train_one_update()` - runs one complete update (collect rollouts, update, logging, etc.)
- Abstract environment interface (implement your own env and plug it in)
- Vectorized environments and wrappers
- Logging: console, TensorBoard, HTML

## Roadmap

Planned additions:
- DQN
- QR-DQN
- Rainbow-style components

## Dependencies

Required:
- numpy
- torch

## Installation

From the repository root:

```bash
pip install -e .
