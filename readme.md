# 🚀 Representation Learning in RL with lower bound of Bisimulation 

A sophisticated RL implementation with state embedding capabilities for enhanced learning performance. This framework supports both vanilla and energy-based embeddings with comprehensive experiment tracking via Weights & Biases.

## 🌟 Features

- **Dual Embedding Support** 🧠: Vanilla and energy-based state embeddings
- **Real-time Visualization** 📊: Live training metrics and performance plots
- **Hyperparameter Sweeps** 🔍: Automated parameter optimization with W&B
- **Modular Architecture** 🏗️: Clean separation of concerns for easy experimentation

## 📁 Project Structure

| File | Description | Emoji |
|------|-------------|--------|
| `main.py` | Main training script and entry point | 🎯 |
| `target_main.py` | Target network implementation main | 🎯🎯 |
| `base_agent.py` | Core agent class with basic RL functionality | 🤖 |
| `target_base_agent.py` | Base agent with target network support | 🎯🤖 |
| `actor.py` | Policy network implementation | 🎭 |
| `critic.py` | Value function approximator | ⭐ |
| `buffer.py` | Experience replay buffer | 💾 |
| `state_embedding.py` | State embedding base classes | 🧩 |
| `shared.py` | Shared utilities and helper functions | 🔗 |
| `plot.py` | Visualization and plotting utilities | 📈 |
| `argument_parser.py` | Command-line argument configuration | ⚙️ |
| `configure.py` | Configuration management | 🛠️ |
| `sweep_configuration.py` | W&B sweep configuration | 🔍 |

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
./requirements.sh

# Install PyTorch (choose appropriate version for your system)
# Visit: https://pytorch.org/get-started/locally/
```

### 2. Basic Training
```bash
# Train with default parameters
python main.py

# Train with custom embedding and logging
python main.py --embedding energy --use_log True --wandb

# Train with specific batch size and embedding coefficient
python main.py -b 512 -ec 0.6 -e energy --wandb
```

### 3. Advanced Examples

```bash
# Energy embedding with log transformation
python main.py -e energy -ulog True -b 128 -ec 0.8 --wandb

# Custom environment with vanilla embedding
python main.py -env "CartPole-v1" -e vanilla -b 256 --wandb

```

### ⚙️ Configuration Parameters

| Parameter | Flag | Type | Default | Description | Emoji |
|-----------|------|------|---------|-------------|--------|
| Batch Size | `-b`, `--batch_size` | `int` | `256` | Training batch size | 📦 |
| Embedding Loss Coefficient | `-ec`, `--embedding_loss_coeff` | `float` | `0.4` | Weight for embedding loss | ⚖️ |
| Embedding Type | `-e`, `--embedding` | `str` | `"vanilla"` | Type of embedding (`vanilla`/`energy`) | 🧠 |
| Environment | `-env`, `--env` | `str` | `"LunarLander-v2"` | Gym environment name | 🌙 |
| Use Log Transform | `-ulog`, `--use_log` | `bool` | `False` | Log transform for feature difference | 📊 |
| W&B Logging | `--wandb` | `flag` | `False` | Enable W&B experiment tracking | 📈 |
| W&B Entity | `-we`, `--wandb_entity` | `str` | `'da24d008-iit-madras'` | W&B account/team name | 👥 |
| W&B Project | `-wp`, `--wandb_project` | `str` | `'da7400-test'` | W&B project name | 🎯 |
| W&B Sweep | `--wandb_sweep` | `flag` | `False` | Enable parameter sweeping | 🔍 |
| Sweep ID | `--sweep_id` | `str` | `None` | Existing sweep ID to continue | 🔁 |


### 🤝 Contributing
Feel free to fork this repository and submit pull requests for any improvements!

---
### Notes

This is a part of the course work project DA7400 (July-Nov 2025) taught by Prof Balaraman Ravindran in IIT Madras. 

