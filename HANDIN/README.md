# RL Hockey Project — Submission Package

**Algorithm:** DreamerV3 (world-model based RL)  
**Author:** Carl Kueschall

This package contains the source code for training and running the DreamerV3 hockey agent, ready for evaluation by examiners.

---

## Structure

```
HANDIN/
├── DreamerV3/           # Training and evaluation code
│   ├── train_hockey.py  # Main training script
│   ├── test_hockey.py  # Local evaluation
│   ├── configs/        # Configuration
│   ├── opponents/      # Self-play, PFSP, fixed opponents
│   └── scripts/        # Evaluation and analysis tools
├── comprl-hockey-agent/ # Tournament client
│   ├── run_client.py   # Connects to competition server
│   ├── best_selfplay_336k.pth  # Pre-trained checkpoint (93.5% vs weak+strong)
│   └── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- PyTorch
- hockey-env, comprl (see comprl-hockey-agent/requirements.txt)

---

## Quick Start

### 1. Train from scratch (optional)

```bash
cd DreamerV3
conda activate py310  # or your env with PyTorch
pip install hockey gymnasium pyyaml wandb  # DreamerV3 deps

# Install deps (optional, if not already installed)
pip install -r requirements.txt

# Quick test run (no W&B)
python train_hockey.py --opponent weak --no_wandb --gradient_steps 10000

# Full training
python train_hockey.py --opponent weak --seed 42 --use_dreamsmooth
```

Checkpoints are saved to `DreamerV3/results/checkpoints/`.

### 2. Run tournament client

The client uses the included `best_selfplay_336k.pth` checkpoint by default.

```bash
cd comprl-hockey-agent
pip install -r requirements.txt

# With environment variables
export COMPRL_SERVER_URL=comprl.cs.uni-tuebingen.de
export COMPRL_SERVER_PORT=65335
export COMPRL_ACCESS_TOKEN=<YOUR_TOKEN>
python run_client.py --args --agent=strong

# Or with command-line args
python run_client.py --server-url comprl.cs.uni-tuebingen.de \
  --server-port 65335 --token <TOKEN> --args --agent=strong
```

To use a different checkpoint (e.g. from training):

```bash
export DREAMER_CHECKPOINT=/path/to/your/checkpoint.pth
python run_client.py --args --agent=strong
```

### 3. Local evaluation

```bash
cd DreamerV3
python test_hockey.py --checkpoint ../comprl-hockey-agent/best_selfplay_336k.pth \
  --opponent weak --episodes 100
```

---

## Benchmark (best_selfplay_336k.pth)

| Opponent | Win Rate |
|----------|----------|
| Weak Bot | 90% |
| Strong Bot | 97% |
| **Combined** | **93.5%** |

*100 episodes each, seed 42.*

---

## AI Usage Declaration

- **Claude Code:** Discussion, decision support, concept learning, implementation assistance (agent internals, training loop, architecture).
- **AI autocomplete:** Core logic implementation, followed by manual review.
- **Claude Code:** Repetitive scripting, formatting, pipeline glue.
