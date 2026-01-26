# DreamerV3 Benchmark: 266k Checkpoint (Best Combined Performance)

## Summary

| Metric                      | Value           |
| --------------------------- | --------------- |
| **Combined Win Rate** | **88.5%** |
| vs Weak Win Rate            | 87.0%           |
| vs Strong Win Rate          | 90.0%           |
| Gradient Steps              | 266,100         |
| Training Episodes           | 11,995          |
| Seed                        | 43              |

## Benchmark Results (100 episodes each)

### vs Weak Opponent

| Metric      | Value           |
| ----------- | --------------- |
| Win Rate    | 87.0% (87 wins) |
| Loss Rate   | 6.0% (6 losses) |
| Draw Rate   | 7.0% (7 draws)  |
| Mean Reward | +7.24 ± 5.45   |
| Avg Steps   | 103.2 ± 69.2   |

### vs Strong Opponent

| Metric      | Value           |
| ----------- | --------------- |
| Win Rate    | 90.0% (90 wins) |
| Loss Rate   | 5.0% (5 losses) |
| Draw Rate   | 5.0% (5 draws)  |
| Mean Reward | +7.39 ± 5.44   |
| Avg Steps   | 86.5 ± 63.8    |

## Training Configuration

```yaml
Algorithm: DreamerV3
Seed: 43
Opponent: Mixed (70% weak, 30% strong during fine-tuning)

# Architecture
recurrent_size: 256
latent_length: 16
latent_classes: 16
encoded_obs_size: 256

# Training
batch_size: 32
batch_length: 32
imagination_horizon: 15
discount: 0.997
entropy_scale: 0.0003

# Fine-tuning LRs (reduced from defaults)
lr_world: 0.00005
lr_actor: 0.00002
lr_critic: 0.00002
```

## W&B Run

- **Run ID**: `qmf432nx`
- **Project**: `rl-hockey`
- **Run Name**: `balanced-finetune-DreamerV3-final-finetune-seed43_weak_seed43_20260125_215906`
- **Link**: https://wandb.ai/rl-hockey/qmf432nx

## Files in this Benchmark

```
266k_best_combined/
├── README.md                    # This file
├── checkpoint_266k.pth          # Model checkpoint
├── wandb_training_log.txt       # Training metrics from W&B
├── videos/
│   ├── vs_weak.mp4              # 5 episodes vs weak opponent
│   └── vs_strong.mp4            # 5 episodes vs strong opponent
└── plots/
    ├── evaluation_overview.png  # Main performance visualization
    ├── performance_heatmap.png  # Comparison heatmap
    └── top_checkpoints.png      # Top 5 rankings
```

## Training History

This checkpoint was produced through:

1. **Initial Training** (seed 42): Mixed opponent training with self-play
2. **Fine-tuning** (seed 43): Balanced fine-tuning focusing on weak opponent (70% weak, 30% strong)
3. **Checkpoint Selection**: Selected at 266k gradient steps as best combined performer

## Usage

```python
from dreamer import Dreamer
from utils import loadConfig

config = loadConfig("hockey.yml")
agent = Dreamer(obs_size, action_size, action_low, action_high, device, config.dreamer)
agent.loadCheckpoint("checkpoint_266k.pth")

# Use in environment
action, h, z = agent.act(observation, h, z)
```

## Competition Deployment

To deploy this checkpoint in the COMPRL tournament:

```bash
cd 02-SRC/comprl-hockey-agent
export DREAMER_CHECKPOINT="/path/to/checkpoint_266k.pth"
python run_client.py --server-url <URL> --server-port <PORT> --token <TOKEN>
```

## Notes

- This checkpoint achieved the **best combined performance** across all tested checkpoints
- Outperformed both specialized weak-only and strong-only trained models
- The fine-tuning strategy of 70% weak / 30% strong helped maintain strong opponent performance while improving weak opponent handling
- Draw rate is low (5-7%) indicating decisive gameplay

## Benchmark Date

2026-01-26
