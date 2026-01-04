# RL Hockey Project

Authors: Serhat Alpay, Carl Kueschall

Reinforcement Learning project for training agents to play hockey using TD3 algorithm with self-play.

## Quick Start

```bash
cd 02-SRC/TD3

python train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 25000 \
    --seed 46 \
    --eps 1.0 \
    --eps_min 0.05 \
    --eps_decay 0.99985 \
    --warmup_episodes 500 \
    --batch_size 512 \
    --train_freq 10 \
    --lr_actor 0.0003 \
    --lr_critic 0.0003 \
    --gamma 0.99 \
    --tau 0.005 \
    --policy_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --grad_clip 1.0 \
    --buffer_size 500000 \
    --reward_shaping \
    --q_clip 25.0 \
    --hidden_actor 400 400 \
    --hidden_critic 400 400 200 \
    --log_interval 100 \
    --save_interval 500 \
    --gif_interval 1000 \
    --eval_interval 1000 \
    --self_play_start 8000 \
    --self_play_pool_size 10 \
    --self_play_save_interval 500 \
    --self_play_weak_ratio 0.4 \
    --use_dual_buffers \
    --use_pfsp \
    --pfsp_mode variance \
    --dynamic_anchor_mixing \
    --performance_gated_selfplay \
    --selfplay_gate_winrate 0.80 \
    --selfplay_gate_variance 0.3 \
    --regression_rollback \
    --regression_threshold 0.15
```

## Testing

```bash
cd 02-SRC/TD3

python test_hockey.py \
    --checkpoint results/checkpoints/TD3_Hockey_NORMAL_weak_final_seed42.pth \
    --opponent weak \
    --episodes 100 \
    --render human
```

## Key Features

- **TD3 Algorithm**: Twin Delayed DDPG with target smoothing and Q-value clipping
- **Self-Play**: Automatic opponent pool management with PFSP selection
- **Anti-Forgetting**: Dual replay buffers and dynamic anchor mixing
- **Reward Shaping**: PBRS (Potential-Based Reward Shaping) and strategic bonuses
- **Performance Gating**: Self-play activates only when agent reaches competence threshold

## File Structure

```
02-SRC/TD3/
├── train_hockey.py          # Main training script
├── test_hockey.py           # Testing script
├── config/
│   └── parser.py            # Command line arguments
├── agents/
│   ├── td3_agent.py         # TD3 agent
│   ├── ddpg_agent.py        # DDPG agent (alternative)
│   ├── model.py             # Neural network models
│   ├── memory.py            # Replay buffer
│   └── noise.py             # OU noise for exploration
├── rewards/
│   ├── pbrs.py              # Potential-Based Reward Shaping
│   └── strategic_rewards.py # Strategic bonuses (opponent-aware, diversity)
├── opponents/
│   ├── self_play.py         # Self-play manager
│   └── pfsp.py              # PFSP opponent selection
├── evaluation/
│   └── evaluator.py         # Evaluation functions
├── metrics/
│   └── metrics_tracker.py   # Metrics tracking
└── visualization/
    └── gif_recorder.py      # GIF recording for W&B
```

## Main Components

### TD3 Agent - Carl Kueschall

- Twin critics to reduce overestimation bias
- Delayed policy updates (every 2 steps)
- Target smoothing with clipped noise
- Q-value clipping to prevent explosion

### SAC Agent - Serhat Alpay

#TODO

### Reward Shaping

- **PBRS**: Potential-based reward shaping that preserves optimal policy
- **Strategic Rewards**: Opponent-aware shooting, attack diversity, forcing bonuses

### Self-Play

- Opponent pool management with automatic curriculum
- PFSP (Prioritized Fictitious Self-Play) for opponent selection
- Dual replay buffers to prevent catastrophic forgetting
- Performance-gated activation (requires 85%+ win rate vs weak)
