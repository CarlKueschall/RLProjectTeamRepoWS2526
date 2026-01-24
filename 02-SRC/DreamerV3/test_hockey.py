"""
DreamerV3 Evaluation Script for Hockey.

AI Usage Declaration:
This file was developed with assistance from Claude Code.

Usage:
    # Evaluate against weak opponent
    python test_hockey.py --checkpoint results/checkpoints/model_5k.pth --opponent weak

    # Evaluate against strong opponent with video
    python test_hockey.py --checkpoint results/checkpoints/model_5k.pth --opponent strong --record --episodes 20

    # Evaluate against another checkpoint
    python test_hockey.py --checkpoint results/checkpoints/model_A.pth \
        --opponent_checkpoint results/checkpoints/model_B.pth --record

    # Custom output path and more episodes
    python test_hockey.py --checkpoint results/checkpoints/model_5k.pth \
        --opponent weak --record --output eval_video.mp4 --episodes 50
"""

import argparse
import os
import sys

import numpy as np
import torch
import hockey.hockey_env as h_env
from hockey.hockey_env import Mode

from dreamer import Dreamer
from utils import loadConfig, seedEverything


def get_device(device_str=None):
    """Get best available device."""
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DreamerOpponent:
    """Wraps a Dreamer agent to act as an opponent."""

    def __init__(self, agent):
        self.agent = agent
        self.h = None
        self.z = None
        self.name = "dreamer_checkpoint"

    def act(self, obs):
        action, self.h, self.z = self.agent.act(obs, self.h, self.z)
        return action

    def reset(self):
        self.h = None
        self.z = None


def run_eval_episode(env, agent, opponent, render=False):
    """
    Run a single evaluation episode.

    Returns:
        total_reward: Episode reward
        steps: Number of steps
        outcome: 'win', 'loss', or 'draw'
        frames: List of RGB frames if render=True, else empty
    """
    obs, _ = env.reset()
    h, z = None, None
    total_reward = 0.0
    steps = 0
    frames = []
    done = False

    while not done:
        if render:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)

        action, h, z = agent.act(obs, h, z)
        obs_opponent = env.obs_agent_two()
        action_opponent = opponent.act(obs_opponent)
        next_obs, reward, done, truncated, info = env.step(np.hstack([action, action_opponent]))
        done = done or truncated
        total_reward += reward
        obs = next_obs
        steps += 1

    # Capture final frame
    if render:
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)

    if info.get('winner', 0) == 1:
        outcome = 'win'
    elif info.get('winner', 0) == -1:
        outcome = 'loss'
    else:
        outcome = 'draw'

    return total_reward, steps, outcome, frames


def save_mp4(all_frames, output_path, fps=30):
    """
    Save list of frame lists as a single MP4 video.

    Episodes are concatenated sequentially with a brief separator.
    """
    try:
        import imageio
    except ImportError:
        print("ERROR: imageio is required for video recording. Install with: pip install imageio[ffmpeg]")
        return False

    if not all_frames:
        print("No frames to save.")
        return False

    # Flatten all episodes into one continuous video with separator frames
    video_frames = []
    for ep_idx, frames in enumerate(all_frames):
        if not frames:
            continue
        video_frames.extend(frames)
        # Add 15 dark separator frames between episodes (0.5s at 30fps)
        if ep_idx < len(all_frames) - 1:
            separator = np.zeros_like(frames[0])
            separator[:] = 30  # Dark gray instead of pure black
            for _ in range(15):
                video_frames.append(separator)

    if not video_frames:
        print("No frames collected.")
        return False

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    try:
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                    macro_block_size=1, pixelformat='yuv420p')
        for frame in video_frames:
            writer.append_data(frame)
        writer.close()
        print(f"Video saved: {output_path} ({len(video_frames)} frames, {len(video_frames)/fps:.1f}s)")
        return True
    except Exception as e:
        print(f"Error saving video: {e}")
        # Fallback: try without codec specification
        try:
            imageio.mimsave(output_path, video_frames, fps=fps)
            print(f"Video saved (fallback): {output_path}")
            return True
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return False


def create_agent(checkpoint_path, config, device):
    """Create and load a Dreamer agent from a checkpoint."""
    env = h_env.HockeyEnv(mode=Mode.NORMAL, keep_mode=True)
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0] // 2
    action_low = env.action_space.low[:action_size].tolist()
    action_high = env.action_space.high[:action_size].tolist()
    env.close()

    agent = Dreamer(observation_size, action_size, action_low, action_high, device, config.dreamer)
    agent.loadCheckpoint(checkpoint_path)
    return agent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DreamerV3 Hockey Agent")

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to agent checkpoint (.pth)")

    # Opponent selection (mutually exclusive)
    opponent_group = parser.add_mutually_exclusive_group(required=True)
    opponent_group.add_argument("--opponent", type=str, choices=["weak", "strong"],
                                help="Fixed opponent type")
    opponent_group.add_argument("--opponent_checkpoint", type=str,
                                help="Path to opponent checkpoint (.pth)")

    # Evaluation settings
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Recording
    parser.add_argument("--record", action="store_true",
                        help="Record episodes as MP4 video")
    parser.add_argument("--record_episodes", type=int, default=None,
                        help="Number of episodes to record (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: auto-generated)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video FPS (default: 30)")

    # Technical
    parser.add_argument("--config", type=str, default="hockey.yml",
                        help="Config file (default: hockey.yml)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/mps/cpu, default: auto)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-episode results")

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if args.opponent_checkpoint and not os.path.exists(args.opponent_checkpoint):
        print(f"ERROR: Opponent checkpoint not found: {args.opponent_checkpoint}")
        sys.exit(1)

    # Load config and setup
    config = loadConfig(args.config)
    device = get_device(args.device)
    seedEverything(args.seed)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")

    # Create environment
    env = h_env.HockeyEnv(mode=Mode.NORMAL, keep_mode=True)
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0] // 2
    action_low = env.action_space.low[:action_size].tolist()
    action_high = env.action_space.high[:action_size].tolist()

    # Create and load agent
    agent = Dreamer(observation_size, action_size, action_low, action_high, device, config.dreamer)
    try:
        agent.loadCheckpoint(args.checkpoint)
    except RuntimeError as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        print("This likely means the checkpoint was saved with a different architecture version.")
        print("Ensure the checkpoint matches the current config (e.g., Two-Hot Symlog reward model).")
        env.close()
        sys.exit(1)
    print(f"Agent loaded (trained {agent.totalGradientSteps} gradient steps, {agent.totalEpisodes} episodes)")

    # Create opponent
    if args.opponent_checkpoint:
        opponent_agent = Dreamer(observation_size, action_size, action_low, action_high, device, config.dreamer)
        try:
            opponent_agent.loadCheckpoint(args.opponent_checkpoint)
        except RuntimeError as e:
            print(f"ERROR: Failed to load opponent checkpoint: {e}")
            env.close()
            sys.exit(1)
        opponent = DreamerOpponent(opponent_agent)
        opponent_name = os.path.basename(args.opponent_checkpoint)
        print(f"Opponent: checkpoint ({opponent_name})")
    else:
        from opponents import FixedOpponent
        opponent = FixedOpponent(weak=(args.opponent == "weak"))
        opponent_name = args.opponent
        print(f"Opponent: {opponent_name}")

    # Determine which episodes to record
    record_episodes = set()
    if args.record:
        n_record = args.record_episodes if args.record_episodes else args.episodes
        n_record = min(n_record, args.episodes)
        record_episodes = set(range(n_record))
        print(f"Recording {n_record} episodes")

    # Run evaluation
    print(f"\nEvaluating {args.episodes} episodes vs {opponent_name}...")
    print("-" * 60)

    rewards = []
    outcomes = {'win': 0, 'loss': 0, 'draw': 0}
    steps_list = []
    all_recorded_frames = []

    for ep in range(args.episodes):
        opponent.reset()
        render = ep in record_episodes
        reward, steps, outcome, frames = run_eval_episode(env, agent, opponent, render=render)

        rewards.append(reward)
        outcomes[outcome] += 1
        steps_list.append(steps)

        if render and frames:
            all_recorded_frames.append(frames)

        if args.verbose:
            print(f"  Episode {ep+1:3d}: {outcome:4s} | reward={reward:+6.2f} | steps={steps:3d}")

        # Print progress every 10% for non-verbose mode
        if not args.verbose and (ep + 1) % max(1, args.episodes // 10) == 0:
            wr = outcomes['win'] / (ep + 1)
            print(f"  Progress: {ep+1}/{args.episodes} | Win rate: {wr:.1%}")

    # Results summary
    win_rate = outcomes['win'] / args.episodes
    loss_rate = outcomes['loss'] / args.episodes
    draw_rate = outcomes['draw'] / args.episodes
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_steps = np.mean(steps_list)

    print("-" * 60)
    print(f"\nRESULTS ({args.episodes} episodes vs {opponent_name}):")
    print(f"  Win rate:  {win_rate:.1%} ({outcomes['win']} wins)")
    print(f"  Loss rate: {loss_rate:.1%} ({outcomes['loss']} losses)")
    print(f"  Draw rate: {draw_rate:.1%} ({outcomes['draw']} draws)")
    print(f"  Reward:    {mean_reward:+.3f} +/- {std_reward:.3f}")
    print(f"  Avg steps: {mean_steps:.1f}")
    print(f"  Checkpoint: {agent.totalGradientSteps} grad steps, {agent.totalEpisodes} episodes")

    # Save video if recorded
    if args.record and all_recorded_frames:
        if args.output:
            output_path = args.output
        else:
            # Auto-generate output path
            ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
            output_path = f"results/videos/eval_{ckpt_name}_vs_{opponent_name}.mp4"

        save_mp4(all_recorded_frames, output_path, fps=args.fps)

    env.close()
    return win_rate


if __name__ == "__main__":
    main()
