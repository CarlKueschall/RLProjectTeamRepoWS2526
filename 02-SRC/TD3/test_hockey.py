import argparse
import pathlib
import numpy as np
import torch
from tqdm import tqdm
import hockey.hockey_env as h_env
from hockey.hockey_env import BasicOpponent, Mode
from agents.td3_agent import TD3Agent

def parse_args():
    #########################################################
    # Parse command line arguments for testing
    parser = argparse.ArgumentParser(description='Test TD3 agent on Hockey environment')
    # Environment
    parser.add_argument('--mode', type=str, default='NORMAL',
                        choices=['NORMAL', 'TRAIN_SHOOTING', 'TRAIN_DEFENSE'],
                        help='Hockey game mode (default: NORMAL)')
    parser.add_argument('--opponent', type=str, default='weak',
                        choices=['weak', 'strong', 'self'],
                        help='Opponent type (default: weak)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of test episodes (default: 100)')
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (.pth)')
    #########################################################
    # Network architecture (must match the checkpoint!)
    #########################################################
    parser.add_argument('--hidden_actor', type=int, nargs='+', default=[400, 400],
                        help='Hidden layer sizes for actor network (default: [400, 400])')
    parser.add_argument('--hidden_critic', type=int, nargs='+', default=[400, 400, 200],
                        help='Hidden layer sizes for critic network (default: [400, 400, 200])')
    # Evaluation
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic policy (no exploration noise)')
    parser.add_argument('--no_deterministic', dest='deterministic', action='store_false',
                        help='Use stochastic policy (with exploration noise)')






    #########################################################
    # Visualization
    parser.add_argument('--render', type=str, default=None,
                        choices=[None, 'rgb_array', 'human'],
                        help='Render mode: None (no vis), rgb_array (fast, recordable), human (watch game)')
    # Output
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-episode results')
    parser.add_argument('--save_video', type=str, default=None,
                        help='Save video to file (requires --render rgb_array)')
    return parser.parse_args()

def get_mode(mode_str):
    #########################################################
    # Convert string to Mode enum
    if mode_str == 'NORMAL':
        return Mode.NORMAL
    elif mode_str == 'TRAIN_SHOOTING':
        return Mode.TRAIN_SHOOTING
    elif mode_str == 'TRAIN_DEFENSE':
        return Mode.TRAIN_DEFENSE
    else:
        raise ValueError(f"Unknown mode: {mode_str}")


def get_max_timesteps(mode):
    #########################################################
    # Get max timesteps for mode
    if mode == Mode.NORMAL:
        return 250
    else:  # TRAIN_SHOOTING or TRAIN_DEFENSE
        return 80


def prepare_observation(obs):
    #########################################################
    # Prepare observation for agent (no truncation needed)
    #########################################################
    return obs


def load_checkpoint(agent, checkpoint_path: str, device: torch.device):
    #########################################################
    # Load agent state from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    #########################################################
    # Handle multiple checkpoint formats
    #########################################################
    if isinstance(checkpoint, tuple):
        # Format 1: Direct tuple from agent.state()
        agent.restore_state(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    elif isinstance(checkpoint, dict) and 'agent_state' in checkpoint:
        # Format 2: Dict with 'agent_state' key (from training script)
        agent.restore_state(checkpoint['agent_state'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    elif isinstance(checkpoint, dict) and 'policy' in checkpoint:
        #########################################################
        # Format 3: Dict with individual network keys
        #########################################################
        agent.policy.load_state_dict(checkpoint['policy'])
        agent.Q1.load_state_dict(checkpoint['Q1'])
        agent.Q2.load_state_dict(checkpoint['Q2'])
        agent.policy_target.load_state_dict(checkpoint['policy_target'])
        agent.Q1_target.load_state_dict(checkpoint['Q1_target'])
        agent.Q2_target.load_state_dict(checkpoint['Q2_target'])

        #########################################################
        # Load optimizers if available
        #########################################################
        if 'policy_optimizer' in checkpoint:
            agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        if 'Q1_optimizer' in checkpoint:
            agent.Q1_optimizer.load_state_dict(checkpoint['Q1_optimizer'])
        if 'Q2_optimizer' in checkpoint:
            agent.Q2_optimizer.load_state_dict(checkpoint['Q2_optimizer'])

        print(f"Loaded checkpoint from {checkpoint_path}")

        # Print config if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"Mode: {config.get('mode', 'N/A')}, Opponent: {config.get('opponent', 'N/A')}")
            print(f"Config: lr_actor={config.get('learning_rate_actor', 'N/A')}, "
                  f"gamma={config.get('discount', 'N/A')}, "
                  f"policy_freq={config.get('policy_freq', 'N/A')}")
    else:
        # Unrecognized format
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
        else:
            keys = type(checkpoint)
        raise ValueError(f"Unrecognized checkpoint format. Keys: {keys}")


def test(args):
    #########################################################
    # Test TD3 agent on hockey
    #########################################################
    # Set random seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    #########################################################
    # Get device
    #########################################################
    from agents.device import get_device
    device = get_device()

    print("###############################")
    print(f"TD3 Testing: Hockey ({args.mode} mode)")
    print(f"Device: {device.type.upper()}")
    print(f"Opponent: {args.opponent}")
    print(f"Episodes: {args.episodes}")
    print(f"Render: {args.render if args.render else 'None'}")
    print("###############################")

    #########################################################
    # Create environment (hockey doesn't use render_mode in constructor)
    #########################################################
    mode = get_mode(args.mode)
    env = h_env.HockeyEnv(mode=mode, keep_mode=True)

    #########################################################
    # Create opponent
    #########################################################
    if args.opponent == 'self':
        opponent = None  # Will create second agent
    else:
        opponent = BasicOpponent(weak=(args.opponent == 'weak'))

    #########################################################
    # Create agent with architecture matching checkpoint
    #########################################################
    # The checkpoint was trained with obs_dim=18 and agent outputs 4-dim actions
    # The environment takes 8-dim actions (4 per player), but agent only outputs 4
    from gymnasium import spaces
    obs_space = env.observation_space
    # Agent observes full 18-dim observation (not truncated)

    # Create agent action space - only 4 dims (agent's own actions, not environment's 8)
    agent_action_space = spaces.Box(
        low=env.action_space.low[:4],
        high=env.action_space.high[:4],
        dtype=env.action_space.dtype
    )

    agent = TD3Agent(
        obs_space,
        agent_action_space,
        force_cpu=(device.type == 'cpu'),
        hidden_sizes_actor=args.hidden_actor,
        hidden_sizes_critic=args.hidden_critic,
    )

    #########################################################
    # Load checkpoint
    #########################################################
    checkpoint_path = pathlib.Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    load_checkpoint(agent, str(checkpoint_path), device)

    #########################################################
    # Set exploration epsilon for testing
    #########################################################
    test_eps = 0.0 if args.deterministic else 0.1  # no noise if deterministic

    #########################################################
    # For self-play, create second agent (without checkpoint for now)
    #########################################################
    agent2 = None
    agent2_eps = 0.0
    if args.opponent == 'self':  # self-play mode
        agent2 = TD3Agent(
            env.observation_space,
            env.action_space,
            force_cpu=(device.type == 'cpu'),
            hidden_sizes_actor=args.hidden_actor,
            hidden_sizes_critic=args.hidden_critic,
        )
        agent2_eps = 0.0

    #########################################################
    # Testing metrics
    #########################################################
    rewards_p1 = []
    rewards_p2 = []
    wins = 0
    losses_count = 0
    ties = 0
    goals_scored = 0
    goals_conceded = 0

    max_timesteps = get_max_timesteps(mode)
    video_frames = [] if args.save_video else None

    #########################################################
    # Testing loop
    #########################################################
    for episode in tqdm(range(args.episodes), desc="Testing"):
        obs, info = env.reset()
        obs = prepare_observation(obs)
        obs_agent2 = env.obs_agent_two()
        obs_agent2 = prepare_observation(obs_agent2)
        # FIX: Mirror angles for P2
        obs_agent2[2] = np.arctan2(-np.sin(obs_agent2[2]), -np.cos(obs_agent2[2]))
        obs_agent2[8] = np.arctan2(-np.sin(obs_agent2[8]), -np.cos(obs_agent2[8]))
        agent.reset()
        if agent2:
            agent2.reset()  # reset second agent too

        episode_reward_p1 = 0
        episode_reward_p2 = 0
        winner = 0

        for t in range(max_timesteps):
            #########################################################
            # Agent 1 action
            action1 = agent.act(obs, eps=test_eps)

            #########################################################
            # Agent 2 action
            #########################################################
            if args.opponent == 'self':
                action2 = agent2.act(obs_agent2, eps=agent2_eps)
            else:
                action2 = opponent.act(obs_agent2)

            # Ensure only 4-dim actions are passed regardless of agent output dim
            action_combined = np.hstack([action1[:4], action2[:4]])

            # Step environment
            obs, r1, done, truncated, info = env.step(action_combined)
            obs = prepare_observation(obs)

            # Get agent 2's reward
            r2 = env.get_reward_agent_two(info)  # opponent's reward

            episode_reward_p1 += r1
            episode_reward_p2 += r2

            #########################################################
            # Update observations

            obs_agent2 = env.obs_agent_two()
            obs_agent2 = prepare_observation(obs_agent2)
            # FIX: Mirror angles for P2
            obs_agent2[2] = np.arctan2(-np.sin(obs_agent2[2]), -np.cos(obs_agent2[2]))
            obs_agent2[8] = np.arctan2(-np.sin(obs_agent2[8]), -np.cos(obs_agent2[8]))

            ######################################################
            # Render
            #########################################################
            if args.render == 'human':
                env.render(mode='human')
            elif args.render == 'rgb_array' and video_frames is not None:
                frame = env.render(mode='rgb_array')
                video_frames.append(frame)

            if done or truncated:
                winner = info.get('winner', 0)
                break

        #########################################################
        # Track metrics
        #########################################################
        rewards_p1.append(episode_reward_p1)
        rewards_p2.append(episode_reward_p2)

        if winner == 1:
            wins += 1
            goals_scored += 1
        elif winner == -1:
            losses_count += 1
            goals_conceded += 1
        else:
            ties += 1

        if args.verbose:
            if winner == 1:
                result = "WIN"
            elif winner == -1:
                result = "LOSS"
            else:
                result = "TIE"
            print(f"Episode {episode+1:3d}: {result:4s} | P1 Reward: {episode_reward_p1:7.2f} | "
                  f"P2 Reward: {episode_reward_p2:7.2f} | Goals: {goals_scored}-{goals_conceded}")

    #########################################################
    # Calculate statistics
    #########################################################
    total_games = wins + losses_count + ties
    win_rate = wins / total_games if total_games > 0 else 0
    mean_reward_p1 = np.mean(rewards_p1)
    std_reward_p1 = np.std(rewards_p1)
    mean_reward_p2 = np.mean(rewards_p2)

    #########################################################
    # Print results
    #########################################################
    print("###############################")
    print(f"Test Results: Hockey ({args.mode} vs {args.opponent})")
    print("###############################")
    print(f"Episodes played:    {args.episodes}")
    print()
    print(f"Wins:               {wins:4d} ({win_rate:5.1%})")
    print(f"Losses:             {losses_count:4d}")
    print(f"Ties:               {ties:4d}")
    print()
    print(f"Goals scored:       {goals_scored}")
    print(f"Goals conceded:     {goals_conceded}")
    print(f"Goal difference:    {goals_scored - goals_conceded:+d}")
    print()
    print(f"Mean P1 reward:     {mean_reward_p1:7.2f} ± {std_reward_p1:7.2f}")
    print(f"Mean P2 reward:     {mean_reward_p2:7.2f}")
    print("###############################")

    #########################################################
    # Performance assessment
    #########################################################
    if args.mode == 'NORMAL' and args.opponent == 'weak':
        if win_rate >= 0.55:
            print(f"PASS: Win rate ({win_rate:.1%}) exceeds 55% threshold!")
        else:
            print(f"FAIL: Win rate ({win_rate:.1%}) below 55% threshold")
        print()

    #########################################################
    # Save video if requested
    #########################################################
    if video_frames and args.save_video:
        try:
            import imageio
            # Create directory if it doesn't exist
            video_path = pathlib.Path(args.save_video)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(video_path), video_frames, fps=30)
            print(f"Video saved to {args.save_video}")
        except ImportError:
            print("imageio not installed. Install with: pip install imageio")
            print("Video not saved.")

    env.close()

    return {
        'wins': wins,
        'losses': losses_count,
        'ties': ties,
        'win_rate': win_rate,
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'mean_reward_p1': mean_reward_p1,
        'std_reward_p1': std_reward_p1,
        'mean_reward_p2': mean_reward_p2,
    }


if __name__ == '__main__':
    args = parse_args()
    results = test(args)
