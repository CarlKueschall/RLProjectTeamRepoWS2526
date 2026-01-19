"""
Diagnostic test: Run agent as Player 2 to check perspective handling.
If performance drops significantly, there's a perspective issue.
"""
import argparse
import numpy as np
import torch
import hockey.hockey_env as h_env
from hockey.hockey_env import BasicOpponent, Mode
from agents.td3_agent import TD3Agent
from gymnasium import spaces


def test_as_player2(checkpoint_path, hidden_actor, hidden_critic, episodes=50):
    """Test agent as Player 2 (swapped roles from normal)."""

    from agents.device import get_device
    device = get_device()

    print("=" * 60)
    print("DIAGNOSTIC: Testing Agent as PLAYER 2")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {episodes}")
    print()

    # Create environment with keep_mode=True
    env = h_env.HockeyEnv(mode=Mode.NORMAL, keep_mode=True)

    # Create opponent (will be Player 1)
    opponent = BasicOpponent(weak=True)

    # Create agent action space (4D)
    agent_action_space = spaces.Box(
        low=env.action_space.low[:4],
        high=env.action_space.high[:4],
        dtype=env.action_space.dtype
    )

    # Create agent
    agent = TD3Agent(
        env.observation_space,
        agent_action_space,
        force_cpu=(device.type == 'cpu'),
        hidden_sizes_actor=hidden_actor,
        hidden_sizes_critic=hidden_critic,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'agent_state' in checkpoint:
        agent.restore_state(checkpoint['agent_state'])
    else:
        agent.restore_state(checkpoint)

    # Put in eval mode
    agent.policy.eval()

    print("Agent loaded. Running tests...")
    print()

    wins_as_p1 = 0
    wins_as_p2 = 0
    ties_as_p1 = 0
    ties_as_p2 = 0
    losses_as_p1 = 0
    losses_as_p2 = 0

    # Test as Player 1 (normal)
    print("Testing as PLAYER 1 (normal)...")
    for ep in range(episodes):
        obs, info = env.reset()
        obs_p2 = env.obs_agent_two()
        agent.reset()

        for t in range(250):
            # Agent is P1, opponent is P2
            action1 = agent.act(obs, eps=0.0)
            action2 = opponent.act(obs_p2)
            action_combined = np.hstack([action1[:4], action2[:4]])

            obs, r1, done, truncated, info = env.step(action_combined)
            obs_p2 = env.obs_agent_two()

            if done or truncated:
                break

        winner = info.get('winner', 0)
        if winner == 1:
            wins_as_p1 += 1
        elif winner == -1:
            losses_as_p1 += 1
        else:
            ties_as_p1 += 1

    p1_winrate = wins_as_p1 / episodes * 100
    print(f"  As P1: {wins_as_p1} wins, {losses_as_p1} losses, {ties_as_p1} ties ({p1_winrate:.1f}% win rate)")

    # Test as Player 2 (SWAPPED - this simulates tournament when agent is assigned P2)
    print("\nTesting as PLAYER 2 (swapped roles)...")
    for ep in range(episodes):
        obs, info = env.reset()
        obs_p2 = env.obs_agent_two()
        agent.reset()

        for t in range(250):
            # SWAPPED: Opponent is P1, Agent is P2
            # Agent sees P2's observation (mirrored)
            action_opponent = opponent.act(obs)  # Opponent sees P1 obs
            action_agent = agent.act(obs_p2, eps=0.0)  # Agent sees P2 obs (mirrored)

            # Combine: P1 action first, then P2 action
            action_combined = np.hstack([action_opponent[:4], action_agent[:4]])

            obs, r1, done, truncated, info = env.step(action_combined)
            obs_p2 = env.obs_agent_two()

            if done or truncated:
                break

        # Note: winner=1 means P1 won, winner=-1 means P2 won
        # Since agent is P2, winner=-1 is a WIN for our agent
        winner = info.get('winner', 0)
        if winner == -1:  # P2 (our agent) won
            wins_as_p2 += 1
        elif winner == 1:  # P1 (opponent) won, we lost
            losses_as_p2 += 1
        else:
            ties_as_p2 += 1

    p2_winrate = wins_as_p2 / episodes * 100
    print(f"  As P2: {wins_as_p2} wins, {losses_as_p2} losses, {ties_as_p2} ties ({p2_winrate:.1f}% win rate)")

    print()
    print("=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"Win rate as Player 1: {p1_winrate:.1f}%")
    print(f"Win rate as Player 2: {p2_winrate:.1f}%")
    print()

    if p2_winrate < p1_winrate - 20:
        print("⚠️  SIGNIFICANT PERFORMANCE DROP AS PLAYER 2!")
        print("   This indicates a perspective/mirroring issue.")
        print("   The agent may need to handle P2 observations differently.")
    elif p2_winrate < p1_winrate - 10:
        print("⚠️  Moderate performance drop as Player 2.")
        print("   There may be a minor perspective issue.")
    else:
        print("✓ Performance similar for both positions.")
        print("  Perspective handling appears correct.")

    print("=" * 60)

    env.close()
    return p1_winrate, p2_winrate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hidden_actor', type=int, nargs='+', default=[1024, 1024])
    parser.add_argument('--hidden_critic', type=int, nargs='+', default=[1024, 1024, 200])
    parser.add_argument('--episodes', type=int, default=50)
    args = parser.parse_args()

    test_as_player2(args.checkpoint, args.hidden_actor, args.hidden_critic, args.episodes)
