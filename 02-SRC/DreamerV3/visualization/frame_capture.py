"""
Frame capture utilities for DreamerV3 GIF recording.

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import numpy as np


def record_episode_frames_dreamer(env, agent, max_timesteps=250):
    """
    Record frames from an episode using DreamerV3 agent interface.

    Args:
        env: HockeyEnvDreamer environment (handles opponent internally)
        agent: HockeyDreamer agent with act(obs, deterministic) interface
        max_timesteps: Maximum timesteps per episode

    Returns:
        frames: List of RGB frames
        winner: Episode outcome (1=win, -1=loss, 0=draw)
    """
    frames = []
    obs, info = env.reset()
    agent.reset()
    winner = 0

    for t in range(max_timesteps):
        # Render frame
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)

        # Get action from DreamerV3 agent (deterministic for evaluation)
        action = agent.act(obs, deterministic=True)

        # Step environment (opponent action handled by env wrapper)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            winner = info.get('winner', 0)
            break

    return frames, winner


def record_episode_frames(env, agent, opponent, mode, max_timesteps, eps=0.0,
                          self_play_opponent=None):
    """
    Record frames from an episode (TD3 interface for backwards compatibility).

    Args:
        env: Hockey environment
        agent: TD3 agent with act(obs, eps) interface
        opponent: Opponent agent
        mode: Game mode
        max_timesteps: Maximum timesteps per episode
        eps: Exploration epsilon (0 for deterministic)
        self_play_opponent: Self-play opponent if in self-play mode

    Returns:
        frames: List of RGB frames
        winner: Episode outcome (1=win, -1=loss, 0=draw)
    """
    frames = []
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()

    # Reset agent if it has a reset method
    if hasattr(agent, 'reset'):
        agent.reset()

    winner = 0

    for t in range(max_timesteps):
        # Render frame
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)

        # Agent actions
        action1 = agent.act(obs, eps=eps)

        # Opponent action: use self-play opponent if provided, otherwise use fixed opponent
        if self_play_opponent is not None:
            action2 = self_play_opponent.act(obs_agent2, eps=0.0)
        else:
            action2 = opponent.act(obs_agent2)

        # Combine actions (slice to 4 dims each)
        action_combined = np.hstack([action1[:4], action2[:4]])

        # Step
        obs, r1, done, truncated, info = env.step(action_combined)
        obs_agent2 = env.obs_agent_two()

        if done or truncated:
            winner = info.get('winner', 0)
            break

    return frames, winner
