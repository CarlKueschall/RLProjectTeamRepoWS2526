"""
This file was developed with assistance from AI: autocomplete and discussion
about the contents and behavior of the code.
"""

import numpy as np


def record_episode_frames_dreamer(env, agent, max_timesteps=250):
    """Record frames from episode. DreamerV3 interface (env handles opponent)."""
    frames = []
    obs, info = env.reset()
    agent.reset()
    winner = 0

    for t in range(max_timesteps):
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)

        action = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            winner = info.get('winner', 0)
            break

    return frames, winner


def record_episode_frames(env, agent, opponent, mode, max_timesteps, eps=0.0,
                          self_play_opponent=None):
    """Record frames. TD3 interface - agent + opponent passed separately."""
    frames = []
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()

    if hasattr(agent, 'reset'):
        agent.reset()

    winner = 0

    for t in range(max_timesteps):
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)

        action1 = agent.act(obs, eps=eps)

        if self_play_opponent is not None:
            action2 = self_play_opponent.act(obs_agent2, eps=0.0)
        else:
            action2 = opponent.act(obs_agent2)

        action_combined = np.hstack([action1[:4], action2[:4]])
        obs, r1, done, truncated, info = env.step(action_combined)
        obs_agent2 = env.obs_agent_two()

        if done or truncated:
            winner = info.get('winner', 0)
            break

    return frames, winner
