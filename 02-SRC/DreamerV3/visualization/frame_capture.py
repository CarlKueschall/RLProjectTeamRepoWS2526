"""
AI Usage Declaration:
This file was developed with assistance from AI autocomplete features in Cursor AI IDE.
"""

import numpy as np
def record_episode_frames(env, agent, opponent, mode, max_timesteps, eps=0.0,
                       self_play_opponent=None):
    frames = []
    obs, info = env.reset()  # Use environment's default seeding (matches evaluation)
    obs_agent2 = env.obs_agent_two()
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
            # Use self-play opponent (recorded GIFs during self-play)
            action2 = self_play_opponent.act(obs_agent2, eps=0.0)
        else:
            # Use fixed opponent (weak/strong)
            action2 = opponent.act(obs_agent2)

        # primary agent might have 8 outputs, but we only want its P1 actions
        # By slicing both to [:4], we ensure proper player separation
        action_combined = np.hstack([action1[:4], action2[:4]])
        # Step
        obs, r1, done, truncated, info = env.step(action_combined)
        obs_agent2 = env.obs_agent_two()
        # Note: env.obs_agent_two() already returns properly mirrored observations
        # No angle transformation needed - the environment handles this internally

        if done or truncated:
            winner = info.get('winner', 0)
            break
    return frames, winner
