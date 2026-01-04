import hockey.hockey_env as h_env
from hockey.hockey_env import BasicOpponent
def evaluate_vs_opponent(agent, opponent, mode, num_episodes=100, max_timesteps=250, eval_seed=None):
    import numpy as np
    eval_env = h_env.HockeyEnv(mode=mode, keep_mode=True)
    wins = 0
    losses = 0
    ties = 0
    total_reward = 0
    # running the loop deterministically for the given number of episodes with a specified seed.
    ###########################################################
    for i in range(num_episodes):
        # Use seeded resets for reproducibility and diversity
        if eval_seed is not None:
            np.random.seed(eval_seed + i)
            reset_seed = np.random.randint(0, 1000000)
        else:
            reset_seed = None

        obs, info = eval_env.reset(seed=reset_seed)
        episode_reward = 0
        episode_ended = False

        for t in range(max_timesteps):
            action1 = agent.act(obs, eps=0.0)  # Deterministic evaluation
            obs_agent2 = eval_env.obs_agent_two()
            # FIX (2026-01-03): Mirror angles for P2
            obs_agent2[2] = np.arctan2(-np.sin(obs_agent2[2]), -np.cos(obs_agent2[2]))
            obs_agent2[8] = np.arctan2(-np.sin(obs_agent2[8]), -np.cos(obs_agent2[8]))
            action2 = opponent.act(obs_agent2)

            # CRITICAL FIX (2026-01-03): Action Slicing
            obs, r1, done, truncated, info = eval_env.step(np.hstack([action1[:4], action2[:4]]))
            episode_reward += r1

            if done or truncated:
                winner = info.get('winner', 0)
                if winner == 1:
                    wins += 1
                elif winner == -1:
                    losses += 1
                else:
                    ties += 1
                episode_ended = True
                break
        if not episode_ended:
            # Check if info has winner field from last step
            winner = info.get('winner', 0)
            if winner == 1:
                wins += 1
            elif winner == -1:
                losses += 1
            else:
                # No clear winner after max_timesteps = tie/draw
                ties += 1


        total_reward += episode_reward

    # close the evaluation environment
    eval_env.close()    


    total = wins + losses + ties
    decisive_games = wins + losses
    #########################################################
    # return the metrics for wandb logging
    #########################################################


    return {
        'win_rate': wins / total if total > 0 else 0,  # All games (includes ties)
        'win_rate_decisive': wins / decisive_games if decisive_games > 0 else 0,  # Only decisive games
        'loss_rate': losses / total if total > 0 else 0,
        'tie_rate': ties / total if total > 0 else 0,
        'avg_reward': total_reward / num_episodes,
        'wins': wins,
        'losses': losses,
        'ties': ties,
    }
