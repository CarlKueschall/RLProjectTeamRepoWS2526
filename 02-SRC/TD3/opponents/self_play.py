import os
import pathlib
from collections import deque
import numpy as np
import torch
import agents.td3_agent as TD3Agent
from agents.model import Model

class SelfPlayManager:
    ######################################################
    # Manage self-play opponent pool and selection.
    ######################################################
    def __init__(self, pool_size=10, save_interval=500,
                 weak_ratio=0.5, device=None,
                 use_pfsp=False, pfsp_mode="variance",
                 dynamic_anchor_mixing=False,
                 performance_gated=False, gate_winrate=0.90, gate_variance=0.10,
                 regression_rollback=False, regression_threshold=0.15):
        ########################################################
        # Initialize self-play manager.
        #Arguments:
        # pool_size: Number of past checkpoints to keep
        # save_interval: Save interval for adding new opponents
        # weak_ratio: Ratio of episodes to train against weak opponent
        # device: Torch device for loading opponents
        # use_pfsp: Use PFSP opponent selection
        # pfsp_mode: PFSP mode ('variance' or 'hard')
        # dynamic_anchor_mixing: Adjust weak_ratio based on forgetting
        # performance_gated: Gate activation on performance
        # gate_winrate: Min win-rate vs weak to activate
        # gate_variance: Max rolling variance to activate
        # regression_rollback: Enable automatic rollback
        # regression_threshold: Rollback threshold (e.g., 0.15 = 15%)
        self.pool_size = pool_size
        self.save_interval = save_interval
        self.weak_ratio = weak_ratio
        self.current_anchor_ratio = weak_ratio  # For dynamic mixing
        self.device = device

        #########################################################
        # PFSP settings
        self.use_pfsp = use_pfsp
        self.pfsp_mode = pfsp_mode

        #########################################################
        # Dynamic anchor mixing
        self.dynamic_anchor_mixing = dynamic_anchor_mixing

        #########################################################
        # Performance gating
        #########################################################
        self.performance_gated = performance_gated
        self.gate_winrate = gate_winrate
        self.gate_variance = gate_variance

        #########################################################
        # Regression rollback
        
        self.regression_rollback = regression_rollback
        self.regression_threshold = regression_threshold

        #########################################################
        # Self-play state
        self.active = False
        self.start_episode = 0
        self.pool = []  # List of checkpoint paths
        self.opponent = None  # Currently loaded opponent network
        self.opponent_path = None
        self.current_opponent_idx = -1
        self.current_opponent_episode = 0

        ########################################################
        # PFSP tracking
        self.opponent_winrates = {}  # {path: deque(results)}
        self.opponent_games_played = {}  # {path: int}
        self.opponent_episodes = {}  # {path: episode_num}

        #########################################################
        # Regression tracking
        
        self.best_eval_vs_weak = 0.0
        self.best_checkpoint_path = None
        self.consecutive_eval_drops = 0

    def activate(self, episode, checkpoints_dir, agent):
        #########################################################
        # Activate self-play and seed the pool.
        #########################################################

        if self.active:
            return

        # Check performance gating
        if self.performance_gated:
            # Performance gating is checked elsewhere, just activate here
            pass

        self.active = True
        self.start_episode = episode

        # Seed the pool with current agent
        seed_path = pathlib.Path(checkpoints_dir) / f'selfplay_seed_ep{episode}.pth'
        agent_state = agent.state()
        torch.save({'agent_state': agent_state}, seed_path)
        self.pool.append(str(seed_path))

        if self.use_pfsp:
            self.opponent_winrates[str(seed_path)] = deque(maxlen=100)
            self.opponent_games_played[str(seed_path)] = 0
            self.opponent_episodes[str(seed_path)] = episode

        print(f"Self-play activated at episode {episode}")
        print(f"Pool seeded with {len(self.pool)} opponent(s)")

    def should_activate(self, episode, eval_vs_weak, rolling_variance):
            #########################################################
        # Check if self-play should activate (performance gating).
        #########################################################
 
        if not self.performance_gated:
            return episode >= self.start_episode

        # Performance-gated activation  
        episodes_needed = 500  # Need at least 500 episodes of eval data
        if episode < self.start_episode + episodes_needed:
            return False

        if eval_vs_weak < self.gate_winrate:
            return False

        if rolling_variance > self.gate_variance:
            return False

        return True

    def update_pool(self, episode, agent, checkpoints_dir):
        #########################################################
        # Add current agent to opponent pool.
        #########################################################

        if not self.active or episode % self.save_interval != 0:
            return

        new_path = pathlib.Path(checkpoints_dir) / f'selfplay_pool_ep{episode}.pth'
        torch.save({'agent_state': agent.state()}, new_path)
        self.pool.append(str(new_path))

        if self.use_pfsp:
            self.opponent_winrates[str(new_path)] = deque(maxlen=100)
            self.opponent_games_played[str(new_path)] = 0
            self.opponent_episodes[str(new_path)] = episode

        #########################################################
        # Keep pool bounded
        #########################################################
        removed_episode = None
        if len(self.pool) > self.pool_size:
            old_path = self.pool.pop(0)
            removed_episode = self.opponent_episodes.get(old_path, "unknown")

            if self.use_pfsp and old_path in self.opponent_winrates:
                del self.opponent_winrates[old_path]
                del self.opponent_games_played[old_path]
                del self.opponent_episodes[old_path]

        return removed_episode

    def select_opponent(self, episode):
        #########################################################
        # Select opponent from pool for this episode.
        #########################################################

        if not self.active or not self.pool:
            return True  # Use weak if pool empty

        #########################################################
        # Decide whether to use weak opponent
        #########################################################
        use_weak = np.random.random() < self.current_anchor_ratio

        if not use_weak:
            #########################################################
            # Select from pool
            #########################################################
            if self.use_pfsp:
                selected_path = self._pfsp_select()
            else:
                selected_path = np.random.choice(self.pool)

            #########################################################
            # Load opponent
            #########################################################
            self._load_opponent(selected_path)

        else:
            self.opponent = None  # Will use fixed opponent

        return use_weak

    def _pfsp_select(self):
        #########################################################
        # Select opponent using PFSP weighting.
        #########################################################
        weights = []
        valid_opponents = []

        for opp_path in self.pool:
            if opp_path in self.opponent_winrates and len(self.opponent_winrates[opp_path]) >= 10:
                results = list(self.opponent_winrates[opp_path])
                wins = sum(1 for r in results if r == 1)
                winrate = wins / len(results)

                from .pfsp import pfsp_weight
                weight = pfsp_weight(winrate, mode=self.pfsp_mode)
                weights.append(weight)
                valid_opponents.append(opp_path)
            else:
                weights.append(1.0)
                valid_opponents.append(opp_path)

        if sum(weights) > 0:
            weights_array = np.array(weights) / sum(weights)
            return np.random.choice(valid_opponents, p=weights_array)
        else:
            return np.random.choice(self.pool)

    def _load_opponent(self, path):
        #########################################################
        # Load opponent network from checkpoint.
        #########################################################
        if path == self.opponent_path:
            return  # Already loaded    

        try:
            opponent_state = torch.load(path, map_location=self.device)

            #########################################################
            # Extract policy state
            #########################################################
            if isinstance(opponent_state, tuple):
                policy_state = opponent_state[2]
            elif isinstance(opponent_state, dict) and 'agent_state' in opponent_state:
                agent_state = opponent_state['agent_state']
                if isinstance(agent_state, tuple):
                    policy_state = agent_state[2]
                else:
                    policy_state = agent_state
            else:
                policy_state = opponent_state

            #########################################################
            # Create opponent network if needed
            #########################################################
            if self.opponent is None:
                #########################################################
                # Need to infer sizes from policy_state
                #########################################################
                readout_shape = policy_state.get('readout.weight', 'N/A').shape if isinstance(policy_state, dict) else 'N/A'
                output_size = readout_shape[1] if readout_shape != 'N/A' else 4
                input_size = policy_state.get('0.weight', 'N/A').shape[1] if isinstance(policy_state, dict) else 18

                self.opponent = TD3Agent(
                    input_size=input_size,
                    hidden_sizes=[256, 256],
                    output_size=output_size,
                    output_activation=torch.nn.Tanh()
                ).to(self.device)

            #########################################################
            # Load the matching keys into the opponent
            current_state = self.opponent.state_dict()
            filtered_state = {}
            for key, param in policy_state.items():
                if key in current_state:
                    if param.shape == current_state[key].shape:
                        filtered_state[key] = param

            if filtered_state:
                self.opponent.load_state_dict(filtered_state, strict=False)
                self.opponent.eval()
                self.opponent_path = path
                self.current_opponent_idx = self.pool.index(path)
                self.current_opponent_episode = self.opponent_episodes.get(path, 0)

        except Exception as e:
            #########################################################
            print(f"Failed to load self-play opponent: {e}")
            #########################################################
            self.opponent = None

    def get_action(self, obs):
        #########################################################
        # Get action from self-play opponent.
        #########################################################
        # Args:
        #     obs: Opponent observation
        # Returns:
        #     action: Opponent action (or None if using fixed opponent)
        if self.opponent is None:
            return None

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.opponent(obs_tensor).cpu().numpy()[0]
        #########################################################
        return action

    def record_result(self, winner, use_weak):
        #########################################################
        # Record game result against opponent.
        # winner: Game result (1=win, -1=loss, 0=tie)
        # use_weak: Whether opponent was weak (not self-play)
        if use_weak or not self.use_pfsp:
            return

        if self.opponent_path in self.opponent_winrates:
            self.opponent_winrates[self.opponent_path].append(winner)
            self.opponent_games_played[self.opponent_path] += 1
            #########################################################

    def update_anchor_ratio(self, drop_from_peak):
        #########################################################
        # Update anchor ratio based on forgetting detection.
        #########################################################
        # Args:
        #     drop_from_peak: How much eval vs weak dropped from peak
        if not self.dynamic_anchor_mixing:
            return

        if drop_from_peak > 0.10:  # Dropped >10% from peak
            #########################################################
            # Boost anchor training
            #########################################################
            self.current_anchor_ratio = min(0.7, self.current_anchor_ratio + 0.2)
            print(f"Forgetting detected! Boosting anchor ratio to {self.current_anchor_ratio:.0%}")
        elif drop_from_peak < 0.03 and self.current_anchor_ratio > self.weak_ratio:
            #########################################################
            # Performance recovered, reduce anchor back to baseline
            #########################################################
            self.current_anchor_ratio = max(self.weak_ratio, self.current_anchor_ratio - 0.1)
            print(f"Performance stable. Reducing anchor ratio to {self.current_anchor_ratio:.0%}")

    def check_regression(self, eval_vs_weak):
        #########################################################
        # Check for performance regression.
        #########################################################
        # Args:
        #     eval_vs_weak: Current evaluation vs weak
        # Returns:
        #     tuple: (should_rollback, checkpoint_path_to_load)
        if not self.regression_rollback:
            return False, None

        if eval_vs_weak > self.best_eval_vs_weak:

            self.best_eval_vs_weak = eval_vs_weak
            self.best_checkpoint_path = None  # Will be set when saving
            self.consecutive_eval_drops = 0
            return False, None

        drop_from_best = self.best_eval_vs_weak - eval_vs_weak


        if drop_from_best > self.regression_threshold:
            self.consecutive_eval_drops += 1
            if self.consecutive_eval_drops >= 2:
                #########################################################
                # Rollback!
                #########################################################
                print(f"Regression detected! Dropping {drop_from_best:.1%} from best")
                return True, self.best_checkpoint_path
        else:
            #########################################################
            # Minor drop or stable
            #########################################################
            self.consecutive_eval_drops = 0
            return False, None

    def set_best_checkpoint(self, path):
        # Set the best checkpoint path for rollback.
        self.best_checkpoint_path = path

    def get_stats(self):
        # get log stats, useful for see what's happening

        stats = {
            'active': 1.0 if self.active else 0.0,  # are we do self play?? 1 if yes
            'pool_size': len(self.pool),  # how many bots we have
            'using_weak_opponent': 1.0 if self.opponent is None else 0.0,  # 1 if using weak, 0 if not
            'weak_ratio_target': self.weak_ratio,  # target percent vs weak
        }

        # anchor ratio stuff, if cheating for forgetting or smthg
        if self.dynamic_anchor_mixing:
            stats['current_anchor_ratio'] = self.current_anchor_ratio  # may be above weak_ratio sometimes

        #########################################################
        # pfsp winrate, only show if have that
        if self.use_pfsp and len(self.opponent_winrates) > 0:
            all_winrates = []
            for results in self.opponent_winrates.values():
                if len(results) >= 10: 
                    wins = sum(1 for r in results if r == 1)
                    all_winrates.append(wins / len(results))

            if all_winrates:
                stats['pfsp_avg_winrate'] = np.mean(all_winrates)  # average
                stats['pfsp_min_winrate'] = np.min(all_winrates)  # worst
                stats['pfsp_max_winrate'] = np.max(all_winrates)  # best

            # which opponent now, show index & episode
            if self.current_opponent_idx >= 0:
                stats['opponent_current_index'] = self.current_opponent_idx
                stats['opponent_current_episode'] = self.current_opponent_episode
                # not real age, is how long since start
                stats['opponent_current_age'] = self.start_episode - self.current_opponent_episode if self.active else 0
        #########################################################


        # regression save, if roll back mode on
        if self.regression_rollback:
            stats['best_eval_vs_weak'] = self.best_eval_vs_weak  # best seen weak eval
            stats['consecutive_drops'] = self.consecutive_eval_drops  # how many bad in row

        return stats 
