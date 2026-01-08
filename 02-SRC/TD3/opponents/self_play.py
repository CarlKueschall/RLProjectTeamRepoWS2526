"""
AI Usage Declaration:
This file was developed with assistance from AI autocomplete features in Cursor AI IDE.
"""

import os
import pathlib
from collections import deque
import numpy as np
import torch
from agents.td3_agent import TD3Agent
from agents.model import Model

class SelfPlayManager:
    ######################################################
    # Manage self-play opponent pool and selection.
    ######################################################
    def __init__(self, pool_size=10, save_interval=500,
                 weak_ratio=0.5, device=None,
                 use_pfsp=False, pfsp_mode="variance",
                 dynamic_anchor_mixing=False,
                 performance_gated=False, gate_winrate=0.90,
                 regression_rollback=False, regression_threshold=0.15,
                 observation_space=None, action_space=None):
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
        # regression_rollback: Enable automatic rollback
        # regression_threshold: Rollback threshold (e.g., 0.15 = 15%)
        # observation_space: Gym observation space for creating opponents
        # action_space: Gym action space for creating opponents
        self.pool_size = pool_size
        self.save_interval = save_interval
        self.weak_ratio = weak_ratio
        self.current_anchor_ratio = weak_ratio  # For dynamic mixing
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space

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

        ################################################
        # PFSP tracking
        self.opponent_winrates = {}  # {path: deque(results)}
        self.opponent_games_played = {}  # {path: int}
        self.opponent_episodes = {}  # {path: episode_num}

        #################################################
        # Regression tracking
        
        self.best_eval_vs_weak = 0.0
        self.best_checkpoint_path = None
        self.consecutive_eval_drops = 0

        #################################################
        # Anchor buffer balance tracking (for weak/strong balance)
        self.anchor_weak_count = 0  # Count of weak opponent episodes in anchor
        self.anchor_strong_count = 0  # Count of strong opponent episodes in anchor

    def activate(self, episode, checkpoints_dir, agent):
        ######################################################
        # Activate self-play and seed the pool with historical checkpoints.
        #####################################################

        if self.active:
            return

        # Check performance gating
        if self.performance_gated:
            # Performance gating is checked elsewhere, just activate here
            pass

        self.active = True
        self.start_episode = episode
        
        # Reset anchor balance tracking when self-play activates
        # (experiences before activation are already in anchor buffer)
        self.anchor_weak_count = 0
        self.anchor_strong_count = 0

        # Seed the pool with current agent
        seed_path = pathlib.Path(checkpoints_dir) / f'selfplay_seed_ep{episode}.pth'
        agent_state = agent.state()
        torch.save({'agent_state': agent_state}, seed_path)
        self.pool.append(str(seed_path))

        if self.use_pfsp:
            self.opponent_winrates[str(seed_path)] = deque(maxlen=100)
            self.opponent_games_played[str(seed_path)] = 0
            self.opponent_episodes[str(seed_path)] = episode

        # Seed pool with historical checkpoints (for diversity)
        # Look for recent checkpoints in parent directory
        parent_dir = pathlib.Path(checkpoints_dir).parent / 'checkpoints'
        if parent_dir.exists():
            # Find checkpoints from the last N episodes before activation
            seed_interval = max(500, (episode - self.start_episode) // (self.pool_size - 1)) if episode > 1000 else 500
            for ep in range(max(1000, episode - seed_interval * (self.pool_size - 1)), episode, seed_interval):
                checkpoint_pattern = f'*_ep{ep}_*.pth'
                matching_files = list(parent_dir.glob(checkpoint_pattern))
                if matching_files and len(self.pool) < self.pool_size:
                    checkpoint_path = str(matching_files[0])
                    self.pool.append(checkpoint_path)
                    if self.use_pfsp:
                        self.opponent_winrates[checkpoint_path] = deque(maxlen=100)
                        self.opponent_games_played[checkpoint_path] = 0
                        self.opponent_episodes[checkpoint_path] = ep

        print("\n" + "="*70)
        print(f"🎮 SELF-PLAY ACTIVATED AT EPISODE {episode}! 🎮")
        print("="*70)
        print(f"Pool seeded with {len(self.pool)} opponent(s)")
        print(f"Dynamic anchor mixing: {self.dynamic_anchor_mixing}")
        print(f"PFSP enabled: {self.use_pfsp}")
        print(f"Regression rollback: {self.regression_rollback}")
        print("="*70 + "\n")

    def should_activate(self, episode, eval_vs_weak):
            #########################################################
        # Check if self-play should activate (performance gating).
        #########################################################
 
        if not self.performance_gated:
            return episode >= self.start_episode

        # Performance-gated activation
        # Need at least one evaluation before we can check performance gates
        if eval_vs_weak is None:
            return False

        # Check win-rate gate
        if eval_vs_weak < self.gate_winrate:
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
        # Returns: 'weak', 'strong', or 'self-play'
        #########################################################

        if not self.active or not self.pool:
            return 'weak'  # Use weak if pool empty or not active

        #########################################################
        # Decide whether to use anchor opponent (weak/strong) or self-play
        #########################################################
        use_anchor = np.random.random() < self.current_anchor_ratio

        if use_anchor:
            #########################################################
            # Balance weak vs strong in anchor buffer
            # Target: approximately equal distribution
            #########################################################
            total_anchor = self.anchor_weak_count + self.anchor_strong_count
            
            if total_anchor == 0:
                # Start with weak
                self.anchor_weak_count += 1
                self.opponent = None
                return 'weak'
            
            # Calculate current ratio
            weak_ratio_in_anchor = self.anchor_weak_count / total_anchor if total_anchor > 0 else 0.5
            
            # Prefer the one that's underrepresented (target: 50/50)
            if weak_ratio_in_anchor < 0.5:
                # Weak is underrepresented, use weak
                self.anchor_weak_count += 1
                self.opponent = None
                return 'weak'
            elif weak_ratio_in_anchor > 0.5:
                # Strong is underrepresented, use strong
                self.anchor_strong_count += 1
                self.opponent = None
                return 'strong'
            else:
                # Balanced, random choice
                if np.random.random() < 0.5:
                    self.anchor_weak_count += 1
                    self.opponent = None
                    return 'weak'
                else:
                    self.anchor_strong_count += 1
                    self.opponent = None
                    return 'strong'
        else:
            ####################################
            # Select from self-play pool
            if self.use_pfsp:
                selected_path = self._pfsp_select()
            else:
                selected_path = np.random.choice(self.pool)

            # Load opponent
            self._load_opponent(selected_path)
            return 'self-play'

    def _pfsp_select(self):
        # Select opponent using PFSP weighting.
        
        weights = []
        valid_opponents = []

        for opp_path in self.pool:
            if opp_path in self.opponent_winrates and len(self.opponent_winrates[opp_path]) >= 10:
                results = list(self.opponent_winrates[opp_path])
                wins = 0
                for r in results:
                    if r == 1:
                        wins += 1
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
            checkpoint = torch.load(path, map_location=self.device)

            #########################################################
            # Extract agent state tuple (Q1_state, Q2_state, policy_state)
            #########################################################
            if isinstance(checkpoint, dict) and 'agent_state' in checkpoint:
                agent_state = checkpoint['agent_state']
            elif isinstance(checkpoint, tuple):
                agent_state = checkpoint
            else:
                # Fallback: assume it's the agent_state directly
                agent_state = checkpoint

            # Ensure agent_state is a tuple of (Q1, Q2, policy)
            if not isinstance(agent_state, tuple) or len(agent_state) != 3:
                raise ValueError(f"Expected agent_state to be tuple of (Q1, Q2, policy), got {type(agent_state)}")

            Q1_state, Q2_state, policy_state = agent_state

            #########################################################
            # Infer network architecture from checkpoint
            #########################################################
            # Extract critic hidden sizes from Q1 state dict
            # Model structure: layers.0, layers.1, ..., layers.N, output_layer
            # hidden_sizes = [output_dim of layers.0, output_dim of layers.1, ..., output_dim of layers.N]
            critic_hidden_sizes = []
            if isinstance(Q1_state, dict):
                layer_idx = 0
                while f'layers.{layer_idx}.weight' in Q1_state:
                    # Output dimension of this layer (shape[0] is the output size)
                    layer_output_size = Q1_state[f'layers.{layer_idx}.weight'].shape[0]
                    critic_hidden_sizes.append(layer_output_size)
                    layer_idx += 1
                # All layers.X are hidden layers, output_layer is separate
            
            # Extract actor hidden sizes from policy state dict
            actor_hidden_sizes = []
            if isinstance(policy_state, dict):
                layer_idx = 0
                while f'layers.{layer_idx}.weight' in policy_state:
                    # Output dimension of this layer
                    layer_output_size = policy_state[f'layers.{layer_idx}.weight'].shape[0]
                    actor_hidden_sizes.append(layer_output_size)
                    layer_idx += 1
                # All layers.X are hidden layers, output_layer is separate

            #########################################################
            # Create opponent network with inferred architecture
            # Always recreate to ensure architecture matches checkpoint
            #########################################################
            if self.observation_space is None or self.action_space is None:
                raise ValueError("observation_space and action_space must be provided to SelfPlayManager for loading opponents")
            
            # Use inferred architecture, or defaults if inference failed
            if not actor_hidden_sizes:
                actor_hidden_sizes = [256, 256]
            if not critic_hidden_sizes:
                critic_hidden_sizes = [256, 256, 128]
            
            # Create TD3Agent with inferred architecture (recreate if needed)
            self.opponent = TD3Agent(
                self.observation_space,
                self.action_space,
                hidden_sizes_actor=actor_hidden_sizes,
                hidden_sizes_critic=critic_hidden_sizes
            )
            # Set policy to eval mode
            self.opponent.policy.eval()

            #########################################################
            # Restore state using TD3Agent's restore_state method
            #########################################################
            self.opponent.restore_state(agent_state)
            # Ensure policy stays in eval mode
            self.opponent.policy.eval()
            
            self.opponent_path = path
            self.current_opponent_idx = self.pool.index(path)
            self.current_opponent_episode = self.opponent_episodes.get(path, 0)

        except Exception as e:
            print(f"Failed to load self-play opponent: {e}")
            import traceback
            traceback.print_exc()
            self.opponent = None

    def get_action(self, obs):
        #########################################################
        # Get action from self-play opponent.
        #########################################################
        if self.opponent is None:
            return None

        # TD3Agent.act() expects numpy array and handles device conversion internally
        # Use eps=0.0 for deterministic evaluation (no exploration noise)
        action = self.opponent.act(obs, eps=0.0)
        return action

    def record_result(self, winner, use_weak):
        #########################################################
        # Record game result against opponent.
        # winner: Game result (1=win, -1=loss, 0=tie)
        # use_weak: Whether opponent was weak (not self-play) - deprecated, kept for compatibility
        # Note: This is only called for self-play opponents now
        if use_weak or not self.use_pfsp:
            return

        if self.opponent_path in self.opponent_winrates:
            self.opponent_winrates[self.opponent_path].append(winner)
            self.opponent_games_played[self.opponent_path] += 1
            #########################################################

    def update_anchor_ratio(self, drop_from_peak):
        #########################################################
        # Update anchor ratio based on forgetting detection.

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
                
                print(f"Regression detected! Dropping {drop_from_best:.1%} from best")
                return True, self.best_checkpoint_path
            else:
                # Regression detected but need more consecutive drops
                return False, None
        else:
            
            # Minor drop or stable
            
            self.consecutive_eval_drops = 0
            return False, None

    def set_best_checkpoint(self, path):
        # Set the best checkpoint path for rollback.
        self.best_checkpoint_path = path

    def get_stats(self):
        #########################################################
        # Comprehensive self-play metrics for W&B tracking.
        # Provides detailed insights into opponent mixing, buffer
        # distribution, PFSP selection, and training dynamics.
        #########################################################

        stats = {
            #########################################################
            # Self-Play Status
            #########################################################
            'selfplay/active': 1.0 if self.active else 0.0,
            'selfplay/pool_size': len(self.pool),
            
            #########################################################
            # Buffer and Opponent Mixing
            #########################################################
            'selfplay/weak_ratio_target': self.weak_ratio,
        }

        #########################################################
        # Dynamic Anchor Mixing Metrics
        # Tracks adjustment of weak/strong ratio to prevent forgetting
        #########################################################
        if self.dynamic_anchor_mixing:
            stats['selfplay/anchor_ratio_current'] = self.current_anchor_ratio

        #########################################################
        # Anchor Buffer Balance (weak vs strong distribution)
        # Critical for preventing catastrophic forgetting of base skills
        #########################################################
        total_anchor = self.anchor_weak_count + self.anchor_strong_count
        if total_anchor > 0:
            anchor_weak_ratio = self.anchor_weak_count / total_anchor
            anchor_strong_ratio = self.anchor_strong_count / total_anchor
            stats['selfplay/anchor_weak_episodes'] = self.anchor_weak_count
            stats['selfplay/anchor_strong_episodes'] = self.anchor_strong_count
            stats['selfplay/anchor_weak_ratio'] = anchor_weak_ratio
            stats['selfplay/anchor_strong_ratio'] = anchor_strong_ratio
            stats['selfplay/anchor_balance_score'] = 1.0 - abs(anchor_weak_ratio - 0.5) * 2.0  # 1.0 = perfect balance, 0.0 = all one type

        #########################################################
        # PFSP (Prioritized Fictitious Self-Play) Metrics
        # Shows opponent selection curriculum and curriculum diversity
        #########################################################
        if self.use_pfsp and len(self.opponent_winrates) > 0:
            opponent_winrates_list = []
            games_played_list = []
            
            for opp_path in self.opponent_winrates:
                results = list(self.opponent_winrates[opp_path])
                if len(results) >= 10:
                    wins = sum(1 for r in results if r == 1)
                    winrate = wins / len(results)
                    opponent_winrates_list.append(winrate)
                    games_played_list.append(len(results))

            if opponent_winrates_list:
                #########################################################
                # PFSP Opponent Statistics
                # Tracks which opponents are being used and their win rates
                #########################################################
                stats['selfplay/pfsp_num_opponents_tracked'] = len(opponent_winrates_list)
                stats['selfplay/pfsp_avg_winrate'] = np.mean(opponent_winrates_list)
                stats['selfplay/pfsp_std_winrate'] = np.std(opponent_winrates_list)
                stats['selfplay/pfsp_min_winrate'] = np.min(opponent_winrates_list)
                stats['selfplay/pfsp_max_winrate'] = np.max(opponent_winrates_list)
                stats['selfplay/pfsp_median_winrate'] = np.median(opponent_winrates_list)
                stats['selfplay/pfsp_diversity_metric'] = np.std(opponent_winrates_list)  # Higher = more diverse difficulty

            #########################################################
            # Current Opponent Tracking
            # Shows which opponent is currently being trained against
            #########################################################
            if self.current_opponent_idx >= 0 and self.active:
                stats['selfplay/opponent_pool_index'] = self.current_opponent_idx
                stats['selfplay/opponent_checkpoint_episode'] = self.current_opponent_episode
                opponent_age = self.start_episode - self.current_opponent_episode if self.active else 0
                stats['selfplay/opponent_age_episodes'] = opponent_age

        #########################################################
        # Regression Rollback Protection Metrics
        # Monitors performance drops and rollback triggers
        #########################################################
        if self.regression_rollback:
            stats['selfplay/best_eval_vs_weak'] = self.best_eval_vs_weak
            stats['selfplay/consecutive_eval_drops'] = self.consecutive_eval_drops
            stats['selfplay/rollback_enabled'] = 1.0

        return stats 
