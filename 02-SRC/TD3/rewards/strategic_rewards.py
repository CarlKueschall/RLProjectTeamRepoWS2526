import numpy as np


class StrategicRewardShaper:
    ######################################################
    # Strategic reward shaping with opponent awareness and diversity bonuses.
    ######################################################
    def __init__(self):
        #########################################################
        # Initialize strategic reward shaper.
        self.reset()

    def compute(self, obs_next, info, dist_to_puck):
        #########################################################
        # Compute strategic bonuses for a single step.
        #########################################################
        #Arguments:
        # obs_next: Next observation (18 dims)
        # info: Environment info dict
        # dist_to_puck: Distance from agent to puck
        #Returns:
        # dict: Strategic bonuses {'puck_touches', 'proximity', 'direction', 'goal_proximity',
        #                          'shot_bonus', 'shot_penalty', 'diversity', 'forcing'}
        bonuses = {
            'puck_touches': 0.0,  # how many times we touched puck
            'proximity': 0.0,  # how close we are to puck
            'direction': 0.0,  # moving puck toward goal
            'goal_proximity': 0.0,  # puck near their goal
            'shot_bonus': 0.0,  # good shot (clear path)
            'shot_penalty': 0.0,  # bad shot (blocked)
            'diversity': 0.0,  # varied attack patterns
            'forcing': 0.0,  # made opponent move
        }

        #########################################################
        # Extract positions
        #########################################################
        puck_x, puck_y = obs_next[12], obs_next[13]
        puck_velocity_x = obs_next[14]
        opponent_x, opponent_y = obs_next[6], obs_next[7]

        #########################################################
        # Environment's physics-based touch detection
        #########################################################
        env_touch_reward = info.get('reward_touch_puck', 0.0)
        env_closeness_reward = info.get('reward_closeness_to_puck', 0.0)
        env_puck_direction = info.get('reward_puck_direction', 0.0)

        #########################################################
        # FIX 1: Reducing the touch reward from 0.8 -> 0.3 because it was too high
        #########################################################
        if env_touch_reward > 0:
            self.puck_touches += 1
            bonuses['puck_touches'] = env_touch_reward * 0.3  # Reduced from 0.8 → 0.3, was too high before

        # FIX 1: Had to cut closeness reward in half (0.2 -> 0.1), was giving too much
        bonuses['proximity'] = env_closeness_reward * 0.1  # Reduced from 0.2 → 0.1

        #########################################################
        # FIX 1: Dropped puck direction reward from 0.6 -> 0.2, this was causing wild shooting
        # This was a major contributor to wild shooting (0.6 * 250 steps = 150 total!)
        bonuses['direction'] = env_puck_direction * 0.2  # Reduced from 0.6 → 0.2, was causing chaos

        # FIX 1: Lowered goal proximity reward from 0.233 -> 0.1, need less dense rewards
        opponent_goal_x = 2.5  # Opponent's goal position (right side)
        puck_goal_dist = abs(opponent_goal_x - puck_x)
        if puck_goal_dist < 1.5:  # Puck is near opponent's goal
            bonuses['goal_proximity'] = (1.5 - puck_goal_dist) * 0.1  # Reduced (max +0.15), less dense reward

        #########################################################
        # FIX 2: Adding opponent-aware shooting, check if they're blocking
        #########################################################
        if dist_to_puck < 0.8 and puck_velocity_x > 0.5:  # Near puck and moving toward goal
            # Check if opponent is blocking the shot path
            shot_vector = np.array([opponent_goal_x - puck_x, 0 - puck_y])
            shot_distance = np.linalg.norm(shot_vector)
            if shot_distance > 1e-6:
                shot_unit = shot_vector / shot_distance
                opp_to_puck = np.array([opponent_x - puck_x, opponent_y - puck_y])
                projection_length = np.dot(opp_to_puck, shot_unit)

                opponent_blocking = False
                if 0 < projection_length < shot_distance:
                    closest_point = np.array([puck_x, puck_y]) + projection_length * shot_unit
                    perpendicular_dist = np.linalg.norm(opp_to_puck - projection_length * shot_unit)
                    opponent_blocking = perpendicular_dist < 0.5  # Within shot corridor

                #########################################################
                # FIX 2: Now tracking clear vs blocked shots so we can see shot quality
                if opponent_blocking:
                    self.shots_blocked += 1
                    # Penalty for shooting when opponent blocks (-0.2)
                    bonuses['shot_penalty'] = -0.2  # don't shoot into opponent!
                else:
                    self.shots_clear += 1
                    # Reward for shooting with clear path (+0.15, increased from old +0.1)
                    shooting_bonus = min(0.15, puck_velocity_x * 0.2)
                    bonuses['shot_bonus'] = shooting_bonus  # good shot, clear path
                    #########################################################
                    # FIX 3: Tracking shot angles for diversity, only counting clear shots
                    shot_angle = np.arctan2(puck_y, opponent_goal_x - puck_x)
                    self.shot_angles.append(shot_angle)

                    # FIX 3: Also tracking which side we attack from for diversity bonus
                    if puck_y > 0.3:
                        self.attack_sides.append('left')
                    elif puck_y < -0.3:
                        self.attack_sides.append('right')
                    else:
                        self.attack_sides.append('center')

        #########################################################
        # Backup distance-based touch detection
        # In case env rewards aren't working, use our own with larger threshold
        # FIX 1: Cutting proximity bonus in half (0.2 -> 0.1) here too
        if dist_to_puck < 0.5 and env_touch_reward == 0:
            # Only give our reward if env didn't detect touch
            proximity_bonus = (0.5 - dist_to_puck) * 0.1  # Reduced (max +0.05)
            bonuses['proximity'] += proximity_bonus
            if dist_to_puck < 0.3:  # More generous touch threshold
                self.puck_touches += 1  # Count our own touches too
        #########################################################

        return bonuses

    def compute_episode_end_bonuses(self):
        #########################################################
        # Compute bonuses that depend on entire episode stats.
        #Returns:
        # dict: Diversity and forcing bonuses
        bonuses = {
            'diversity': 0.0,
            'forcing': 0.0,
        }

        #########################################################
        # FIX 3: Adding diversity bonus to reward varied attack patterns
        if len(self.attack_sides) >= 3:
            unique_sides = len(set(self.attack_sides))
            if unique_sides >= 2:  # Attacked from at least 2 different sides
                bonuses['diversity'] = 0.5 * unique_sides  # +0.5 per unique side (max +1.5), mix it up!

        #########################################################
        # FIX 4: New opponent forcing metric, reward when we make them move around
        #########################################################
        if len(self.opponent_positions) > 10:
            opponent_distances = []
            for i in range(1, len(self.opponent_positions)):
                dist = np.linalg.norm(np.array(self.opponent_positions[i]) -
                                      np.array(self.opponent_positions[i-1]))
                opponent_distances.append(dist)

            self.total_opponent_movement = sum(opponent_distances)
            self.avg_opponent_movement = np.mean(opponent_distances) if opponent_distances else 0.0

            # Reward if we forced opponent to move significantly
            if self.total_opponent_movement > 5.0:  # Opponent moved more than 5 units
                bonuses['forcing'] = min(1.0, self.total_opponent_movement * 0.1)  # make them work for it

        #########################################################
        # Store bonuses for logging
        self.diversity_bonus = bonuses['diversity']
        self.forcing_bonus = bonuses['forcing']

        return bonuses

    def record_opponent_position(self, opponent_pos):
        #########################################################
        # Record opponent position for forcing metric.
        self.opponent_positions.append(opponent_pos)

    def get_episode_stats(self):
        #########################################################
        # Get episode statistics for logging.
        #Returns:
        # dict: Episode statistics
        return {
            'puck_touches': self.puck_touches,
            'shots_blocked': self.shots_blocked,
            'shots_clear': self.shots_clear,
            'shot_quality_ratio': (self.shots_clear / (self.shots_blocked + self.shots_clear)
                                 if (self.shots_blocked + self.shots_clear) > 0 else 0.0),
            'attack_sides_unique': len(set(self.attack_sides)),
            'total_opponent_movement': self.total_opponent_movement,
            'avg_opponent_movement': self.avg_opponent_movement,
            'attack_diversity_bonus': self.diversity_bonus,
            'forcing_bonus': self.forcing_bonus,
        }

    def reset(self):
        #########################################################
        # Reset episode-specific state.
        self.puck_touches = 0
        self.shots_blocked = 0
        self.shots_clear = 0
        self.shot_angles = []
        self.attack_sides = []
        self.opponent_positions = []
        self.total_opponent_movement = 0.0
        self.avg_opponent_movement = 0.0
        self.diversity_bonus = 0.0
        self.forcing_bonus = 0.0

