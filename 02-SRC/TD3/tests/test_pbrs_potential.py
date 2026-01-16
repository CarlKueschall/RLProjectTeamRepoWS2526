"""
PBRS Potential Function Validation Tests

These tests verify the minimal PBRS implementation:
1. φ_chase only rewards when puck is MOVING (prevents "do nothing" exploit)
2. φ_defensive provides triangle defense in own half
3. Terminal potential is forced to 0
4. Scaling keeps PBRS < sparse rewards

Run with: python tests/test_pbrs_potential.py
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards.pbrs import (
    compute_potential,
    compute_pbrs,
    PBRSReward,
    get_potential_components,
    PUCK_MOVING_THRESHOLD,
    OWN_GOAL,
    MAX_DISTANCE,
)


def create_obs(player_pos=(0, 0), puck_pos=(0, 0), puck_vel=(0, 0)):
    """Create a test observation with specified positions/velocities."""
    obs = np.zeros(18)
    obs[0:2] = player_pos     # Player position
    obs[12:14] = puck_pos     # Puck position
    obs[14:16] = puck_vel     # Puck velocity
    return obs


class TestChaseComponent:
    """Test that φ_chase only rewards when puck is MOVING."""

    def test_stationary_puck_no_reward(self):
        """Standing next to stationary puck should give ZERO chase reward."""
        # Agent right next to stationary puck
        obs = create_obs(player_pos=(0, 0), puck_pos=(0.1, 0), puck_vel=(0, 0))
        components = get_potential_components(obs)

        assert not components['puck_is_moving'], "Puck should be stationary"
        assert components['phi_chase'] == 0.0, "NO reward for stationary puck!"

    def test_moving_puck_rewards_proximity(self):
        """Being close to a MOVING puck should give reward."""
        # Agent near moving puck
        obs = create_obs(player_pos=(0, 0), puck_pos=(0.5, 0), puck_vel=(1.0, 0))
        components = get_potential_components(obs)

        assert components['puck_is_moving'], "Puck should be moving"
        assert components['phi_chase'] < 0, "Chase component should be negative (closer = less negative)"
        assert components['phi_chase'] > -0.1, "Close to puck = small negative"

    def test_far_from_moving_puck_more_negative(self):
        """Being far from moving puck should be worse (more negative)."""
        obs_close = create_obs(player_pos=(0, 0), puck_pos=(0.5, 0), puck_vel=(1.0, 0))
        obs_far = create_obs(player_pos=(0, 0), puck_pos=(5, 0), puck_vel=(1.0, 0))

        phi_close = get_potential_components(obs_close)['phi_chase']
        phi_far = get_potential_components(obs_far)['phi_chase']

        assert phi_close > phi_far, "Closer to puck should be less negative (better)"

    def test_threshold_boundary(self):
        """Puck at exactly the moving threshold."""
        # Just below threshold - should be 0
        obs_slow = create_obs(puck_pos=(1, 0), puck_vel=(0.29, 0))
        # Just above threshold - should be active
        obs_fast = create_obs(puck_pos=(1, 0), puck_vel=(0.31, 0))

        comp_slow = get_potential_components(obs_slow)
        comp_fast = get_potential_components(obs_fast)

        assert comp_slow['phi_chase'] == 0.0, "Below threshold = no reward"
        assert comp_fast['phi_chase'] < 0.0, "Above threshold = active"


class TestDoNothingExploit:
    """Test that the 'stand next to stationary puck' exploit is prevented."""

    def test_no_reward_for_standing_near_stationary_puck(self):
        """The exact scenario that was causing problems: walk up and freeze."""
        # Puck spawns at (-2, 0), agent walks to (-2.1, 0) and stops
        obs = create_obs(player_pos=(-2.1, 0), puck_pos=(-2, 0), puck_vel=(0, 0))
        components = get_potential_components(obs)

        assert components['phi_chase'] == 0.0, "CRITICAL: No reward for standing near stationary puck!"

    def test_hitting_puck_enables_chase_reward(self):
        """After hitting puck (it moves), agent can get chase reward."""
        # Before hit: puck stationary
        obs_before = create_obs(player_pos=(-2.1, 0), puck_pos=(-2, 0), puck_vel=(0, 0))
        # After hit: puck moving
        obs_after = create_obs(player_pos=(-2.1, 0), puck_pos=(-1.5, 0), puck_vel=(2, 0))

        phi_before = get_potential_components(obs_before)['phi_chase']
        phi_after = get_potential_components(obs_after)['phi_chase']

        assert phi_before == 0.0, "No chase reward before hitting"
        assert phi_after < 0.0, "Chase reward activates after puck moves"

    def test_standing_still_no_pbrs_accumulation(self):
        """Standing still should not accumulate PBRS rewards."""
        # Same state twice (standing still)
        obs = create_obs(player_pos=(-2.1, 0), puck_pos=(-2, 0), puck_vel=(0, 0))

        pbrs = compute_pbrs(obs, obs, done=False)

        # PBRS = gamma * phi(s') - phi(s) = 0.99 * phi - phi = -0.01 * phi
        # But phi = 0 (only defensive might be active), so PBRS should be very small
        assert abs(pbrs) < 1.0, f"Standing still should give minimal PBRS, got {pbrs}"


class TestDefensivePositioning:
    """Test triangle defense when puck in own half."""

    def test_defense_active_in_own_half(self):
        """Defensive component should be active when puck in own half."""
        obs = create_obs(player_pos=(0, 0), puck_pos=(-2, 0))
        components = get_potential_components(obs)

        assert components['puck_in_own_half'], "Puck at x=-2 is in own half"
        assert components['phi_defensive'] != 0, "Defense should be active in own half"

    def test_defense_inactive_in_opponent_half(self):
        """Defensive component should be zero when puck in opponent half."""
        obs = create_obs(player_pos=(0, 0), puck_pos=(2, 0))
        components = get_potential_components(obs)

        assert not components['puck_in_own_half'], "Puck at x=2 is in opponent half"
        assert components['phi_defensive'] == 0, "Defense should be inactive in opponent half"

    def test_good_defensive_position(self):
        """Good defensive position should have smaller penalty."""
        puck_pos = (-3, 0)
        ideal_pos = OWN_GOAL + 0.4 * (np.array(puck_pos) - OWN_GOAL)

        obs_good = create_obs(player_pos=ideal_pos, puck_pos=puck_pos)
        obs_bad = create_obs(player_pos=(0, 2), puck_pos=puck_pos)

        comp_good = get_potential_components(obs_good)
        comp_bad = get_potential_components(obs_bad)

        # At ideal position, error should be ~0, so phi_defensive ~0
        assert comp_good['phi_defensive'] > -0.01, "Perfect position = ~0 penalty"
        assert comp_good['phi_defensive'] > comp_bad['phi_defensive'], \
            "Good position should have smaller penalty"

    def test_defensive_position_calculation(self):
        """Verify the ideal defensive position is calculated correctly."""
        puck_pos = (-2, 0)
        obs = create_obs(puck_pos=puck_pos)
        components = get_potential_components(obs)

        # Ideal = OWN_GOAL + 0.4 * (puck - OWN_GOAL)
        # OWN_GOAL = (-4.5, 0), puck = (-2, 0)
        # Ideal = (-4.5, 0) + 0.4 * (2.5, 0) = (-4.5 + 1.0, 0) = (-3.5, 0)
        expected_ideal = np.array([-3.5, 0])

        assert np.allclose(components['ideal_defensive_pos'], expected_ideal), \
            f"Ideal pos should be {expected_ideal}, got {components['ideal_defensive_pos']}"


class TestTerminalPotential:
    """Test that terminal states have zero potential."""

    def test_terminal_potential_is_zero(self):
        """PBRS at terminal state should force phi(s') = 0."""
        obs = create_obs(puck_pos=(0, 0), puck_vel=(1, 0))
        obs_next = create_obs(puck_pos=(0.1, 0), puck_vel=(1, 0))

        # Terminal transition
        pbrs_terminal = compute_pbrs(obs, obs_next, done=True)

        # Terminal case: F = gamma * 0 - phi(s) = -phi(s)
        phi_current = compute_potential(obs)
        expected_terminal = 0.99 * 0 - phi_current

        assert abs(pbrs_terminal - expected_terminal) < 1e-6, \
            "Terminal PBRS should be -phi(s)"


class TestScaling:
    """Test that PBRS scaling keeps shaped rewards appropriate."""

    def test_default_scale_is_0_02(self):
        """Default pbrs_scale should be 0.02."""
        shaper = PBRSReward()
        assert shaper.pbrs_scale == 0.02, "Default scale should be 0.02"

    def test_max_potential_range(self):
        """Verify the potential range is as expected."""
        # Worst case: far from moving puck in own half, bad defensive position
        obs_worst = create_obs(player_pos=(4, 2), puck_pos=(-4, 0), puck_vel=(1, 0))
        # Best case: on top of moving puck, perfect defensive position
        puck_pos = (-2, 0)
        ideal_pos = OWN_GOAL + 0.4 * (np.array(puck_pos) - OWN_GOAL)
        obs_best = create_obs(player_pos=ideal_pos, puck_pos=puck_pos, puck_vel=(1, 0))

        phi_worst = compute_potential(obs_worst)
        phi_best = compute_potential(obs_best)

        print(f"phi_worst: {phi_worst}")
        print(f"phi_best: {phi_best}")
        print(f"Range: {phi_best - phi_worst}")

        # phi_worst should be very negative (far from puck + bad defense)
        # phi_best should be much less negative (near puck + good defense)
        assert phi_worst < -100, f"Worst case should be < -100, got {phi_worst}"
        assert phi_best > -20, f"Best case should be > -20, got {phi_best}"
        assert phi_best > phi_worst, "Best should be better than worst"

    def test_episode_shaping_less_than_sparse(self):
        """Total episode PBRS should be less than sparse reward magnitude."""
        # Max potential change
        obs_worst = create_obs(player_pos=(4, 2), puck_pos=(-4, 0), puck_vel=(1, 0))
        puck_pos = (-2, 0)
        ideal_pos = OWN_GOAL + 0.4 * (np.array(puck_pos) - OWN_GOAL)
        obs_best = create_obs(player_pos=ideal_pos, puck_pos=puck_pos, puck_vel=(1, 0))

        phi_worst = compute_potential(obs_worst)
        phi_best = compute_potential(obs_best)

        max_episode_shaping = abs(phi_best - phi_worst) * 0.02
        sparse_reward = 10.0  # Win/loss sparse reward (before reward_scale)

        print(f"Max episode PBRS (0.02 scale): {max_episode_shaping}")
        print(f"Sparse reward: {sparse_reward}")

        assert max_episode_shaping < sparse_reward, \
            f"Episode PBRS ({max_episode_shaping}) should be < sparse ({sparse_reward})"


class TestPBRSRewardClass:
    """Test the PBRSReward wrapper class."""

    def test_basic_computation(self):
        """Basic PBRS computation should work."""
        shaper = PBRSReward(pbrs_scale=0.02)
        obs = create_obs(puck_pos=(0, 0), puck_vel=(1, 0))
        obs_next = create_obs(puck_pos=(0.1, 0), puck_vel=(1, 0))

        reward = shaper.compute(obs, obs_next, done=False)

        assert isinstance(reward, float), "Should return float"

    def test_constant_weight_mode(self):
        """Constant weight mode should not anneal."""
        shaper = PBRSReward(constant_weight=True, annealing_episodes=100)
        shaper.set_self_play_start(0)

        weight = shaper.get_annealing_weight(episode=500)
        assert weight == 1.0, "Constant weight should always be 1.0"


class TestNoEncodedStrategy:
    """Test that we're NOT encoding specific strategies."""

    def test_no_shooting_direction_reward(self):
        """There should be no reward for shooting in any particular direction."""
        # Shooting toward goal
        obs_toward = create_obs(puck_pos=(0, 0), puck_vel=(2, 0))
        # Shooting away from goal
        obs_away = create_obs(puck_pos=(0, 0), puck_vel=(-2, 0))
        # Shooting sideways
        obs_side = create_obs(puck_pos=(0, 0), puck_vel=(0, 2))

        # All should have similar phi_chase (only depends on distance, not direction)
        # The puck is at same position in all cases
        comp_toward = get_potential_components(obs_toward)
        comp_away = get_potential_components(obs_away)
        comp_side = get_potential_components(obs_side)

        # phi_chase only depends on agent-puck distance, which is same for all
        assert comp_toward['phi_chase'] == comp_away['phi_chase'] == comp_side['phi_chase'], \
            "Shot direction should not affect chase reward"

    def test_no_puck_to_goal_reward(self):
        """Puck position relative to goal should NOT be directly rewarded."""
        # Puck near opponent goal (moving)
        obs_near_goal = create_obs(puck_pos=(4, 0), puck_vel=(1, 0))
        # Puck far from opponent goal (moving)
        obs_far_goal = create_obs(puck_pos=(-4, 0), puck_vel=(1, 0))

        # Same agent position in both
        comp_near = get_potential_components(obs_near_goal)
        comp_far = get_potential_components(obs_far_goal)

        # phi_chase depends on agent-puck distance, not puck-goal distance
        # With agent at (0,0), distances are 4 and 4, so chase should be same
        assert abs(comp_near['phi_chase'] - comp_far['phi_chase']) < 0.01, \
            "Puck-to-goal distance should not affect reward"


def run_all_tests():
    """Run all tests without pytest."""
    import traceback

    test_classes = [
        TestChaseComponent,
        TestDoNothingExploit,
        TestDefensivePositioning,
        TestTerminalPotential,
        TestScaling,
        TestPBRSRewardClass,
        TestNoEncodedStrategy,
    ]

    total = 0
    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in methods:
            total += 1
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  PASS: {method_name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL: {method_name}")
                print(f"        {e}")
                failed += 1
            except Exception as e:
                print(f"  ERROR: {method_name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
