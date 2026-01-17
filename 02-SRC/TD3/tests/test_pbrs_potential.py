"""
PBRS Potential Function Validation Tests

These tests verify the PBRS implementation with reduced stationary puck weight:
1. φ_chase rewards proximity to puck (full weight for moving, 30% for stationary)
2. Terminal potential is forced to 0
3. Scaling keeps PBRS < sparse rewards
4. Reduced stationary weight prevents "stand and wait" exploit while guiding approach

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
    STATIONARY_PUCK_WEIGHT,
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
    """Test φ_chase with full/reduced weight based on puck movement."""

    def test_stationary_puck_reduced_reward(self):
        """Standing next to stationary puck should give REDUCED (30%) reward."""
        # Agent right next to stationary puck
        obs = create_obs(player_pos=(0, 0), puck_pos=(0.1, 0), puck_vel=(0, 0))
        components = get_potential_components(obs)

        assert not components['puck_is_moving'], "Puck should be stationary"
        assert components['weight_applied'] == STATIONARY_PUCK_WEIGHT, "Should use reduced weight"
        # phi_chase should be 30% of what it would be if moving
        expected = STATIONARY_PUCK_WEIGHT * components['phi_chase_base']
        assert abs(components['phi_chase'] - expected) < 1e-6, "Should be 30% of base"

    def test_moving_puck_full_reward(self):
        """Being close to a MOVING puck should give FULL reward."""
        # Agent near moving puck
        obs = create_obs(player_pos=(0, 0), puck_pos=(0.5, 0), puck_vel=(1.0, 0))
        components = get_potential_components(obs)

        assert components['puck_is_moving'], "Puck should be moving"
        assert components['weight_applied'] == 1.0, "Should use full weight"
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
        # Just below threshold - should use reduced weight
        obs_slow = create_obs(puck_pos=(1, 0), puck_vel=(0.29, 0))
        # Just above threshold - should use full weight
        obs_fast = create_obs(puck_pos=(1, 0), puck_vel=(0.31, 0))

        comp_slow = get_potential_components(obs_slow)
        comp_fast = get_potential_components(obs_fast)

        assert comp_slow['weight_applied'] == STATIONARY_PUCK_WEIGHT, "Below threshold = reduced"
        assert comp_fast['weight_applied'] == 1.0, "Above threshold = full"
        # Fast should be more negative (full weight)
        assert comp_fast['phi_chase'] < comp_slow['phi_chase'], "Moving puck = stronger signal"

    def test_stationary_vs_moving_magnitude_ratio(self):
        """Stationary puck should give exactly 30% of moving puck reward."""
        obs_stationary = create_obs(player_pos=(0, 0), puck_pos=(3, 0), puck_vel=(0, 0))
        obs_moving = create_obs(player_pos=(0, 0), puck_pos=(3, 0), puck_vel=(1, 0))

        phi_stationary = compute_potential(obs_stationary)
        phi_moving = compute_potential(obs_moving)

        ratio = phi_stationary / phi_moving
        assert abs(ratio - STATIONARY_PUCK_WEIGHT) < 1e-6, \
            f"Ratio should be {STATIONARY_PUCK_WEIGHT}, got {ratio}"


class TestDoNothingExploit:
    """Test that the 'stand and wait' exploit is prevented via reduced magnitude."""

    def test_reduced_reward_for_stationary_puck(self):
        """Stationary puck gives reduced (30%) reward - guides approach but not exploit."""
        # Puck spawns at (-2, 0), agent at (-2.1, 0)
        obs = create_obs(player_pos=(-2.1, 0), puck_pos=(-2, 0), puck_vel=(0, 0))
        components = get_potential_components(obs)

        # Should have reduced (30%) chase reward, not zero
        assert components['phi_chase'] < 0, "Should have some negative phi (close to puck)"
        assert components['weight_applied'] == STATIONARY_PUCK_WEIGHT, "Should use reduced weight"

    def test_hitting_puck_increases_chase_signal(self):
        """After hitting puck (it moves), chase signal increases (full weight)."""
        # Before hit: puck stationary
        obs_before = create_obs(player_pos=(-2.1, 0), puck_pos=(-2, 0), puck_vel=(0, 0))
        # After hit: puck moving
        obs_after = create_obs(player_pos=(-2.1, 0), puck_pos=(-1.5, 0), puck_vel=(2, 0))

        comp_before = get_potential_components(obs_before)
        comp_after = get_potential_components(obs_after)

        assert comp_before['weight_applied'] == STATIONARY_PUCK_WEIGHT, "Before = reduced"
        assert comp_after['weight_applied'] == 1.0, "After = full"
        # After hitting, the signal is stronger (full weight) even though distance changed
        assert comp_after['phi_chase'] < comp_before['phi_chase'], "Moving puck = stronger signal"

    def test_standing_still_no_pbrs_accumulation(self):
        """Standing still gives zero PBRS (F = γφ' - φ = 0 when φ' = φ)."""
        # Same state twice (standing still, puck not moving)
        obs = create_obs(player_pos=(-2.1, 0), puck_pos=(-2, 0), puck_vel=(0, 0))

        phi = compute_potential(obs)
        pbrs = compute_pbrs(obs, obs, done=False)

        # phi is non-zero (reduced weight), but PBRS is 0 for same state
        assert phi < 0, f"Potential should be negative (close to puck), got {phi}"
        assert abs(pbrs) < 1e-6, f"PBRS should be ~0 when nothing changes, got {pbrs}"

    def test_approaching_stationary_puck_gives_positive_pbrs(self):
        """Moving TOWARD stationary puck should give positive PBRS."""
        # Agent starts far, moves closer to stationary puck
        obs_far = create_obs(player_pos=(-4, 0), puck_pos=(-2, 0), puck_vel=(0, 0))
        obs_close = create_obs(player_pos=(-2.5, 0), puck_pos=(-2, 0), puck_vel=(0, 0))

        pbrs = compute_pbrs(obs_far, obs_close, done=False)

        # PBRS should be positive (rewarding approach)
        assert pbrs > 0, f"Approaching puck should give positive PBRS, got {pbrs}"
        # But it should be small (reduced weight)
        assert pbrs < 0.5, f"PBRS should be small (reduced weight), got {pbrs}"

    def test_walk_to_puck_total_reward_small(self):
        """Walking to stationary puck should give small total PBRS (not exploit-worthy)."""
        # Agent walks from far corner to puck
        obs_start = create_obs(player_pos=(-4, 2), puck_pos=(-2, 0), puck_vel=(0, 0))
        obs_end = create_obs(player_pos=(-2.1, 0), puck_pos=(-2, 0), puck_vel=(0, 0))

        phi_start = compute_potential(obs_start)
        phi_end = compute_potential(obs_end)

        # Total PBRS for walking to puck (telescoping sum)
        # F_total ≈ γ*phi_end - phi_start (ignoring intermediate steps)
        total_pbrs = 0.99 * phi_end - phi_start

        print(f"phi_start: {phi_start}, phi_end: {phi_end}")
        print(f"Total PBRS for walking to stationary puck: {total_pbrs}")

        # With pbrs_scale=0.02, this becomes ~0.02 * total_pbrs
        scaled_pbrs = 0.02 * total_pbrs
        print(f"Scaled PBRS: {scaled_pbrs}")

        # Should be positive but small (not worth exploiting vs sparse ±10)
        assert total_pbrs > 0, "Walking toward puck should be positive"
        assert scaled_pbrs < 0.3, f"Scaled PBRS should be small, got {scaled_pbrs}"


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
        # Worst case: far from moving puck
        obs_worst = create_obs(player_pos=(4, 2), puck_pos=(-4, 0), puck_vel=(1, 0))
        # Best case: on top of moving puck
        obs_best = create_obs(player_pos=(0, 0), puck_pos=(0.1, 0), puck_vel=(1, 0))

        phi_worst = compute_potential(obs_worst)
        phi_best = compute_potential(obs_best)

        print(f"phi_worst: {phi_worst}")
        print(f"phi_best: {phi_best}")
        print(f"Range: {phi_best - phi_worst}")

        # φ_chase only: range is [-100, 0]
        assert phi_worst < -80, f"Worst case should be < -80, got {phi_worst}"
        assert phi_best > -5, f"Best case should be > -5, got {phi_best}"
        assert phi_best > phi_worst, "Best should be better than worst"

    def test_episode_shaping_less_than_sparse(self):
        """Total episode PBRS should be less than sparse reward magnitude."""
        # Max potential change (φ_chase only now)
        obs_worst = create_obs(player_pos=(4, 2), puck_pos=(-4, 0), puck_vel=(1, 0))
        obs_best = create_obs(player_pos=(0, 0), puck_pos=(0.1, 0), puck_vel=(1, 0))

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


class TestIndependentAnnealing:
    """Test the new independent annealing functionality."""

    def test_no_annealing_when_disabled(self):
        """When anneal_start=0, no annealing should happen."""
        shaper = PBRSReward(pbrs_scale=0.03, anneal_start=0)

        assert shaper.get_annealing_weight(episode=0) == 1.0
        assert shaper.get_annealing_weight(episode=5000) == 1.0
        assert shaper.get_annealing_weight(episode=100000) == 1.0

    def test_annealing_before_start(self):
        """Before anneal_start, weight should be 1.0."""
        shaper = PBRSReward(pbrs_scale=0.03, anneal_start=3000, anneal_episodes=3000)

        assert shaper.get_annealing_weight(episode=0) == 1.0
        assert shaper.get_annealing_weight(episode=1000) == 1.0
        assert shaper.get_annealing_weight(episode=2999) == 1.0

    def test_annealing_at_start(self):
        """At exactly anneal_start, weight should be 1.0."""
        shaper = PBRSReward(pbrs_scale=0.03, anneal_start=3000, anneal_episodes=3000)

        # At start, we're 0 episodes into annealing, so weight = 1.0
        weight = shaper.get_annealing_weight(episode=3000)
        assert weight == 1.0, f"At anneal_start, weight should be 1.0, got {weight}"

    def test_annealing_midpoint(self):
        """At midpoint, weight should be ~0.5."""
        shaper = PBRSReward(pbrs_scale=0.03, anneal_start=3000, anneal_episodes=3000)

        # Midpoint: episode 4500 = 1500 into 3000 duration = 50%
        weight = shaper.get_annealing_weight(episode=4500)
        assert abs(weight - 0.5) < 0.01, f"At midpoint, weight should be ~0.5, got {weight}"

    def test_annealing_at_end(self):
        """At anneal_start + anneal_episodes, weight should be 0."""
        shaper = PBRSReward(pbrs_scale=0.03, anneal_start=3000, anneal_episodes=3000)

        weight = shaper.get_annealing_weight(episode=6000)
        assert weight == 0.0, f"At end of annealing, weight should be 0.0, got {weight}"

    def test_annealing_after_end(self):
        """After annealing completes, weight should remain 0."""
        shaper = PBRSReward(pbrs_scale=0.03, anneal_start=3000, anneal_episodes=3000)

        assert shaper.get_annealing_weight(episode=7000) == 0.0
        assert shaper.get_annealing_weight(episode=100000) == 0.0

    def test_annealing_affects_reward(self):
        """Annealing should scale the actual reward."""
        shaper = PBRSReward(pbrs_scale=0.03, anneal_start=3000, anneal_episodes=3000)
        obs = create_obs(player_pos=(0, 0), puck_pos=(3, 0), puck_vel=(1, 0))
        obs_next = create_obs(player_pos=(1, 0), puck_pos=(3, 0), puck_vel=(1, 0))

        # Before annealing: full reward
        reward_before = shaper.compute(obs, obs_next, done=False, episode=2000)

        # At midpoint: half reward
        reward_mid = shaper.compute(obs, obs_next, done=False, episode=4500)

        # After annealing: zero reward
        reward_after = shaper.compute(obs, obs_next, done=False, episode=7000)

        print(f"Reward before: {reward_before}")
        print(f"Reward mid: {reward_mid}")
        print(f"Reward after: {reward_after}")

        assert abs(reward_mid - reward_before * 0.5) < 0.001, \
            f"Midpoint reward should be half of full reward"
        assert reward_after == 0.0, \
            f"After annealing, reward should be 0, got {reward_after}"

    def test_independent_annealing_priority(self):
        """Independent annealing should take priority over legacy self-play annealing."""
        # Set up both independent and legacy annealing
        shaper = PBRSReward(
            pbrs_scale=0.03,
            anneal_start=1000,       # Independent: start at 1000
            anneal_episodes=1000,    # Independent: end at 2000
            constant_weight=False,   # Legacy: enable
            annealing_episodes=5000  # Legacy: would take longer
        )
        shaper.set_self_play_start(500)  # Legacy: would start at 500

        # Independent annealing should be used (starts at 1000, not 500)
        assert shaper.get_annealing_weight(episode=750) == 1.0, "Before independent start"
        assert shaper.get_annealing_weight(episode=1500) == 0.5, "Midpoint of independent"
        assert shaper.get_annealing_weight(episode=2500) == 0.0, "After independent end"


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

    def test_no_defensive_positioning_reward(self):
        """There should be NO defensive positioning reward - only chase component."""
        # Puck in own half - used to trigger defensive component
        obs = create_obs(player_pos=(0, 0), puck_pos=(-2, 0), puck_vel=(0, 0))
        components = get_potential_components(obs)

        # With puck stationary, phi should be reduced (30%) chase only, no defensive
        assert components['weight_applied'] == STATIONARY_PUCK_WEIGHT, "Should use reduced weight"
        # phi_chase and phi_total should be equal (no separate defensive component)
        assert components['phi_chase'] == components['phi_total'], \
            "Only chase component should exist (no defensive)"


def run_all_tests():
    """Run all tests without pytest."""
    import traceback

    test_classes = [
        TestChaseComponent,
        TestDoNothingExploit,
        TestTerminalPotential,
        TestScaling,
        TestPBRSRewardClass,
        TestIndependentAnnealing,
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
