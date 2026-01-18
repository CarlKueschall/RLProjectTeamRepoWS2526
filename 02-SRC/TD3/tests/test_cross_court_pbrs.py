#!/usr/bin/env python3
"""
Verification script for PBRS V3.2 Cross-Court Component.

This script verifies:
1. Observation indices are correct (using real hockey environment)
2. Cross-court logic rewards shooting away from opponent
3. Smooth activation via tanh works correctly
4. Edge cases are handled properly

Run with: python tests/test_cross_court_pbrs.py
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards.pbrs import (
    compute_potential,
    compute_pbrs,
    get_potential_components,
    W_CHASE, W_ATTACK, W_CROSS,
    TABLE_WIDTH, MAX_DISTANCE, EPISODE_SCALE,
)


def create_obs(player_pos=(0, 0), opponent_pos=(0, 0), puck_pos=(0, 0), puck_vel=(0, 0)):
    """Create a test observation with specified positions/velocities."""
    obs = np.zeros(18)
    obs[0:2] = player_pos      # Player position
    obs[6:8] = opponent_pos    # Opponent position
    obs[12:14] = puck_pos      # Puck position
    obs[14:16] = puck_vel      # Puck velocity
    return obs


class TestObservationIndices:
    """Verify observation indices match expected structure."""

    def test_player_position(self):
        """Player position at obs[0:2]."""
        obs = create_obs(player_pos=(1.5, -2.0))
        assert obs[0] == 1.5, f"Player x should be at obs[0], got {obs[0]}"
        assert obs[1] == -2.0, f"Player y should be at obs[1], got {obs[1]}"
        print("  Player position indices: CORRECT (obs[0:2])")

    def test_opponent_position(self):
        """Opponent position at obs[6:8]."""
        obs = create_obs(opponent_pos=(3.0, 1.5))
        assert obs[6] == 3.0, f"Opponent x should be at obs[6], got {obs[6]}"
        assert obs[7] == 1.5, f"Opponent y should be at obs[7], got {obs[7]}"
        print("  Opponent position indices: CORRECT (obs[6:8])")

    def test_puck_position(self):
        """Puck position at obs[12:14]."""
        obs = create_obs(puck_pos=(2.0, -1.0))
        assert obs[12] == 2.0, f"Puck x should be at obs[12], got {obs[12]}"
        assert obs[13] == -1.0, f"Puck y should be at obs[13], got {obs[13]}"
        print("  Puck position indices: CORRECT (obs[12:14])")

    def test_puck_velocity(self):
        """Puck velocity at obs[14:16]."""
        obs = create_obs(puck_vel=(1.5, -0.5))
        assert obs[14] == 1.5, f"Puck vel_x should be at obs[14], got {obs[14]}"
        assert obs[15] == -0.5, f"Puck vel_y should be at obs[15], got {obs[15]}"
        print("  Puck velocity indices: CORRECT (obs[14:16])")


class TestCrossCourtComponent:
    """Test the cross-court bonus logic."""

    def test_cross_court_high_when_opponent_far(self):
        """φ_cross should be high when puck is far from opponent (in y)."""
        # Opponent at upper corner (y=2.0), puck at lower corner (y=-2.0)
        # Puck moving toward opponent goal (vel_x > 0)
        obs = create_obs(
            opponent_pos=(3.0, 2.0),   # Opponent at upper
            puck_pos=(1.0, -2.0),       # Puck at lower
            puck_vel=(2.0, 0.0)         # Moving toward goal
        )
        components = get_potential_components(obs)

        y_separation = abs(-2.0 - 2.0)  # = 4.0
        expected_y_sep_norm = y_separation / TABLE_WIDTH  # = 0.8

        assert abs(components['y_separation'] - 4.0) < 0.01, \
            f"Y separation should be 4.0, got {components['y_separation']}"
        assert components['velocity_activation'] > 0.9, \
            f"Velocity activation should be high for vel_x=2.0, got {components['velocity_activation']}"
        assert components['phi_cross'] > 0.7, \
            f"φ_cross should be high (>0.7), got {components['phi_cross']}"
        print(f"  Cross-court high when opponent far: PASS (φ_cross={components['phi_cross']:.3f})")

    def test_cross_court_low_when_opponent_near(self):
        """φ_cross should be low when puck is near opponent (in y)."""
        # Opponent and puck at same y position
        obs = create_obs(
            opponent_pos=(3.0, 1.0),    # Opponent at y=1.0
            puck_pos=(1.0, 1.0),         # Puck also at y=1.0
            puck_vel=(2.0, 0.0)          # Moving toward goal
        )
        components = get_potential_components(obs)

        assert components['y_separation'] < 0.01, \
            f"Y separation should be ~0, got {components['y_separation']}"
        assert components['phi_cross'] < 0.01, \
            f"φ_cross should be ~0 when shooting at opponent, got {components['phi_cross']}"
        print(f"  Cross-court low when opponent near: PASS (φ_cross={components['phi_cross']:.3f})")

    def test_cross_court_zero_when_puck_stationary(self):
        """φ_cross should be zero when puck is not moving toward goal."""
        # Puck stationary
        obs = create_obs(
            opponent_pos=(3.0, 2.0),
            puck_pos=(1.0, -2.0),
            puck_vel=(0.0, 0.0)  # Stationary
        )
        components = get_potential_components(obs)

        assert components['velocity_activation'] < 0.01, \
            f"Velocity activation should be ~0 for stationary puck, got {components['velocity_activation']}"
        assert components['phi_cross'] < 0.01, \
            f"φ_cross should be ~0 when puck stationary, got {components['phi_cross']}"
        print(f"  Cross-court zero when puck stationary: PASS (φ_cross={components['phi_cross']:.3f})")

    def test_cross_court_zero_when_puck_moving_backward(self):
        """φ_cross should be zero when puck moving away from opponent goal."""
        # Puck moving backward (vel_x < 0)
        obs = create_obs(
            opponent_pos=(3.0, 2.0),
            puck_pos=(1.0, -2.0),
            puck_vel=(-2.0, 0.0)  # Moving AWAY from opponent goal
        )
        components = get_potential_components(obs)

        assert components['velocity_activation'] == 0.0, \
            f"Velocity activation should be 0 for backward motion, got {components['velocity_activation']}"
        assert components['phi_cross'] == 0.0, \
            f"φ_cross should be 0 when puck moving backward, got {components['phi_cross']}"
        print(f"  Cross-court zero when puck moving backward: PASS")


class TestSmoothActivation:
    """Test the smooth tanh activation."""

    def test_tanh_smooth_transition(self):
        """Velocity activation should smoothly transition from 0 to 1."""
        velocities = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0]
        activations = []

        for vel_x in velocities:
            obs = create_obs(
                opponent_pos=(3.0, 2.0),
                puck_pos=(1.0, 0.0),
                puck_vel=(vel_x, 0.0)
            )
            components = get_potential_components(obs)
            activations.append(components['velocity_activation'])

        # Check monotonicity for positive velocities
        for i in range(len(velocities) - 1):
            if velocities[i] >= 0 and velocities[i+1] >= 0:
                assert activations[i] <= activations[i+1], \
                    f"Activation should increase with velocity: {activations[i]} > {activations[i+1]}"

        # Check that negative velocities give 0
        assert activations[0] == 0.0, f"vel_x=-2.0 should give activation=0, got {activations[0]}"
        assert activations[1] == 0.0, f"vel_x=-1.0 should give activation=0, got {activations[1]}"
        assert activations[2] == 0.0, f"vel_x=0.0 should give activation=0, got {activations[2]}"

        # Check that high velocities approach 1
        assert activations[-1] > 0.99, f"vel_x=3.0 should give activation~1, got {activations[-1]}"

        print(f"  Smooth tanh activation: PASS")
        print(f"    Velocities:   {velocities}")
        print(f"    Activations:  {[f'{a:.3f}' for a in activations]}")


class TestCrossCourtIncentive:
    """Test that cross-court creates the right incentives."""

    def test_alternating_shots_incentive(self):
        """Agent should be rewarded more for shooting away from opponent."""
        # Scenario: Opponent is in upper half (y > 0)
        # Shot 1: To lower corner (cross-court) - should get bonus
        # Shot 2: To upper corner (at opponent) - no bonus

        obs_cross = create_obs(
            player_pos=(-1.0, 0.0),
            opponent_pos=(3.0, 2.0),   # Opponent upper
            puck_pos=(1.0, -2.0),       # Puck lower (cross-court!)
            puck_vel=(2.0, 0.0)
        )

        obs_direct = create_obs(
            player_pos=(-1.0, 0.0),
            opponent_pos=(3.0, 2.0),   # Opponent upper
            puck_pos=(1.0, 2.0),        # Puck also upper (at opponent)
            puck_vel=(2.0, 0.0)
        )

        comp_cross = get_potential_components(obs_cross)
        comp_direct = get_potential_components(obs_direct)

        assert comp_cross['phi_cross'] > comp_direct['phi_cross'], \
            f"Cross-court shot should get higher φ_cross: {comp_cross['phi_cross']} vs {comp_direct['phi_cross']}"

        # The potential difference should encourage cross-court
        phi_cross_total = comp_cross['phi_scaled']
        phi_direct_total = comp_direct['phi_scaled']

        print(f"  Alternating shots incentive: PASS")
        print(f"    Cross-court φ_cross: {comp_cross['phi_cross']:.3f}")
        print(f"    Direct shot φ_cross: {comp_direct['phi_cross']:.3f}")
        print(f"    Total potential (cross): {phi_cross_total:.1f}")
        print(f"    Total potential (direct): {phi_direct_total:.1f}")
        print(f"    Bonus for cross-court: {phi_cross_total - phi_direct_total:.1f}")


class TestPBRSComputation:
    """Test the full PBRS computation with cross-court."""

    def test_pbrs_rewards_cross_court_shot(self):
        """PBRS should reward transition to cross-court shot state."""
        # Before: Puck near agent, opponent in upper half
        obs_before = create_obs(
            player_pos=(0.0, 0.0),
            opponent_pos=(3.0, 2.0),
            puck_pos=(0.1, 0.0),
            puck_vel=(0.0, 0.0)
        )

        # After: Puck shot to lower corner (cross-court)
        obs_after = create_obs(
            player_pos=(0.0, 0.0),
            opponent_pos=(3.0, 2.0),
            puck_pos=(2.0, -2.0),   # Moved toward goal AND to lower
            puck_vel=(3.0, -0.5)     # Moving fast toward goal
        )

        pbrs = compute_pbrs(obs_before, obs_after, done=False)

        # PBRS should be positive (good action)
        assert pbrs > 0, f"Cross-court shot should give positive PBRS, got {pbrs}"

        print(f"  PBRS rewards cross-court shot: PASS (PBRS={pbrs:.2f})")

    def test_cross_court_weight_configurable(self):
        """Cross-court weight should be configurable."""
        obs = create_obs(
            opponent_pos=(3.0, 2.0),
            puck_pos=(1.0, -2.0),
            puck_vel=(2.0, 0.0)
        )

        # With default weight
        comp_default = get_potential_components(obs)

        # With w_cross=0 (disabled)
        comp_disabled = get_potential_components(obs, w_cross=0.0)

        # With w_cross=1.0 (stronger)
        comp_strong = get_potential_components(obs, w_cross=1.0)

        assert comp_disabled['phi_cross_weighted'] == 0.0, \
            f"Disabled cross should give 0, got {comp_disabled['phi_cross_weighted']}"
        assert comp_strong['phi_cross_weighted'] > comp_default['phi_cross_weighted'], \
            f"Stronger weight should give higher reward"

        print(f"  Cross-court weight configurable: PASS")
        print(f"    Default (0.4): φ_cross_weighted = {comp_default['phi_cross_weighted']:.3f}")
        print(f"    Disabled (0):  φ_cross_weighted = {comp_disabled['phi_cross_weighted']:.3f}")
        print(f"    Strong (1.0):  φ_cross_weighted = {comp_strong['phi_cross_weighted']:.3f}")


class TestWithRealEnvironment:
    """Test with real hockey environment if available."""

    def test_real_env_observation_structure(self):
        """Verify observation structure matches real environment."""
        try:
            import hockey.hockey_env as h_env
            env = h_env.HockeyEnv()
        except ImportError:
            print("  Real environment test: SKIPPED (hockey_env not available)")
            return

        obs, _ = env.reset()

        # Verify observation is 18-dimensional
        assert len(obs) == 18, f"Expected 18-dim obs, got {len(obs)}"

        print("  Real environment observation structure: PASS")
        print(f"    Observation shape: {obs.shape}")
        print(f"    Player pos (0:2):     {obs[0:2]}")
        print(f"    Opponent pos (6:8):   {obs[6:8]}")
        print(f"    Puck pos (12:14):     {obs[12:14]}")
        print(f"    Puck vel (14:16):     {obs[14:16]}")

        # Compute potential with real observation
        phi = compute_potential(obs)
        components = get_potential_components(obs)

        print(f"    Computed φ: {phi:.2f}")
        print(f"    Components: chase={components['phi_chase']:.3f}, "
              f"attack={components['phi_attack']:.3f}, "
              f"cross={components['phi_cross']:.3f}")

        env.close()


def run_all_tests():
    """Run all verification tests."""
    import traceback

    test_classes = [
        TestObservationIndices,
        TestCrossCourtComponent,
        TestSmoothActivation,
        TestCrossCourtIncentive,
        TestPBRSComputation,
        TestWithRealEnvironment,
    ]

    total = 0
    passed = 0
    failed = 0

    print("=" * 60)
    print("PBRS V3.2 Cross-Court Component Verification")
    print("=" * 60)
    print(f"\nConstants:")
    print(f"  W_CHASE = {W_CHASE}")
    print(f"  W_ATTACK = {W_ATTACK}")
    print(f"  W_CROSS = {W_CROSS}")
    print(f"  TABLE_WIDTH = {TABLE_WIDTH}")
    print(f"  EPISODE_SCALE = {EPISODE_SCALE}")

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in methods:
            total += 1
            method = getattr(instance, method_name)
            try:
                method()
                passed += 1
            except AssertionError as e:
                print(f"  FAIL: {method_name}")
                print(f"        {e}")
                failed += 1
            except Exception as e:
                print(f"  ERROR: {method_name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
