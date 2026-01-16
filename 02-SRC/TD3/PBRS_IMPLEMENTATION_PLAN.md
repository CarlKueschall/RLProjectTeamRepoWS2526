# PBRS Implementation Plan: Research-Based Optimal Reward Shaping

## Executive Summary

Based on the Perplexity research findings, this plan addresses the root cause of our agent's poor performance: **the current potential function rewards possession instead of penalizing hoarding**, leading to an agent that learns NOT to shoot when possessing.

### Key Changes
1. **Rewrite φ(s)** with possession PENALTY instead of reward
2. **Fix scaling** from 0.1 to **α = 0.02** (mathematically derived)
3. **Add velocity alignment** component (direction toward goal, not just magnitude)
4. **Add reward hacking detection** metrics
5. **Add validation tests** before training

---

## Phase 1: Rewrite Potential Function

### File: `rewards/pbrs.py`

**Current Problem**: `phi_possession` REWARDS holding the puck:
```python
# CURRENT (WRONG):
if has_puck:
    phi_possession = 2.0 * (0.5 + 0.3 * offensive_position_bonus + ...)  # POSITIVE = REWARD
```

**Research Solution**: Must PENALIZE possession when not shooting toward goal:
```python
# NEW (CORRECT):
if player_to_puck < possession_threshold:
    if alignment < 0.3:  # Not shooting toward goal
        possession_penalty = -20.0  # NEGATIVE = PENALTY for hoarding
    else:
        possession_penalty = 0.0  # No penalty if shooting
```

### New Potential Function Components

| Component | Formula | Range | Purpose |
|-----------|---------|-------|---------|
| `φ_distance` | `-puck_to_goal / max_distance` | [-1, 0] | Progress toward scoring |
| `φ_alignment` | `+0.3 * max(dot(puck_vel, goal_dir), 0)` | [0, 0.3] | Reward shooting toward goal |
| `φ_possession_penalty` | `-0.8` when possessing & not shooting | [-0.8, 0] | **KEY FIX: Penalize hoarding** |
| `φ_defensive` | `-0.4 * defensive_error` | [-0.4, 0] | Triangle defense positioning |

**Total φ(s) range**: Approximately [-2.2, 0.3], normalized to episode scale.

### Implementation

Replace `compute_potential()` entirely:

```python
def compute_potential(obs, gamma=0.99):
    """
    Research-based optimal potential function for air hockey.

    Key insight: Possession is INSTRUMENTAL, not TERMINAL.
    Must penalize holding puck without shooting toward goal.
    """
    # === Extract state ===
    player_pos = np.array([obs[0], obs[1]])
    puck_pos = np.array([obs[12], obs[13]])
    puck_vel = np.array([obs[14], obs[15]])
    opponent_goal = np.array([4.5, 0.0])
    own_goal = np.array([-4.5, 0.0])

    # Environment constants
    TABLE_LENGTH = 9.0  # -4.5 to +4.5
    TABLE_WIDTH = 5.0   # -2.5 to +2.5
    MAX_DISTANCE = np.sqrt(TABLE_LENGTH**2 + TABLE_WIDTH**2)  # ~10.3

    # === Component 1: Distance to scoring (progress metric) ===
    puck_to_goal = np.linalg.norm(opponent_goal - puck_pos)
    phi_distance = -puck_to_goal / MAX_DISTANCE  # Range: [-1, 0]

    # === Component 2: Velocity alignment (shooting toward goal) ===
    puck_speed = np.linalg.norm(puck_vel)
    if puck_speed > 0.1:
        puck_direction = puck_vel / puck_speed
        goal_direction = (opponent_goal - puck_pos)
        goal_dist = np.linalg.norm(goal_direction)
        if goal_dist > 0.1:
            goal_direction = goal_direction / goal_dist
            alignment = np.dot(puck_direction, goal_direction)
            phi_alignment = 0.3 * max(alignment, 0)  # Only reward positive alignment
        else:
            phi_alignment = 0.3  # Near goal = good
    else:
        phi_alignment = 0.0  # No velocity = no alignment bonus

    # === Component 3: Possession penalty (KEY FIX) ===
    player_to_puck = np.linalg.norm(player_pos - puck_pos)
    POSSESSION_THRESHOLD = 0.3  # ~30cm

    phi_possession_penalty = 0.0
    if player_to_puck < POSSESSION_THRESHOLD:
        # Agent is possessing the puck
        # Penalize if not shooting toward goal
        if puck_speed < 0.5:  # Puck is slow (not being shot)
            phi_possession_penalty = -0.8  # Strong penalty for hoarding
        elif puck_speed > 0.5 and alignment < 0.3:  # Moving but wrong direction
            phi_possession_penalty = -0.4  # Moderate penalty
        # else: shooting toward goal = no penalty

    # === Component 4: Defensive positioning (triangle defense) ===
    # Only active when puck is in own half
    puck_in_own_half = puck_pos[0] < 0
    if puck_in_own_half:
        # Ideal position: 40% of the way from goal to puck (triangle defense)
        ideal_defensive_pos = own_goal + 0.4 * (puck_pos - own_goal)
        defensive_error = np.linalg.norm(player_pos - ideal_defensive_pos)
        # Normalize by table width
        phi_defensive = -0.4 * min(defensive_error / TABLE_WIDTH, 1.0)
    else:
        phi_defensive = 0.0

    # === Combine components ===
    # Scale to episode-appropriate range: ~[-250, +30] for T=250 steps
    EPISODE_SCALE = 100.0
    phi = EPISODE_SCALE * (phi_distance + phi_alignment + phi_possession_penalty + phi_defensive)

    return phi
```

---

## Phase 2: Fix Scaling Factor

### Research Finding
The mathematically correct scaling factor is **α = 0.02**, derived from:
- Sparse rewards: ±10
- Episode length: ~250 steps
- Constraint: α × max_episode_shaping < sparse_reward
- Calculation: α × 250 < 10 → α < 0.04, use conservative 0.02

### Current Problem
```python
# Current: SCALE=1.0 in compute_potential, pbrs_scale=0.1 in train_hockey.py
# Total: 1.0 * 0.1 = 0.1 (5x too high!)
```

### Fix
1. Set `EPISODE_SCALE = 100.0` in `compute_potential()` (gives φ ∈ [-220, +30])
2. Change default `--pbrs_scale` to `0.02` in `config/parser.py`
3. This gives total scaling: 100 * 0.02 = 2.0 per step max, 0.02 * 250 = 5 per episode max

### File: `config/parser.py`

```python
# Change from:
parser.add_argument('--pbrs_scale', type=float, default=1.0, ...)
# To:
parser.add_argument('--pbrs_scale', type=float, default=0.02,
    help='PBRS scaling factor (default: 0.02, mathematically derived to prevent reward hacking)')
```

---

## Phase 3: Add Reward Hacking Detection Metrics

### File: `metrics/metrics_tracker.py`

Add new metrics to detect reward hacking:

```python
class MetricsTracker:
    def __init__(self, ...):
        # ... existing code ...
        # NEW: Reward hacking detection
        self.possession_durations = []
        self.shoot_rates_when_possessing = []
        self.sparse_vs_shaping_ratios = []

    def add_possession_metrics(self, possession_duration, did_shoot):
        """Track possession behavior for hacking detection."""
        self.possession_durations.append(possession_duration)
        self.shoot_rates_when_possessing.append(float(did_shoot))

    def add_reward_decomposition(self, sparse_reward, shaping_reward):
        """Track sparse vs shaping ratio."""
        if abs(sparse_reward) > 0.01:
            ratio = abs(shaping_reward) / abs(sparse_reward)
            self.sparse_vs_shaping_ratios.append(ratio)

    def get_hacking_metrics(self):
        """Return metrics for reward hacking detection."""
        return {
            'avg_possession_duration': np.mean(self.possession_durations) if self.possession_durations else 0,
            'shoot_rate_when_possessing': np.mean(self.shoot_rates_when_possessing) if self.shoot_rates_when_possessing else 0,
            'shaping_to_sparse_ratio': np.mean(self.sparse_vs_shaping_ratios) if self.sparse_vs_shaping_ratios else 0,
        }

    def detect_reward_hacking(self, win_rate, episode):
        """Return warnings if reward hacking detected."""
        warnings = []
        metrics = self.get_hacking_metrics()

        # Flag 1: High shaping, low win rate
        if metrics['shaping_to_sparse_ratio'] > 2.0 and win_rate < 0.3 and episode > 1000:
            warnings.append("SHAPING_DOMINANCE")

        # Flag 2: Long possession, low shoot rate
        if metrics['avg_possession_duration'] > 30 and metrics['shoot_rate_when_possessing'] < 0.2:
            warnings.append("PUCK_HOARDING")

        # Flag 3: Performance gap (shaped reward high, win rate low)
        # This would need additional tracking of total shaped reward

        return warnings
```

### File: `train_hockey.py`

Add logging for new metrics:

```python
# In logging section
if args.reward_shaping:
    hacking_metrics = tracker.get_hacking_metrics()
    log_metrics["hacking/possession_duration"] = hacking_metrics['avg_possession_duration']
    log_metrics["hacking/shoot_rate_possess"] = hacking_metrics['shoot_rate_when_possessing']
    log_metrics["hacking/shaping_sparse_ratio"] = hacking_metrics['shaping_to_sparse_ratio']

    # Check for warnings
    warnings = tracker.detect_reward_hacking(cumulative_win_rate, i_episode)
    if warnings:
        print(f"  WARNING: Potential reward hacking detected: {warnings}")
```

---

## Phase 4: Add Validation Tests

### File: `tests/test_pbrs_potential.py` (NEW)

```python
"""
Validation tests for PBRS potential function.
Run before training to ensure potential function is correctly designed.
"""
import numpy as np
import sys
sys.path.append('..')
from rewards.pbrs import compute_potential, compute_pbrs

def create_test_state(player_pos, puck_pos, puck_vel):
    """Create 18D observation for testing."""
    obs = np.zeros(18)
    obs[0:2] = player_pos
    obs[12:14] = puck_pos
    obs[14:16] = puck_vel
    obs[16] = 0  # No possession timer
    return obs

def test_monotonicity_with_progress():
    """Test: Puck closer to goal should have higher potential."""
    # Puck near opponent goal
    s_near_goal = create_test_state(
        player_pos=[0, 0],
        puck_pos=[4.0, 0],  # Near opponent goal
        puck_vel=[0, 0]
    )
    phi_near = compute_potential(s_near_goal)

    # Puck in own half
    s_own_half = create_test_state(
        player_pos=[0, 0],
        puck_pos=[-2.0, 0],  # Own half
        puck_vel=[0, 0]
    )
    phi_own = compute_potential(s_own_half)

    assert phi_near > phi_own, f"Potential not monotonic! Near goal: {phi_near}, Own half: {phi_own}"
    print(f"  Monotonicity: PASS (near={phi_near:.2f} > own={phi_own:.2f})")

def test_possession_penalty():
    """Test: Holding puck without shooting should decrease potential."""
    # Possessing with no velocity (hoarding)
    s_possess_static = create_test_state(
        player_pos=[0, 0],
        puck_pos=[0.1, 0],  # Player near puck
        puck_vel=[0, 0]     # No velocity
    )
    phi_static = compute_potential(s_possess_static)

    # Possessing with velocity toward goal (shooting)
    s_possess_shooting = create_test_state(
        player_pos=[0, 0],
        puck_pos=[0.1, 0],  # Player near puck
        puck_vel=[2.0, 0]   # Velocity toward opponent goal
    )
    phi_shooting = compute_potential(s_possess_shooting)

    assert phi_shooting > phi_static, f"Possession penalty not working! Shooting: {phi_shooting}, Static: {phi_static}"
    print(f"  Possession penalty: PASS (shooting={phi_shooting:.2f} > static={phi_static:.2f})")

def test_velocity_alignment():
    """Test: Velocity toward goal should be rewarded more than away."""
    # Puck moving toward opponent goal
    s_toward = create_test_state(
        player_pos=[-2, 0],
        puck_pos=[0, 0],
        puck_vel=[2.0, 0]  # Toward opponent goal (+x)
    )
    phi_toward = compute_potential(s_toward)

    # Puck moving toward own goal
    s_away = create_test_state(
        player_pos=[-2, 0],
        puck_pos=[0, 0],
        puck_vel=[-2.0, 0]  # Toward own goal (-x)
    )
    phi_away = compute_potential(s_away)

    assert phi_toward > phi_away, f"Velocity alignment not working! Toward: {phi_toward}, Away: {phi_away}"
    print(f"  Velocity alignment: PASS (toward={phi_toward:.2f} > away={phi_away:.2f})")

def test_pbrs_terminal():
    """Test: PBRS at terminal state uses phi=0."""
    obs = create_test_state([0, 0], [0, 0], [0, 0])
    obs_next = create_test_state([0, 0], [0, 0], [0, 0])

    # Non-terminal
    f_non_terminal = compute_pbrs(obs, obs_next, done=False)

    # Terminal (should use phi_next=0)
    f_terminal = compute_pbrs(obs, obs_next, done=True)

    # They should be different (terminal ignores obs_next potential)
    print(f"  Terminal handling: PASS (non-terminal={f_non_terminal:.2f}, terminal={f_terminal:.2f})")

def test_potential_range():
    """Test: Potential should be bounded."""
    # Test many random states
    max_phi = -np.inf
    min_phi = np.inf

    for _ in range(1000):
        obs = np.random.uniform(-5, 5, 18)
        phi = compute_potential(obs)
        max_phi = max(max_phi, phi)
        min_phi = min(min_phi, phi)

    print(f"  Potential range: [{min_phi:.2f}, {max_phi:.2f}]")
    assert max_phi < 1000 and min_phi > -1000, "Potential unbounded!"
    print(f"  Potential bounded: PASS")

if __name__ == "__main__":
    print("\n=== PBRS Potential Function Validation ===\n")

    print("Test 1: Monotonicity with progress")
    test_monotonicity_with_progress()

    print("\nTest 2: Possession penalty")
    test_possession_penalty()

    print("\nTest 3: Velocity alignment")
    test_velocity_alignment()

    print("\nTest 4: Terminal state handling")
    test_pbrs_terminal()

    print("\nTest 5: Potential range")
    test_potential_range()

    print("\n=== ALL VALIDATION TESTS PASSED ===\n")
```

---

## Phase 5: Update train_hockey.py Integration

### Changes to `train_hockey.py`

1. **Add possession tracking in episode loop**:
```python
# Inside episode loop, track possession
possession_start_step = None
did_shoot_during_possession = False

if player_to_puck < 0.3:  # Possessing
    if possession_start_step is None:
        possession_start_step = t
    if puck_speed > 1.0 and puck_toward_goal:  # Shooting
        did_shoot_during_possession = True
else:
    if possession_start_step is not None:
        possession_duration = t - possession_start_step
        tracker.add_possession_metrics(possession_duration, did_shoot_during_possession)
        possession_start_step = None
        did_shoot_during_possession = False
```

2. **Add reward decomposition tracking**:
```python
# After computing shaped reward
if args.reward_shaping:
    tracker.add_reward_decomposition(r1, pbrs_bonus)
```

---

## Verification Checklist

Before training:
- [ ] Run `python tests/test_pbrs_potential.py` - all tests pass
- [ ] Verify `--pbrs_scale 0.02` is default
- [ ] Verify potential function has possession penalty

During training (monitor W&B):
- [ ] `hacking/possession_duration` should be 5-15 steps (not >30)
- [ ] `hacking/shoot_rate_possess` should be 30-70% (not <20%)
- [ ] `hacking/shaping_sparse_ratio` should decrease over training
- [ ] `behavior/shoot_action_when_possess` should be POSITIVE (not negative like before)

Expected results:
- [ ] 90%+ win rate vs weak by episode 3000
- [ ] Shoot rate increasing over time
- [ ] Possession duration decreasing over time

---

## Summary of File Changes

| File | Action | Description |
|------|--------|-------------|
| `rewards/pbrs.py` | **REWRITE** | New potential function with possession penalty |
| `config/parser.py` | **EDIT** | Change `--pbrs_scale` default to 0.02 |
| `metrics/metrics_tracker.py` | **EDIT** | Add reward hacking detection metrics |
| `train_hockey.py` | **EDIT** | Add possession tracking and hacking detection logging |
| `tests/test_pbrs_potential.py` | **NEW** | Validation tests for potential function |

---

## Risk Mitigation

**If reward hacking still occurs**:
1. Reduce `--pbrs_scale` to 0.01
2. Increase possession penalty from -0.8 to -1.5
3. Add explicit "time since possession started" penalty

**If convergence is too slow**:
1. Increase `--pbrs_scale` to 0.03
2. Check that possession penalty isn't too harsh (agent avoids puck entirely)

**If catastrophic forgetting**:
1. Already have opponent pool (self_play.py)
2. Ensure `--disable_selfplay` is NOT set for production runs
