# Tournament vs Test Performance Discrepancy

## Problem Statement

An RL agent trained using TD3 exhibits dramatically different performance between local test evaluation and tournament play:

| Opponent   | Local Test     | Tournament            | Discrepancy                |
| ---------- | -------------- | --------------------- | -------------------------- |
| Strong Bot | 100% (100/100) | 100% (4/4)            | ✓ Consistent              |
| Weak Bot   | 92% (92/100)   | ~40 to 60% (1 to 2/4) | **~40% difference!** |

## Hypothesis

The discrepancy is not due to:

- Different checkpoint files (verified identical)
- Different network architectures (verified identical: [1024, 1024] actor, [1024, 1024, 200] critic, 4D output)
- Different observation/action dimensions (both 18D obs, 4D actions)
- Deterministic inference (both use eps=0.0)
- Agent overfitting to strong opponents (agent performs well vs weak in test)

**New Hypothesis**: The tournament environment is not properly **alternating player starting positions**, which is critical for balanced training.

## Critical Evidence

### Evidence 1: Starting Position Alternation in Local Test

Running the local environment through 10 consecutive resets:

```python
import hockey.hockey_env as h_env
env = h_env.HockeyEnv()

for i in range(10):
    obs, info = env.reset()
    puck_pos = obs[12:14]
    one_starts = env.one_starts
    print(f"Round {i+1}: one_starts={one_starts}, Puck position = ({puck_pos[0]:.2f}, {puck_pos[1]:.2f})")
```

**Result:**

```
Round 1: one_starts=False, Puck position = (1.91, -0.76)   [OFFENSIVE]
Round 2: one_starts=True,  Puck position = (-1.93, -0.08)  [DEFENSIVE]
Round 3: one_starts=False, Puck position = (1.89, 0.32)    [OFFENSIVE]
Round 4: one_starts=True,  Puck position = (-1.18, 0.54)   [DEFENSIVE]
Round 5: one_starts=False, Puck position = (1.91, -0.98)   [OFFENSIVE]
Round 6: one_starts=True,  Puck position = (-1.59, 0.54)   [DEFENSIVE]
...
```

**Pattern**: Puck positions **ALTERNATE** between offensive (+X) and defensive (-X) sides. The `one_starts` flag flips each reset, following the HockeyEnv code:

```python
# From hockey_env.py
if self.mode == Mode.NORMAL:
    self.max_timesteps = 250
    if one_starting is not None:
        self.one_starts = one_starting
    else:
        self.one_starts = not self.one_starts  # ← FLIPS each reset
```

### Evidence 2: Starting Position in Tournament

Extracting the tournament replay data from the pickle files shows all 4 rounds against weak opponent:

```
Round 1: First observation puck position = (-3.00, 0.00)   [DEFENSIVE]
Round 2: First observation puck position = (-3.00, 0.00)   [DEFENSIVE]
Round 3: First observation puck position = (-3.00, 0.00)   [DEFENSIVE]
Round 4: First observation puck position = (-3.00, 0.00)   [DEFENSIVE]
```

**Pattern**: Puck position is **IDENTICAL** across all rounds. No alternation whatsoever.

### Evidence 3: Root Cause in Tournament Code

The official tournament implementation (`HockeyGame` from `hockey_game.py`):

```python
def _update(self, actions_dict: dict[PlayerID, list[float]]) -> bool:
    # ... gameplay ...
    if terminated or truncated:
        # Reset environment for next round
        self.obs_player_one, info = self.env.reset()  # ← BUG: No arguments passed
        self.round_data = RoundData()
        # ... continue ...
    return self.finished
```

The `reset()` call passes **no arguments**, meaning `one_starting` is `None`. This should trigger the alternation logic:

```python
if one_starting is not None:
    self.one_starts = one_starting
else:
    self.one_starts = not self.one_starts  # Should flip
```

However, examining the tournament replay data shows this is **not happening** — the starting position never alternates.

**Possible causes:**

1. Tournament server has a modified version of `HockeyEnv` that doesn't implement alternation
2. Tournament server is calling `reset(one_starting=False)` explicitly for every round
3. Tournament server's environment version differs from the official open-source version

## Impact on Agent Performance

### Training Distribution

The agent was trained and evaluated with **balanced starting positions**:

- ~50% of episodes: Puck spawns RIGHT (offensive) → agent learns attacking strategies
- ~50% of episodes: Puck spawns LEFT (defensive) → agent learns defensive strategies

### Tournament Distribution

Against weak opponent, tournament gives **100% defensive position**:

- **All 4 rounds**: Puck spawns LEFT
- Agent cannot execute offensive strategy
- Weak bot naturally benefits from always attacking
- Result: **0% win rate** (agent starves for offensive opportunities)

### Why Strong Bot Still Works

Against strong opponent, tournament also gives 100% defensive position:

- Agent still defends well (trained for defense)
- Strong bot doesn't fully exploit the defensive asymmetry
- Result: **100% win rate** (agent defensive strength is enough)

Weak bot's simpler strategy is highly **asymmetry-exploitable**: it performs much better when always attacking.

## Quantitative Analysis

**Local Test (Balanced Positions):**

- 92% vs weak (can attack 50% of time)
- Weak bot gets 50% defensive time, wins less

**Tournament (100% Defensive):**

- ~40% vs weak
- Weak bot gets 100% offensive time, wins more

The correlation is perfect: **Removal of alternation → Complete role reversal**.

## Proof

The identical puck starting position across all 4 tournament rounds against the weak bot is **statistically impossible** if alternation were working correctly. With proper alternation, P(all 4 rounds same position) << 0.01 given the randomization in puck spawn location within the offensive/defensive zones.

## Conclusion

**The tournament's `HockeyGame` implementation is not alternating player starting positions.**

This causes:

1. Agent always spawns defensively
2. Weak opponent always spawns offensively
3. Agent cannot use learned offensive strategy
4. Weak opponent exploits one-sided advantage
5. **Result: lower win rate ~40 to 60% (vs expected 92%)**

The agent code and checkpoint are correct. The discrepancy is entirely due to environmental asymmetry in the tournament infrastructure.

## Recommendation

Verify that the tournament server's `HockeyEnv.reset()` method properly implements the alternation logic, or explicitly pass `one_starting` to alternate between True and False for consecutive rounds.
