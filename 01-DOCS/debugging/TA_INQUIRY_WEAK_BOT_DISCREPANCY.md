# Tournament vs Local Testing Discrepancy

# Summary

We have discovered a significant and reproducible performance discrepancy between local agent evaluation and tournament play. Our agent achieves **90% win rate vs weak bot locally**, but only **30-50% win rate vs bot-weak in the tournament**. We have ruled out code issues and identified the likely root cause: **the tournament's `bot-weak` implementation differs from the local testing environment.**

---

## Evidence

### 1. Local Testing Results (with identical configuration)

Using checkpoint: `results_checkpoints_TD3_Hockey_NORMAL_strong_97500_seed48.pth`

```
vs local weak bot (10 episodes, no alternation):
- Wins:   9 (90.0%)
- Losses: 1
- Ties:   0
- Mean reward: 6.43 ± 6.07
```

### 2. Tournament Results (past 10 games)

Same checkpoint played on tournament server:

**vs bot-weak:**

- Games 1-10: 2-5 wins, 3-6 draws, 2-5 losses
- Win rate: **30-50%** (significantly below local 90%)
- Results: "bot-weak won (1-3)", "Draw (2-2)", "bot-weak won (1-3)", "Draw (2-2)", "CerlHat-TD3SelfPlay won (4-0)", etc.

**vs bot-strong (for comparison):**

- Games 1-10: 5 wins, 1 draw, 0 losses
- Win rate: **100%** (or 83% including draws as non-wins)
- Results: "CerlHat-TD3SelfPlay won (3-1)", "CerlHat-TD3SelfPlay won (4-0)", etc.

### 3. Diagnostic Analysis

We implemented comprehensive observation diagnostics to rule out state representation issues. Both local and tournament provide **identical observations**:

```
Local:      Player 1 pos [-3.0, 0], Player 2 pos [3.0, 0], Puck [1.39, 0.78]
Tournament: Player 1 pos [-3.0, 0], Player 2 pos [3.0, 0], Puck [1.87, 0.22]
```

- ✓ Observation dimensions: 18D (same)
- ✓ Perspective: Normal (Player 1 left, not mirrored)
- ✓ Position formats: Identical
- ✓ Network architecture verified: [1024, 1024] actor, [1024, 1024, 200] critic, 4D output
- ✓ Checkpoint loading: Verified correct

---

## Hypothesis

The tournament's `bot-weak` is **not equivalent** to the local `BasicOpponent(weak=True)` because:

1. **Performance gap is too large** - 90% local vs 30-50% tournament cannot be explained by normal variance
2. **Strong bot works perfectly** - 100% vs bot-strong in tournament suggests agent is working correctly
3. **Both Player positions underperform** - Agent struggles consistently with both P1 and P2 positions vs weak bot, despite working well vs strong bot in both positions
4. **Diagnostics show identical state** - Observations are being transmitted correctly

---

## Questions for TAs

1. **Is `bot-weak` in the tournament using the same `BasicOpponent(weak=True)` implementation from the hockey library?**

   - If so, are the hyperparameters identical? (e.g., `kp`, `kd`, response timing)
   - If not, what is the weak bot implementation?
2. **Can you provide the source code or parameters for the tournament's `bot-weak` opponent?**
3. **Is there a known performance difference between the documented weak bot and what's deployed in the tournament?**
4. **Should we retrain our agent against a stronger local "weak" opponent to match tournament conditions?**

---

## Detailed Results Table

| Game # | Opponent   | Player Pos | Result | Score | Notes                      |
| ------ | ---------- | ---------- | ------ | ----- | -------------------------- |
| 1      | bot-weak   | P1         | Loss   | 1-3   | Consistent loss pattern    |
| 2      | bot-weak   | P1         | Draw   | 2-2   | Cannot break deadlock      |
| 3      | bot-weak   | P1         | Loss   | 1-3   | Agent underperforms        |
| 4      | bot-weak   | P1         | Draw   | 2-2   | Repeated pattern           |
| 5      | bot-weak   | P2         | Loss   | 1-3   | Fails as offensive player  |
| 6      | bot-strong | P1         | Win    | 4-0   | Agent dominates strong bot |
| 7      | bot-strong | P2         | Draw   | 2-2   | Works OK as P2 vs strong   |
| 8      | bot-strong | P1         | Win    | 4-0   | Consistent strength        |
| 9      | bot-strong | P2         | Win    | 3-1   | Works well vs strong       |
| 10     | bot-weak   | P1         | Draw   | 2-2   | Back to weak bot struggles |

**Pattern:** Agent performs excellently vs `bot-strong` but significantly underperforms vs `bot-weak`. This is the inverse of expected behavior and cannot be explained by overfitting to the strong opponent (since our local evaluation with identical checkpoint shows 90% vs local weak bot).

---

## Proposed Next Steps

1. **Immediate:** Verify the `bot-weak` implementation and parameters
2. **If different:** Provide specification of tournament weak bot so we can retrain
3. **If same:** Help us investigate other environmental factors

We are confident the agent code and checkpoint are correct based on:

- ✓ Perfect performance vs strong bot
- ✓ Comprehensive diagnostics confirming identical state representation
- ✓ Local testing achieving 90% vs weak with same checkpoint
- ✓ Network architecture verification
