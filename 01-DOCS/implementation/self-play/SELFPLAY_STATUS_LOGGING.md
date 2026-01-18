# Self-Play Status Logging

## What You'll See in Terminal Output

### Progress Bar (Every Episode)

**Before self-play activation:**
```
Training:   1%|          | 100/90500 [00:42<60:00, 25.00ep/s, reward=10.5, avg=8.2, win_rate=75.34%, wins=75/100, SP=WAITING (ep 0)]
```

**After self-play activation:**
```
Training:   1%|          | 300/90500 [02:15<58:00, 25.50ep/s, reward=12.1, avg=9.5, win_rate=82.33%, wins=247/300, SP=ACTIVE (pool:3)]
```

The `SP` field shows:
- `WAITING (ep 0)` - Self-play configured to start at episode 0, waiting for activation gates
- `ACTIVE (pool:5)` - Self-play is active, 5 opponents in pool

---

### Self-Play Activation Check (During Evaluations)

Every time evaluation runs (every 250 episodes), you'll see:

**Before activation:**
```
============================================================================
🎮 SELF-PLAY ACTIVATED AT EPISODE 251! 🎮
============================================================================
Pool seeded with 1 opponent(s)
Dynamic anchor mixing: True
PFSP enabled: True
Regression rollback: True
============================================================================
```

**If not yet ready:**
```
[SELF-PLAY CHECK]
  Win rate vs weak: 85.50% (gate: 20.00%)
  Rolling variance: 0.0234 (gate: 0.5000)
  Status: ✓ ACTIVATION CONDITIONS MET!
```

Or if gates not passed:
```
[SELF-PLAY CHECK]
  Win rate vs weak: 15.50% (gate: 20.00%)
  Rolling variance: 0.6500 (gate: 0.5000)
  Status: ✗ Waiting for gates...
```

---

## Key Metrics in Terminal

| Field | Meaning |
|-------|---------|
| `SP=WAITING (ep 0)` | Self-play configured, waiting for activation |
| `SP=ACTIVE (pool:5)` | Self-play running, 5 agents in opponent pool |
| `Win rate vs weak: 85.50%` | Current win rate against weak bot |
| `Rolling variance: 0.0234` | Stability measure (lower = more stable) |
| `✓ ACTIVATION CONDITIONS MET!` | Ready to activate, will switch next episode |
| `✗ Waiting for gates...` | Not ready yet, gates not met |

---

## Expected Timeline

```
Episode 1-249:
  Progress bar shows: SP=WAITING (ep 0)
  No [SELF-PLAY CHECK] output (no eval yet)

Episode 250 (Evaluation runs):
  Progress bar: SP=WAITING (ep 0)
  Prints evaluations vs weak, strong
  Prints [SELF-PLAY CHECK] with gate status

Episode 251+:
  🎮 SELF-PLAY ACTIVATED banner prints
  Progress bar: SP=ACTIVE (pool:1)
  Agent now plays mixed weak/strong/self-play opponents
  Pool grows to 2, 3, 4... as checkpoints are added
```

---

## No More Guessing!

Now you can **immediately see** in the terminal:
1. Is self-play configured? (`SP=WAITING` or `SP=ACTIVE`)
2. Which gates are blocking activation? (win rate? variance?)
3. **Exactly when** it activates! (🎮 banner)
4. How many opponents in the pool? (`SP=ACTIVE (pool:5)`)

You don't need to wait for WandB anymore! ✅

