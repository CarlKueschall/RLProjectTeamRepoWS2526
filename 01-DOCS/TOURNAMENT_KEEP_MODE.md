# Tournament Server: keep_mode Configuration

## Summary

**The tournament server uses `keep_mode=ON` (18-dimensional observations).**

## What is keep_mode?

`keep_mode` is a hockey environment setting that controls puck behavior:
- **`keep_mode=ON`**: Allows players to hold/carry the puck for a short duration
- **`keep_mode=OFF`**: Puck bounces immediately on contact (no holding)

## Observation Dimension Difference

The `keep_mode` setting affects observation dimensions:
- **`keep_mode=ON`**: 18-dimensional observations
- **`keep_mode=OFF`**: 16-dimensional observations

The extra 2 dimensions in `keep_mode=ON` represent puck holding state information.

## How We Determined This

On January 5, 2026, we connected the tournament client (`run_client.py`) to the tournament server and analyzed the first observation received:

```
TOURNAMENT SERVER OBSERVATION DIMENSION: 18
  → Tournament uses keep_mode=ON (18-dim observations)
```

The diagnostic output confirmed that all observations from the tournament server are 18-dimensional, indicating `keep_mode=ON`.

## Implications for Training

**When training agents for the tournament:**

✅ **DO**: Use `keep_mode=ON` (default behavior)
- This matches the tournament environment
- Ensures observation space consistency (18-dim)
- Agent learns with the same puck physics as tournament

❌ **DON'T**: Use `--no_keep_mode` for tournament training
- This would create a mismatch (16-dim vs 18-dim observations)
- Agent would be trained with different physics than tournament
- Performance would likely degrade in tournament

## Training Scripts

- **For tournament training**: Use default `keep_mode=True` or explicitly set `--keep_mode`
- **For experiments**: The `--no_keep_mode` flag is available for comparative studies, but should not be used for tournament-bound agents

## Technical Details

- **Checkpoint compatibility**: Old checkpoints trained with `keep_mode=OFF` (22D critic input: 18 obs + 4 actions) can be loaded, but the tournament client will pad 16-dim observations to 18-dim if needed
- **Current training**: New checkpoints use 26D critic input (18 obs + 8 actions) for full game state awareness
- **Tournament client**: Automatically handles observation dimension mismatches by padding when necessary

## Date Determined

January 5, 2026

