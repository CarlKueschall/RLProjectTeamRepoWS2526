# Async Training Debug Guide

## Problem Summary

Training gets stuck at ~2500 episodes with `train=0`, meaning the trainer thread never starts training. The buffer fills to ~1366 transitions but never grows beyond that, even though collectors are running at 244 ep/s.

## Root Cause Analysis

The issue is likely one of these:

1. **Transition Queue Overflow (MOST LIKELY)**:
   - macOS has strict semaphore limits
   - Queue is set to size 1000 (now increased to 5000 in the fix)
   - With 2 workers at 244 ep/s total, each episode produces ~250 transitions
   - That's 61,000 transitions/sec trying to go into a small queue
   - Collector workers drop transitions when queue is full

2. **Buffer Filler Not Draining**:
   - The buffer filler thread might be blocking on `add_transition_direct()` calls
   - Each transition requires a lock, so many transitions could cause delays

3. **Buffer Size Calculation Issue**:
   - The `buffer.size` property might not be accurately reflecting transitions added

## Debugging Steps

### Step 1: Run the Debug Test

```bash
cd /Users/carlkueschall/workspace/RLProjectHockey/02-SRC/TD3
python test_async_debug.py --num_workers 2 --run_duration 30
```

This will run for 30 seconds and print detailed debug output showing:
- Collector workers sending transitions
- Buffer filler receiving and adding transitions
- Trainer waiting for buffer warmup

### Step 2: Analyze the Output

Look for these debug messages:

```
[Worker N] Sent 100 transitions, Queue: 500
[Worker N] WARNING: Queue getting full! Size: 4500
[Worker N] WARNING: Dropped 100 transitions (queue overflow)
```

These indicate the queue is overflowing and transitions are being dropped.

```
[BufferFiller] Rate: +1000/s | Total: 5000 | Buffer: 5000 | Queue: 200 | Errors: 0
```

This is good - buffer filler is working and keeping up.

```
[AsyncTrainer] Buffer warmup: 5000/5000
[AsyncTrainer] Warmup complete, starting training
```

This means training will start.

### Step 3: Interpret the Debug Output

| Output | Meaning | Action |
|--------|---------|--------|
| `[Worker N] WARNING: Queue getting full! Size: 4500` | Queue near capacity | Increase queue size or number of buffer filler threads |
| `[BufferFiller] Rate: +0/s` | No transitions flowing | Check if queue has data (Queue field) |
| `[BufferFiller] Errors: 50+` | Exceptions adding to buffer | Check if buffer has issues |
| `[AsyncTrainer] Buffer warmup: N/5000` stalled | Buffer not filling | Transitions are being dropped or not reaching buffer |

## Changes Made

### 1. Increased Queue Size (async_orchestrator.py)
- Changed from 1000 to 5000 on macOS
- Handles burst collection better
- Platform info printed for debugging

### 2. Enhanced Buffer Filler Logging (async_orchestrator.py)
- Shows transitions added per second (not cumulative)
- Shows queue size, buffer size
- Tracks add errors
- Shows buffer internal stats

### 3. Enhanced Collector Logging (async_collector.py)
- Tracks transitions sent vs dropped
- Warns when queue >80% full
- Warns every 100 drops

### 4. Better Error Handling
- Try/except around buffer add operations
- Error tracking and reporting
- Graceful handling of full queues

## If Still Stalled After Fix

If the training still stalls, here's what to check:

### Check 1: Is the Buffer Filler Running?
Look for:
```
[BufferFiller] Starting buffer filler thread
```

If you don't see this, the orchestrator is failing to start the filler.

### Check 2: Are Transitions Reaching the Queue?
Look for:
```
[Worker 0] Sent 100 transitions, Queue: 234
```

If you don't see these, collectors aren't sending transitions (likely hanging on queue operations).

### Check 3: Is Buffer Actually Growing?
Look for:
```
[BufferFiller] Rate: +100/s | Total: 500 | Buffer: 500
```

If Buffer stays at 0 or low number while Rate is high, data is being added but not reflected in size.

### Check 4: Are There Lock Contentions?
If buffer updates are slow (Rate < 100/s with 2 workers), there might be lock issues in AsyncMemory.

## Performance Expectations

### With 2 Workers at ~122 ep/s each:
- Episode rate: 244 ep/s total
- Transitions: ~244 * 250 = 61,000/sec
- Buffer filler should process: ~1000 transitions per batch, batches as fast as possible
- Expected: Buffer fills at 1000-5000+ transitions/sec (after queue draining starts)

### With Proper Setup:
- Buffer should reach 5000 transitions in <1-2 seconds
- Trainer warmup (10,000 transitions) should complete in <5 seconds
- Training should start immediately after

## Next Steps if Issue Persists

1. **Run the debug test** and capture output
2. **Share the output** showing:
   - Where the stall happens
   - Buffer size progression
   - Queue size progression
   - Any error messages

3. **Consider these alternatives**:
   - Reduce `--warmup_episodes` to 10 (only 1000 transitions)
   - Reduce `--num_workers` to 1
   - Try sequential training (`train_hockey.py`) to confirm hardware works

## Files Modified

- `async_orchestrator.py`: Enhanced buffer filler, increased queue size
- `async_collector.py`: Enhanced transition sending with warnings
- `test_async_debug.py`: New debug script for quick testing
- `ASYNC_DEBUG_GUIDE.md`: This file

## Running the Test

```bash
# Quick 30-second test (will show debug output)
python test_async_debug.py --num_workers 2 --run_duration 30

# Save output to file for analysis
python test_async_debug.py --num_workers 2 --run_duration 30 > async_debug_output.txt 2>&1

# Analyze the output
grep -E '\[(Worker|BufferFiller|AsyncTrainer)\]' async_debug_output.txt | head -50
```

## Debugging Checklist

- [ ] Run debug test
- [ ] See [Worker N] "Sent 100 transitions" messages
- [ ] See [BufferFiller] with non-zero rate
- [ ] See buffer size increasing each second
- [ ] See [AsyncTrainer] warmup progress messages
- [ ] See warmup complete and training start
- [ ] If not, note where it stalls and which component isn't working

