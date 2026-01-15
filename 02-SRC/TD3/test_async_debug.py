#!/usr/bin/env python3
"""
Quick debug test for async training.

Run this to capture detailed debug output showing:
1. Collector workers sending transitions
2. Buffer filler receiving transitions
3. Trainer waiting for warmup

Usage:
    python test_async_debug.py --num_workers 2 --run_duration 30

This will run for ~30 seconds and print detailed debug info to stdout.
"""

import argparse
import subprocess
import sys
import time
import signal
from pathlib import Path

def run_async_training(num_workers=2, run_duration=30, max_episodes=100000):
    """Run async training and capture output."""

    cmd = [
        sys.executable,
        "train_hockey_async.py",
        "--mode", "NORMAL",
        "--opponent", "weak",
        "--num_workers", str(num_workers),
        "--max_episodes", str(max_episodes),
        "--warmup_episodes", "50",  # Only 50*100 = 5000 transitions for warmup
        "--no_wandb",  # Disable W&B to reduce noise
    ]

    print("=" * 80)
    print("ASYNC TRAINING DEBUG TEST")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print(f"Run duration: {run_duration} seconds")
    print(f"Workers: {num_workers}")
    print("=" * 80)
    print()

    # Start the training process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    start_time = time.time()
    try:
        # Print output until timeout
        while time.time() - start_time < run_duration:
            line = process.stdout.readline()
            if line:
                print(line.rstrip())
            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Kill the process
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    print()
    print("=" * 80)
    print("DEBUG OUTPUT CAPTURE COMPLETE")
    print("=" * 80)
    print()
    print("WHAT TO LOOK FOR:")
    print("1. [Worker N] Sent 100 transitions - Shows collectors are sending data")
    print("2. [BufferFiller] Last 1s: +X transitions - Shows buffer filler is receiving data")
    print("3. [AsyncTrainer] Buffer warmup: N/M - Shows trainer waiting for buffer warmup")
    print("4. [AsyncTrainer] Warmup complete - If you see this, training will start")
    print()
    print("DEBUGGING GUIDE:")
    print("- If no [Worker N] messages: Collectors aren't running")
    print("- If [Worker N] but no [BufferFiller] additions: Queue isn't flowing to filler")
    print("- If [BufferFiller] +0 transitions: Queue has data but buffer isn't adding it")
    print("- If Buffer.size not increasing: Data is being dropped or locked")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug async training")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--run_duration", type=int, default=30, help="Duration in seconds")

    args = parser.parse_args()

    # Change to TD3 directory if needed
    current_dir = Path.cwd()
    if not (current_dir / "train_hockey_async.py").exists():
        td3_dir = Path(__file__).parent
        import os
        os.chdir(td3_dir)

    run_async_training(args.num_workers, args.run_duration)
