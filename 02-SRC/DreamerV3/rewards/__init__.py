"""
DreamerV3 Rewards.

NOTE: DreamerV3 uses SPARSE REWARDS ONLY.
No reward shaping is needed - the world model handles credit assignment.

The hockey_wrapper.py provides sparse rewards:
- Win: +1.0
- Loss: -1.0
- Fault: -0.33 (optional)

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

# No reward shaping modules - sparse rewards handled in environment wrapper

__all__ = []