"""
Reward shaping for DreamerV3 Hockey.

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

from .pbrs import PBRSRewardShaper, compute_pbrs, compute_potential

__all__ = ['PBRSRewardShaper', 'compute_pbrs', 'compute_potential']
