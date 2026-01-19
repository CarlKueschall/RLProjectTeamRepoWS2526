"""
DreamerV3 Rewards.

Reward Shaping for DreamerV3:
- PBRS (Potential-Based Reward Shaping) provides dense reward signal
- Mathematically guaranteed to preserve optimal policy (Ng et al., 1999)
- Enables world model to learn meaningful reward dynamics
- Solves the sparse reward bootstrapping problem

Base rewards from hockey environment:
- Win: +1.0
- Loss: -1.0
- Draw: 0.0

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

from .pbrs import PBRSRewardShaper, compute_pbrs, compute_potential, get_potential_components

__all__ = [
    'PBRSRewardShaper',
    'compute_pbrs',
    'compute_potential',
    'get_potential_components',
]
