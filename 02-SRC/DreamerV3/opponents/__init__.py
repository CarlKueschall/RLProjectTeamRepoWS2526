"""
Opponent management for DreamerV3 Hockey training.

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

from .base import BaseOpponent
from .fixed import FixedOpponent
from .pfsp import pfsp_weight
from .self_play import SelfPlayManager

__all__ = ['BaseOpponent', 'FixedOpponent', 'pfsp_weight', 'SelfPlayManager']
