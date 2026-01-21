"""
Fixed opponent wrapper for hockey environment BasicOpponent.

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

from hockey.hockey_env import BasicOpponent
from .base import BaseOpponent


class FixedOpponent(BaseOpponent):
    """
    Wrapper for hockey environment BasicOpponent (weak/strong).

    This opponent uses rule-based behavior and doesn't learn.
    """

    def __init__(self, weak=True):
        """
        Initialize a fixed opponent.

        Args:
            weak: If True, use weak opponent. If False, use strong opponent.
        """
        self.opponent = BasicOpponent(weak=weak)
        self.weak = weak
        self.name = "weak" if weak else "strong"

    def act(self, obs):
        """
        Get opponent action from observation.

        Args:
            obs: Opponent's observation (18 dims)

        Returns:
            action: Opponent action (4 dims)
        """
        return self.opponent.act(obs)

    def reset(self):
        """BasicOpponent doesn't have state to reset."""
        pass
