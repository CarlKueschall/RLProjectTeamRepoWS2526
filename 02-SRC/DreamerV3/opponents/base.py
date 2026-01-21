"""
Base class for opponent agents.

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import abc


class BaseOpponent(abc.ABC):
    """
    Abstract base class for opponent agents.

    All opponent types (fixed, self-play) must implement this interface.
    """

    @abc.abstractmethod
    def act(self, obs):
        """
        Get opponent action from observation.

        Args:
            obs: Opponent's observation (18 dims for hockey)

        Returns:
            action: Opponent action (4 dims)
        """
        raise NotImplementedError("Subclasses must implement act()")

    def reset(self):
        """
        Reset opponent state between episodes.

        Override this method if opponent has internal state (e.g., recurrent).
        """
        pass
