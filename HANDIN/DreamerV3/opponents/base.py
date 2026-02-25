"""
This file was developed with assistance from AI: autocomplete and discussion
about the contents and behavior of the code.
"""

import abc


class BaseOpponent(abc.ABC):
    """Abstract base for opponent agents. Fixed and self-play both implement this."""

    @abc.abstractmethod
    def act(self, obs):
        """Return opponent action from obs. obs is 18d for hockey, action is 4d."""
        raise NotImplementedError("Subclasses must implement act()")

    def reset(self):
        """Override if opponent has internal state (e.g. recurrent)."""
        pass
