"""
AI Usage Declaration:
This file was developed with assistance from AI autocomplete features in Cursor AI IDE.
"""

import abc
class BaseOpponent(abc.ABC):
    #########################################################
    # Base class for opponent agents.
    #########################################################
    @abc.abstractmethod
    def act(self, obs):
        #########################################################
        # Get opponent action from observation.
        #Arguments:
        #     obs: Opponent's observation (18 dims)
        # Returns:
        #     action: Opponent action (4 dims)
        raise NotImplementedError("Subclasses must implement act")

    def reset(self):
        #########################################################
        # Reset opponent state between episodes.
        #########################################################
        pass
