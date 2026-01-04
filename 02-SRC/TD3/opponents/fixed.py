from hockey.hockey_env import BasicOpponent
from .base import BaseOpponent
class FixedOpponent(BaseOpponent):
    #########################################################
    # Wrapper for hockey environment BasicOpponent (weak/strong).
    #########################################################
    def __init__(self, weak=True):
        #########################################################
        #Initialize a fixed opponent (weak/strong).
        #Arguments:
        #     weak: If True, use weak opponent. If False, use strong opponent.
 
        self.opponent = BasicOpponent(weak=weak)
        self.weak = weak

    def act(self, obs):
        #########################################################
        # Get opponent action from observation.
        #Arguments:
        #     obs: Opponent's observation (18 dims)
        #Returns:
        #     action: Opponent action (4 dims)
        return self.opponent.act(obs)

    def reset(self):
        #########################################################
        # BasicOpponent doesn't have state to reset
        #########################################################
        pass
