"""
This file was developed with assistance from AI: autocomplete and discussion
about the contents and behavior of the code.
"""

from hockey.hockey_env import BasicOpponent
from .base import BaseOpponent


class FixedOpponent(BaseOpponent):
    """Wraps hockey BasicOpponent (weak/strong). Rule-based, doesn't learn."""

    def __init__(self, weak=True):
        self.opponent = BasicOpponent(weak=weak)
        self.weak = weak
        self.name = "weak" if weak else "strong"

    def act(self, obs):
        return self.opponent.act(obs)

    def reset(self):
        pass
