"""Prioritized Fictitious Self-Play (PFSP) opponent selection."""

import numpy as np


def pfsp_weight(winrate, mode="variance", p=2):
    #########################################################
    # Compute PFSP sampling weight for an opponent based on win-rate.
    #########################################################
    # Args:
    #     winrate: Win rate against this opponent (0-1)
    #     mode: "variance" (focus on ~50%) or "hard" (focus on hardest)
    #     p: Exponent for hard mode
    # Returns:
    #     Weight for sampling this opponent
    if mode == "variance":
        #########################################################
        # Variance curriculum: peaks at 50% win-rate
        #########################################################
        return winrate * (1 - winrate)
    elif mode == "hard":
        #########################################################
        # Hard curriculum: focus on hardest opponents
        #########################################################
        return (1 - winrate) ** p
    else:
        return 1.0  # Uniform fallback
