"""
Prioritized Fictitious Self-Play (PFSP) opponent selection.

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import numpy as np


def pfsp_weight(winrate, mode="variance", p=2):
    """
    Compute PFSP sampling weight for an opponent based on win-rate.

    The key insight is that training against opponents where we have ~50% win rate
    provides the most learning signal (variance curriculum), while training against
    the hardest opponents (hard curriculum) can accelerate learning in some cases.

    Args:
        winrate: Win rate against this opponent (0-1)
        mode: Selection strategy:
            - "variance": Peaks at 50% win-rate (most learning signal)
            - "hard": Focus on hardest opponents (lowest win rate)
            - "uniform": Equal weight for all opponents
        p: Exponent for hard mode (higher = more focus on hard opponents)

    Returns:
        Weight for sampling this opponent (higher = more likely to be selected)
    """
    if mode == "variance":
        # Variance curriculum: peaks at 50% win-rate
        # f(x) = x * (1-x) is maximized at x=0.5
        return winrate * (1 - winrate)

    elif mode == "hard":
        # Hard curriculum: focus on hardest opponents
        # f(x) = (1-x)^p prioritizes low win-rate opponents
        return (1 - winrate) ** p

    elif mode == "uniform":
        # Uniform: equal probability for all opponents
        return 1.0

    else:
        # Fallback to uniform
        return 1.0
