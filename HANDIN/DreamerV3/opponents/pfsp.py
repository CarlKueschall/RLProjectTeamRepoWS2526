"""
This file was developed with assistance from AI: autocomplete and discussion
about the contents and behavior of the code.
"""

import numpy as np


def pfsp_weight(winrate, mode="variance", p=2):
    """
    PFSP sampling weight. ~50% win rate gives most learning signal (variance).
    Hard mode focuses on toughest opponents.
    """
    if mode == "variance":
        return winrate * (1 - winrate)

    elif mode == "hard":
        return (1 - winrate) ** p

    elif mode == "uniform":
        return 1.0

    else:
        return 1.0
