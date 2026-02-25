"""
This file was developed with assistance from AI: autocomplete and discussion
about the contents and behavior of the code.
"""

from .frame_capture import record_episode_frames, record_episode_frames_dreamer
from .gif_recorder import (
    create_gif_for_wandb,
    save_gif_to_wandb,
    create_gif_dreamer,
    save_gif_dreamer,
)

__all__ = [
    'record_episode_frames',
    'record_episode_frames_dreamer',
    'create_gif_for_wandb',
    'save_gif_to_wandb',
    'create_gif_dreamer',
    'save_gif_dreamer',
]
