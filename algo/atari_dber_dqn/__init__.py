"""
Diversity-Based Experience Replay (DBER) for Atari environments.

This implementation is based on the paper:
"Diversity-Based Experience Replay"

Key Components:
- DBERBuffer: Implements diversity-based experience replay buffer
- DBERDQN: Extends DQN with diversity-based prioritization
"""

from algo.atari_dber_dqn.diversity_buffer import (
    DBERBuffer,
    DBERReplayBufferSamples,
)
from algo.atari_dber_dqn.dber_dqn import DBERDQN

__all__ = [
    "DBERBuffer",
    "DBERReplayBufferSamples",
    "DBERDQN",
]

# Version of the atari-dber package
__version__ = "1.0.0" 