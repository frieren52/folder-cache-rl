from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from src.replay import StateArray  # noqa: E402


def state(candidate_count: int = 2, selected: tuple[int, ...] = (), legal: tuple[int, ...] | None = None) -> StateArray:
    action = np.zeros(candidate_count, dtype=np.bool_)
    action[list(range(candidate_count)) if legal is None else list(legal)] = True
    selected_mask = np.zeros(candidate_count, dtype=np.bool_)
    if selected:
        selected_mask[list(selected)] = True
        action[list(selected)] = False
    return StateArray(
        np.zeros((256, 259), dtype=np.float32),
        np.ones(256, dtype=np.bool_),
        np.zeros(2, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        np.arange(candidate_count, dtype=np.int64),
        np.zeros((candidate_count, 274), dtype=np.float32),
        np.ones(candidate_count, dtype=np.bool_),
        action,
        selected_mask,
    )

