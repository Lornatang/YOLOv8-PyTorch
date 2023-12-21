# Copyright 2023 Lornatang Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Initialize all seeds to facilitate reproduction
"""
import logging
import os
import random

import numpy as np
import torch

__all__ = [
    "init_seed",
]

logger = logging.getLogger(__name__)


def init_seed(seed: int = None) -> None:
    r"""Fixed random seed for reproducibility

    Args:
        seed (int, optional): random seed.

    Returns:
        None

    Raise:
        TypeError: if seed is not int.
        ValueError: if seed is not in bounds.

    Examples:
        >>> init_seed(42)
        >>> init_seed()

    Note:
        This function sets the random seed of the following libraries:
        - :mod:`random`
        - :mod:`numpy`
        - :mod:`torch`
        - :mod:`torch.cuda`

    References:
        https://github.com/Lightning-AI/pytorch-lightning/blob/1.0.5/pytorch_lightning/utilities/seed.py
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED", _select_seed_randomly(min_seed_value, max_seed_value))
        seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if (seed > max_seed_value) or (seed < min_seed_value):
        logger.warning(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    logger.info(f"Set random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    r"""Select a random seed in the correct range.

    Args:
        min_seed_value (int, optional): minimum seed value. Defaults to 0.
        max_seed_value (int, optional): maximum seed value. Defaults to 255.

    Returns:
        int: random seed.

    Examples:
        >>> _select_seed_randomly()
        42
    """
    seed = random.randint(min_seed_value, max_seed_value)
    logger.warning(f"No correct seed found, seed set to {seed}")
    return seed
