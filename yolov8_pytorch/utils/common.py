# Copyright 2024 Apache License 2.0. All Rights Reserved.
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
import os
from pathlib import Path

import torch
from omegaconf import DictConfig

from . import TESTS_RUNNING, RUNS_DIR, RANK

__all__ = [
    "get_results_dir", "increment_path", "load_weights",
]


def get_results_dir(config_dict: DictConfig, name: str = None) -> Path:
    r"""Automatically generate a save_dir based on the config_dict.

    Args:
        config_dict (DictConfig): The configuration dictionary.
        name (str, optional): The name of the directory. Defaults to None.

    Returns:
        (Path): The save results directory.
    """
    if config_dict.get("results_dir") is not None:
        save_dir = config_dict.save_dir
    else:
        project = config_dict.project or ("tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / config_dict.task
        name = name or config_dict.name or f"{config_dict.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=config_dict.exist_ok if RANK in (-1, 0) else True)

    return Path(save_dir)


def increment_path(path: str, exist_ok: bool = False):
    r"""Increments a file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and exist_ok is not set to True, the path will be incremented by appending a number and sep to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If mkdir is set to True, the path will be created as a
    directory if it does not already exist.

    Args:
        path (str, pathlib.Path): Path to increment.
        exist_ok (bool, optional): If True, the path will not be incremented and returned as-is. Defaults to False.

    Returns:
        (pathlib.Path): Incremented path.
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        for n in range(2, 9999):
            new_path = f"{path}{n}{suffix}"  # increment path
            if not os.path.exists(new_path):
                break
        path = Path(new_path)

    path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def load_weights(weights_path: str | Path, device: torch.device = None, fused: bool = False):
    if isinstance(weights_path, str):
        weights_path = Path(weights_path)

    checkpoint = torch.load(weights_path, map_location="cpu" if device is None else device)
    weights = (checkpoint.get("ema_model") or checkpoint["model"]).to(device).float()
    weights = weights.fuse().eval() if fused and hasattr(weights, "fused") else weights.eval()
    return weights
