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
import logging
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["generate_feature_maps", "save_feature_maps"]


def generate_feature_maps(x: Tensor, module_type: str, num_features: int = 32) -> List[plt.Axes]:
    r"""Generate feature maps of a given model module during inference.

    Args:
        x (Tensor): Features to be visualized.
        module_type (str): Module type.
        num_features (int, optional): Maximum number of feature maps to plot. Defaults to 32.

    Returns:
        List[plt.Axes]: List of matplotlib Axes objects with the feature maps.
    """
    for m in ["Detect"]:
        if m in module_type:
            return []
    batch, channels, height, width = x.shape
    if height > 1 and width > 1:
        blocks = torch.chunk(x[0].cpu(), channels, dim=0)
        num_features = min(num_features, channels)
        fig, ax = plt.subplots(math.ceil(num_features / 8), 8, tight_layout=True)
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(num_features):
            ax[i].imshow(blocks[i].squeeze())
            ax[i].axis("off")
        return ax
    return []


def save_feature_maps(ax: List[plt.Axes], stage: int, module_type: str, save_dir=Path("results/detect/exp")) -> None:
    r"""Save the generated feature maps to a file.

    Args:
        ax (List[plt.Axes]): List of matplotlib Axes objects with the feature maps.
        stage (int): Module stage within the model.
        module_type (str): Module type.
        save_dir (Path, optional): Directory to save results. Defaults to Path("runs/detect/exp").
    """
    file_path = save_dir / f"stage{stage}_{module_type.split('')[-1]}_features.png"
    logger.info(f"Saving {file_path}... ({len(ax)}/{len(ax) * 8})")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    np.save(str(file_path.with_suffix(".npy")), [a.images[0].get_array().data for a in ax])
