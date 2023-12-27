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
import math
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F_torch

__all__ = [
    "scale_image", "xywh2xyxy",
]


def scale_image(input_image: Tensor, scale_ratio: float = 1.0, retain_shape: bool = False, grid_size: int = 32):
    """
    Scales and pads an image tensor of shape img(batch_size,3,height,width) based on given scale_ratio and grid_size,
    optionally retaining the original shape.

    Args:
        input_image (Tensor): The input image tensor to be scaled.
        scale_ratio (float, optional): The ratio to scale the image. Defaults to 1.0.
        retain_shape (bool, optional): Whether to retain the original shape of the image. Defaults to False.
        grid_size (int, optional): The grid size for padding. Defaults to 32.

    Returns:
        Tensor: The scaled and padded image tensor.
    """
    # If the scale ratio is 1.0, return the original image
    if scale_ratio == 1.0:
        return input_image

    # Get the height and width of the image
    height, width = input_image.shape[2:]

    # Calculate the new size of the image
    new_size = (int(height * scale_ratio), int(width * scale_ratio))

    # Resize the image using bilinear interpolation
    input_image = F_torch.interpolate(input_image, size=new_size, mode="bilinear", align_corners=False)

    # If not retaining the original shape, pad or crop the image
    if not retain_shape:
        height, width = (math.ceil(dim * scale_ratio / grid_size) * grid_size for dim in (height, width))

    # Pad the image to the new size, using the ImageNet mean as padding value
    return F_torch.pad(input_image, [0, width - new_size[1], 0, height - new_size[0]], value=0.447)


def xywh2xyxy(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    r"""Convert bounding box coordinates from (center_x, center_y, width, height) format to (x1, y1, x2, y2) format where
    (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (center_x, center_y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    # Ensure the last dimension of the input is 4 (corresponding to center_x, center_y, width, height)
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'

    # Create an empty tensor or numpy array with the same shape and type as the input
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)

    # Compute half of the width and height
    half_width = x[..., 2] / 2
    half_height = x[..., 3] / 2

    # Compute the coordinates of the top-left corner (x1, y1)
    y[..., 0] = x[..., 0] - half_width
    y[..., 1] = x[..., 1] - half_height

    # Compute the coordinates of the bottom-right corner (x2, y2)
    y[..., 2] = x[..., 0] + half_width
    y[..., 3] = x[..., 1] + half_height

    return y
