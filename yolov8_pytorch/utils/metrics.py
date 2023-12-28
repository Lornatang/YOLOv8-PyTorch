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

import torch
from torch import Tensor

__all__ = [
    "bbox_ioa", "bbox_iou",
]

def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

    Args:
        box1 (np.array): A numpy array of shape (n, 4) representing n bounding boxes.
        box2 (np.array): A numpy array of shape (m, 4) representing m bounding boxes.
        iou (bool): Calculate the standard iou if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.array): A numpy array of shape (n, m) representing the intersection over box2 area.
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def bbox_iou(
        box1: Tensor,
        box2: Tensor,
        xywh: bool = True,
        GIoU: bool = False,
        DIoU: bool = False,
        CIoU: bool = False,
        eps: float = 1e-7,
) -> Tensor:
    r"""Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.

    References:
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L71

    Examples:
        >>> box1 = torch.tensor([0.0, 0.0, 1.0, 1.0])
        >>> box2 = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]])
        >>> bbox_iou(box1, box2, xywh=False)
        tensor([1.0000, 0.2500])
    """
    # Get the coordinates of bounding boxes
    if xywh:  # if boxes are in (x, y, w, h) format, transform them to (x1, y1, x2, y2) format
        (center_x1, center_y1, width1, height1), (center_x2, center_y2, width2, height2) = box1.chunk(4, -1), box2.chunk(4, -1)
        half_width1, half_height1, half_width2, half_height2 = width1 / 2, height1 / 2, width2 / 2, height2 / 2
        box1_x1, box1_x2, box1_y1, box1_y2 = center_x1 - half_width1, center_x1 + half_width1, center_y1 - half_height1, center_y1 + half_height1
        box2_x1, box2_x2, box2_y1, box2_y2 = center_x2 - half_width2, center_x2 + half_width2, center_y2 - half_height2, center_y2 + half_height2
    else:  # if boxes are already in (x1, y1, x2, y2) format
        box1_x1, box1_y1, box1_x2, box1_y2 = box1.chunk(4, -1)
        box2_x1, box2_y1, box2_x2, box2_y2 = box2.chunk(4, -1)
        width1, height1 = box1_x2 - box1_x1, box1_y2 - box1_y1 + eps
        width2, height2 = box2_x2 - box2_x1, box2_y2 - box2_y1 + eps

    # Compute the area of intersection
    intersection = (box1_x2.minimum(box2_x2) - box1_x1.maximum(box2_x1)).clamp_(0) * (box1_y2.minimum(box2_y2) - box1_y1.maximum(box2_y1)).clamp_(0)

    # Compute the area of union
    union = width1 * height1 + width2 * height2 - intersection + eps

    # Compute Intersection over Union (IoU)
    iou = intersection / union
    if CIoU or DIoU or GIoU:
        convex_width = box1_x2.maximum(box2_x2) - box1_x1.minimum(box2_x1)  # width of the smallest enclosing box
        convex_height = box1_y2.maximum(box2_y2) - box1_y1.minimum(box2_y1)  # height of the smallest enclosing box
        if CIoU or DIoU:  # Distance IoU or Complete IoU
            convex_diagonal_squared = convex_width ** 2 + convex_height ** 2 + eps  # square of the diagonal of the smallest enclosing box
            center_distance_squared = ((box2_x1 + box2_x2 - box1_x1 - box1_x2) ** 2 + (
                    box2_y1 + box2_y2 - box1_y1 - box1_y2) ** 2) / 4  # square of the distance between box centers
            if CIoU:  # Complete IoU
                v = (4 / math.pi ** 2) * (torch.atan(width2 / height2) - torch.atan(width1 / height1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                # CIoU
                return iou - (center_distance_squared / convex_diagonal_squared + v * alpha)
                # DIoU
            return iou - center_distance_squared / convex_diagonal_squared
        convex_area = convex_width * convex_height + eps
        # Generalized IoU (GIoU)
        return iou - (convex_area - union) / convex_area
    return iou
