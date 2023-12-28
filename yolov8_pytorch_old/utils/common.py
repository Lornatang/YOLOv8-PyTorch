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
from copy import deepcopy
from typing import List, Tuple

import thop
import torch
from torch import nn, Tensor

__all__ = [
    "convert_bbox_to_letterbox", "convert_letterbox_to_bbox", "calculate_model_parameters", "calculate_model_flops", "make_anchors",
    "select_candidates_in_gts", "select_highest_overlaps", "select_device",
    "TaskAlignedAssigner",
]

logger = logging.getLogger(__name__)

from .metrics import bbox_iou


def convert_bbox_to_letterbox(anchor_points: Tensor, bbox: Tensor, regularization_max: float) -> Tensor:
    r"""Transform bounding box coordinates from (x1, y1, x2, y2) format to distance format (left, top, right, bottom).

    Args:
        anchor_points (Tensor): The anchor points for the bounding boxes.
        bbox (Tensor): The bounding boxes in (x1, y1, x2, y2) format.
        regularization_max (float): The maximum regularization value.

    Returns:
        Tensor: The bounding boxes in distance format (left, top, right, bottom).
    """
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, regularization_max - 0.01)


def convert_letterbox_to_bbox(letterbox: Tensor, anchor_points: Tensor, xywh: bool = True, dim: int = -1) -> Tensor:
    r"""Convert the distance (upper left and lower right) to a bounding box (center x, center y, width, height)
    or bounding box (upper left x, upper left y, lower right x, lower right y)

    Args:
        letterbox (Tensor): The letterbox tensor.
        anchor_points (Tensor): The anchor points.
        xywh (bool, optional): Whether to return a bounding box (center x, center y, width, height).
            Defaults to True.
        dim (int, optional): The dimension along which to concatenate the tensors. Defaults to -1.

    Returns:
        Tensor: The bounding box.

    Examples:
        >>> letterbox = torch.tensor([0.0, 0.0, 1.0, 1.0])
        >>> anchor_points = torch.tensor([0.0, 0.0, 1.0, 1.0])
        >>> convert_letterbox_to_bbox(letterbox, anchor_points)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])
    """
    lt, rb = letterbox.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb

    # xywh bbox
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)

    # xyxy bbox
    return torch.cat((x1y1, x2y2), dim)


def calculate_model_parameters(model: nn.Module) -> int:
    """Calculate and return the total number of parameters in a YOLO model.

    Args:
        model (nn.Module): The YOLO model.

    Returns:
        int: The total number of parameters in the YOLO model.
    """
    return sum(parameter.numel() for parameter in model.parameters())


def calculate_model_flops(model: nn.Module, image_size: int = 640) -> float:
    r"""Calculate and return the FLOPs (Floating Point Operations Per Second) of a YOLO model.

    Args:
        model (nn.Module): The YOLO model.
        image_size (int, optional): The size of the input image. Defaults to 640.

    Returns:
        float: The FLOPs of the YOLO model.

    Note:
        FLOPs gives a rough estimate of the computational complexity of the model.
    """
    # Get the first parameter tensor of the model
    first_param = next(model.parameters())

    # Determine the stride of the model
    stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32

    # Create an empty tensor with the same device as the model's parameters
    input_tensor = torch.empty((1, first_param.shape[1], stride, stride), device=first_param.device)

    # Calculate the FLOPs of the model using thop (Torch Hub of PyTorch)
    flops = thop.profile(deepcopy(model), inputs=[input_tensor], verbose=False)[0] / 1E9 * 2 if thop else 0

    # Ensure the image size is a list for later calculations
    image_size = image_size if isinstance(image_size, list) else [image_size, image_size]

    # Calculate and return the final FLOPs value
    return flops * image_size[0] / stride * image_size[1] / stride


def make_anchors(features: Tensor, strides: List[int], grid_cell_offset: float = 0.5) -> Tuple[Tensor, Tensor]:
    r"""Generate anchor points from the feature map.

    Args:
        features (Tensor): The feature maps from which to generate anchors.
        strides (List[int]): The strides of the feature maps.
        grid_cell_offset (float, optional): The offset to apply when generating the grid cells. Defaults to 0.5.

    Returns:
        Tuple[Tensor, Tensor]: The generated anchor points and stride tensors.

    Examples:
        >>> features = torch.rand(1, 3, 416, 416)
        >>> strides = [8, 16, 32]
        >>> make_anchors(features, strides)
        (tensor([[0.5000, 0.5000],
                 [1.5000, 0.5000],
                 [2.5000, 0.5000],
                 ...,
                 [4.1500, 4.1500],
                 [5.1500, 4.1500],
                 [6.1500, 4.1500]]), tensor([[8.],
                 [8.],
                 [8.],
                 ...,
                 [8.],
                 [8.],
                 [8.]]))

    Raises:
        AssertionError: If the features are None.
    """
    anchor_points, stride_tensor = [], []
    assert features is not None
    dtype, device = features[0].dtype, features[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = features[i].shape
        shift_x = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        shift_y = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        anchor_points.append(torch.stack((shift_x, shift_y), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def select_candidates_in_gts(xy_centers: Tensor, gt_bboxes: Tensor, eps: float = 1e-9) -> Tensor:
    r"""Select the positive anchor center in ground truth bounding boxes.

    Args:
        xy_centers (Tensor): The x and y coordinates of the anchor centers. Shape: (h*w, 2).
        gt_bboxes (Tensor): The ground truth bounding boxes. Shape: (b, n_boxes, 4).
        eps (float, optional): A small value to prevent division by zero. Defaults to 1e-9.

    Returns:
        Tensor: A tensor indicating whether each anchor center is inside a ground truth bounding box. Shape: (b, n_boxes, h*w).

    Examples:
        >>> xy_centers = torch.tensor([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        >>> gt_bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]])
        >>> select_candidates_in_gts(xy_centers, gt_bboxes)
        tensor([[[ True, False, False],
                 [False,  True, False]]])
    """
    num_anchors = xy_centers.shape[0]
    batch_size, num_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(batch_size, num_boxes, num_anchors, -1)
    return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos: Tensor, overlaps: Tensor, num_max_boxes: int) -> Tuple[Tensor, Tensor, Tensor]:
    r"""If an anchor box is assigned to multiple ground truth boxes, select the one with the highest Intersection over Union (IoU).

    Args:
        mask_pos (Tensor): A mask indicating the positive anchor boxes. Shape: (b, n_max_boxes, h*w).
        overlaps (Tensor): The IoU overlaps between anchor boxes and ground truth boxes. Shape: (b, n_max_boxes, h*w).
        num_max_boxes (int): The maximum number of boxes.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: 
            - target_gt_idx (Tensor): The indices of the selected ground truth boxes for each anchor box. Shape: (b, h*w).
            - fg_mask (Tensor): A mask indicating the final selected anchor boxes. Shape: (b, h*w).
            - mask_pos (Tensor): An updated mask indicating the positive anchor boxes. Shape: (b, n_max_boxes, h*w).
    Examples:
        >>> mask_pos = torch.tensor([[[1, 0, 0], [0, 1, 0]]])
        >>> overlaps = torch.tensor([[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]])
        >>> num_max_boxes = 2
        >>> select_highest_overlaps(mask_pos, overlaps, num_max_boxes)
        (tensor([[0, 1]]), tensor([[1, 1]]), tensor([[[1., 0., 0.],
                 [0., 1., 0.]]]))
    """
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)  # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

        is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
        fg_mask = mask_pos.sum(-2)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


def select_device(device: str = "cpu") -> torch.device:
    r"""Automatically select device (CPU or GPU).

    Args:
        device (str, optional): device name. Defaults to "cpu".

    Raises:
        ValueError: device not supported.

    Examples:
        >>> select_device("cpu")
        Use CPU.
        >>> select_device("cuda")
        Use CUDA.
        >>> select_device("gpu")
        Use CUDA.
        >>> select_device("tpu")
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in select_device
        ValueError: Device 'tpu' not supported. Choices: ['cpu', 'cuda', 'gpu']

    Returns:
        torch.device: device.
    """
    supported_devices = ["cpu", "cuda", "gpu"]
    if device not in supported_devices:
        raise ValueError(f"Device '{device}' not supported. Choices: {supported_devices}")

    if device == "cpu":
        logger.info("Use CPU.")
        device = torch.device("cpu")
        if torch.cuda.is_available():
            logger.info("You have a CUDA device, enabling CUDA will give you a large boost in performance.")
    elif device in ["cuda", "gpu"]:
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, switching to CPU.")
            device = torch.device("cpu")
        else:
            logger.info("Use CUDA.")
            device = torch.device("cuda")

    return device


def model_info_for_loggers(trainer):
    """
    Return model info dict with useful model information.

    Example:
        YOLOv8n info for loggers
        ```python
        results = {'model/parameters': 3151904,
                   'model/GFLOPs': 8.746,
                   'model/speed_ONNX(ms)': 41.244,
                   'model/speed_TensorRT(ms)': 3.211,
                   'model/speed_PyTorch(ms)': 18.755}
        ```
    """
    if trainer.args.profile:  # profile ONNX and TensorRT times
        from yolov8_pytorch.utils.benchmarks import ProfileModels
        results = ProfileModels([trainer.last], device=trainer.device).profile()[0]
        results.pop('model/name')
    else:  # only return PyTorch times from most recent validation
        results = {
            'model/parameters': calculate_model_flops(trainer.model),
            'model/GFLOPs': round(calculate_model_flops(trainer.model), 3)}
    results['model/speed_PyTorch(ms)'] = round(trainer.validator.speed['inference'], 3)
    return results

class TaskAlignedAssigner(nn.Module):
    r"""A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int, optional): The number of top candidates to consider. Defaults to 13.
        num_classes (int, optional): The number of classes. Defaults to 80.
        alpha (float, optional): The alpha value for the alignment metric. Defaults to 1.0.
        beta (float, optional): The beta value for the alignment metric. Defaults to 6.0.
        eps (float, optional): A small value to prevent division by zero. Defaults to 1e-9.

    References:
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores: Tensor, pd_bboxes: Tensor, anc_points: Tensor, gt_labels: Tensor, gt_bboxes: Tensor, mask_gt: Tensor):
        """
        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(
            self,
            pd_scores: Tensor,
            pd_bboxes: Tensor,
            gt_labels: Tensor,
            gt_bboxes: Tensor,
            anc_points: Tensor,
            mask_gt: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Get in_gts mask, (b, max_num_obj, h*w).
        
        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            mask_pos (Tensor): shape(bs, max_num_obj, h*w)
            align_metric (Tensor): shape(bs, max_num_obj, h*w)
            overlaps (Tensor): shape(bs, max_num_obj, h*w)

        Examples:
            >>> pd_scores = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
            >>> pd_bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]])
            >>> gt_labels = torch.tensor([[[0], [1]]])
            >>> gt_bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]])
            >>> anc_points = torch.tensor([[0.5, 0.5], [1.5, 1.5]])
            >>> mask_gt = torch.tensor([[[1], [1]]])
            >>> get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
            (tensor([[[ True, False],
                     [False,  True]]]), tensor([[[0.1000, 0.0000],
                     [0.0000, 0.6000]]]), tensor([[[0.2500, 0.0000],
                     [0.0000, 0.2500]]]))
        """
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(
            self,
            pd_scores: Tensor,
            pd_bboxes: Tensor,
            gt_labels: Tensor,
            gt_bboxes: Tensor,
            mask_gt: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            align_metric (Tensor): shape(bs, max_num_obj, h*w)
            overlaps (Tensor): shape(bs, max_num_obj, h*w)

        Examples:
            >>> pd_scores = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
            >>> pd_bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]])
            >>> gt_labels = torch.tensor([[[0], [1]]])
            >>> gt_bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]])
            >>> mask_gt = torch.tensor([[[1], [1]]])
            >>> get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)
            (tensor([[[0.1000, 0.0000],
                     [0.0000, 0.6000]]]), tensor([[[0.2500, 0.0000],
                     [0.0000, 0.2500]]]))
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics: Tensor, largest: bool = True, topk_mask: Tensor = None) -> Tensor:
        r"""Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """

        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels: Tensor, gt_bboxes: Tensor, target_gt_idx: Tensor, fg_mask: Tensor):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes),
                                    dtype=torch.int64,
                                    device=target_labels.device)  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores
