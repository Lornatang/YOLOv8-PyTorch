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
import time
from pathlib import Path

import torch
from torch import nn

from .common import calculate_model_parameters, calculate_model_flops

__all__ = [
    "fuse_conv_and_bn", "inference_mode", "model_summary", "time_sync",
]

logger = logging.getLogger(__name__)


def fuse_conv_and_bn(convolution: nn.Conv2d, batch_norm: nn.BatchNorm2d) -> nn.Conv2d:
    """
    This function fuses Conv2d() and BatchNorm2d() layers into a single Conv2d layer.
    This can reduce the computational cost and memory usage during inference.
    For more details, refer to: https://tehnokv.com/posts/fusing-batchnorm-and-conv/

    Args:
        convolution (nn.Conv2d): The convolution layer to be fused.
        batch_norm (nn.BatchNorm2d): The batch normalization layer to be fused.

    Returns:
        nn.Conv2d: The fused convolution layer.

    Examples:
        >>> convolution = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        >>> batch_norm = nn.BatchNorm2d(32)
        >>> fused_convolution = fuse_conv_and_bn(convolution, batch_norm)
        >>> fused_convolution
        Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    """
    # Create a new convolution layer with the same configuration as the input convolution layer
    fused_convolution = nn.Conv2d(convolution.in_channels,
                                  convolution.out_channels,
                                  kernel_size=convolution.kernel_size,
                                  stride=convolution.stride,
                                  padding=convolution.padding,
                                  dilation=convolution.dilation,
                                  groups=convolution.groups,
                                  bias=True).requires_grad_(False).to(convolution.weight.device)

    # Prepare the filters for the new convolution layer
    weight_convolution = convolution.weight.clone().view(convolution.out_channels, -1)
    weight_batch_norm = torch.diag(batch_norm.weight.div(torch.sqrt(batch_norm.eps + batch_norm.running_var)))
    fused_convolution.weight.copy_(torch.mm(weight_batch_norm, weight_convolution).view(fused_convolution.weight.shape))

    # Prepare the spatial bias for the new convolution layer
    bias_convolution = torch.zeros(convolution.weight.size(0), device=convolution.weight.device) if convolution.bias is None else convolution.bias
    bias_batch_norm = batch_norm.bias - batch_norm.weight.mul(batch_norm.running_mean).div(torch.sqrt(batch_norm.running_var + batch_norm.eps))
    fused_convolution.bias.copy_(torch.mm(weight_batch_norm, bias_convolution.reshape(-1, 1)).reshape(-1) + bias_batch_norm)

    return fused_convolution


def inference_mode():
    def decorate(func):
        """Applies the appropriate torch decorator for inference mode based on the torch version."""
        if torch.is_inference_mode_enabled():
            return func  # already in inference_mode, act as a pass-through
        else:
            return torch.no_grad()(func)  # apply torch.no_grad() decorator

    return decorate


def model_summary(model: nn.Module, image_size: int = 640, verbose: bool = True) -> tuple | None:
    """
    Print the summary of the model including the number of parameters and FLOPs.
    If the model is fused, it will be indicated in the summary.

    Args:
        model (nn.Module): The model to summarize.
        image_size (int, optional): The size of the input image. Defaults to 640.
        verbose (bool, optional): Whether to print the summary. Defaults to True.

    Returns:
        tuple | None: The number of parameters and FLOPs if verbose is True, otherwise None.
    """
    if not verbose:
        return
    model_parameters = calculate_model_parameters(model)  # number of parameters
    model_flops = calculate_model_flops(model, image_size)  # FLOPs
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""  # check if model is fused
    logger.info(f"summary{fused}: {model_parameters} parameters, {model_flops:.1f} GFLOPs\n")  # print summary
    return model_parameters, model_flops


def time_sync():
    r"""Returns the current time, synchronized with the CUDA device if available.

    This function first checks if CUDA is available. If it is, it synchronizes
    the CUDA device with the CPU. This is done to ensure that any CUDA operations
    are completed before getting the current time, making the returned time more
    accurate in the context of the CUDA operations.

    After the optional synchronization, it returns the current time as given by
    time.time().

    Returns:
        float: The current time in seconds since the epoch as a floating point number.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def get_save_dir(args, name=None):
    """Return save_dir as created from train/val/predict arguments."""

    if getattr(args, 'save_dir', None):
        save_dir = args.save_dir
    else:
        from yolov8_pytorch.utils.files import increment_path

        project = args.project / args.task
        name = name or args.name or f'{args.mode}'
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok)

    return Path(save_dir)
