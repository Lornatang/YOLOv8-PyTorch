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
import ast
import contextlib
from typing import Optional

import torch
from omegaconf import DictConfig
from torch import nn

from .module import BasicConv2d, C2f, Concat, Detect, SPPF

__all__ = [
    "create_model_from_yaml", "make_divisible",
]


def create_model_from_yaml(model_config: DictConfig, verbose: bool = False) -> [nn.Module, list]:
    """Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        model_config (DictConfig): The model.yaml dictionary.
        verbose (bool, optional): Whether to print the model architecture. Defaults to False.

    Returns:
        [nn.Module, list]: The PyTorch model and the save list.

    Examples:
        >>> model, save_list = create_model_from_yaml(model_config)
        >>> print(model)
        >>> print(save_list)
        >>> print(sum(x.numel() for x in model.parameters()))
    """
    in_channels = model_config.IN_CHANNELS
    num_classes = model_config.NUM_CLASSES
    depth_multiple = model_config.DEPTH_MULTIPLE
    width_multiple = model_config.WIDTH_MULTIPLE
    max_channels = model_config.MAX_CHANNELS

    channels = [in_channels]
    layers, save_list, out_channels = [], [], channels[-1]
    for i, (_from, _num_layers, _module, _parameters) in enumerate(model_config.ARCH):
        _module = getattr(torch.nn, _module[3:]) if "nn." in _module else globals()[_module]  # get module
        for j, parameters in enumerate(_parameters):
            if isinstance(parameters, str):
                with contextlib.suppress(ValueError):
                    _parameters[j] = locals()[parameters] if parameters in locals() else ast.literal_eval(parameters)

        _num_layers = n_ = max(round(_num_layers * depth_multiple), 1) if _num_layers > 1 else _num_layers  # depth gain
        if _module in (BasicConv2d, C2f, nn.ConvTranspose2d, SPPF):
            in_channels, out_channels = channels[_from], _parameters[0]
            if out_channels != num_classes:
                out_channels = make_divisible(min(out_channels, max_channels) * width_multiple, 8)

            _parameters = [in_channels, out_channels, *_parameters[1:]]
            if _module is C2f:
                _parameters.insert(4, _num_layers)
                _num_layers = 1
        elif _module is nn.BatchNorm2d:
            _parameters = [channels[_from]]
        elif _module is Concat:
            out_channels = sum(channels[x] for x in _from)
        elif _module is Detect:
            _parameters.append([channels[x] for x in _from])
        else:
            out_channels = channels[_from]

        _modules = nn.Sequential(*(_module(*_parameters) for _ in range(_num_layers))) if _num_layers > 1 else _module(*_parameters)
        module_type = str(_module)[8:-2].replace("__main__.", "")
        _modules.i, _modules.f, _modules.type = i, _from, module_type
        save_list.extend(x % i for x in ([_from] if isinstance(_from, int) else _from) if x != -1)
        layers.append(_modules)
        if i == 0:
            channels = []
        channels.append(out_channels)
    return nn.Sequential(*layers), sorted(save_list)


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    r"""This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
