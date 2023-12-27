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
from typing import Optional

import torch
from torch import nn, Tensor

from .conv import BasicConv2d

__all__ = [
    "Bottleneck", "C2f", "SPPF",
]


class Bottleneck(nn.Module):
    r"""Implementation of Bottleneck block."""

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            kernel_size: Optional[int] = 3,
            groups: Optional[int] = 1,
            expansion: Optional[float] = 0.5,
            shortcut: Optional[bool] = True,
    ) -> None:
        r"""Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and expansion.

        Args:
            input_channels (int): The number of input channels.
            output_channels (int): The number of output channels.
            kernel_size (Optional[int], optional): Size of the convolving kernel. Defaults to 3.
            groups (Optional[int], optional): Number of groups for the convolutions. Defaults to 1.
            shortcut (Optional[bool], optional): Whether to use a shortcut connection. Defaults to True.
            expansion (Optional[float], optional): Expansion factor for the hidden layer. Defaults to 0.5.
        """
        super().__init__()
        hidden_channels = int(output_channels * expansion)  # hidden channels
        self.conv1 = BasicConv2d(input_channels, hidden_channels, kernel_size, 1)
        self.conv2 = BasicConv2d(hidden_channels, output_channels, kernel_size, 1, groups=groups)
        self.add = shortcut and input_channels == output_channels

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the bottleneck layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C2f(nn.Module):
    r"""Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            groups: int = 1,
            expansion: float = 0.5,
            num_layers: int = 1,
            shortcut: bool = False,
    ) -> None:
        """
        Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            groups (int, optional): Number of groups for the convolutions. Defaults to 1.
            expansion (float, optional): Expansion factor for the hidden layer. Defaults to 0.5.
            num_layers (int, optional): The number of Bottleneck layers. Defaults to 1.
            shortcut (bool, optional): Whether to use a shortcut connection. Defaults to False.
        """
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)
        self.basic_conv1 = BasicConv2d(in_channels, 2 * self.hidden_channels, 1, 1)
        self.basic_conv2 = BasicConv2d((2 + num_layers) * self.hidden_channels, out_channels, 1)
        self.module_list = nn.ModuleList(
            Bottleneck(self.hidden_channels,
                       self.hidden_channels,
                       3,
                       groups,
                       1.0,
                       shortcut) for _ in range(num_layers))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through C2f layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        y = list(self.basic_conv1(x).chunk(2, 1))
        y.extend(module(y[-1]) for module in self.module_list)
        return self.basic_conv2(torch.cat(y, 1))

    def forward_split(self, x: Tensor) -> Tensor:
        """
        Forward pass using split() instead of chunk().

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        y = list(self.basic_conv1(x).split((self.hidden_channels, self.hidden_channels), 1))
        y.extend(module(y[-1]) for module in self.module_list)
        return self.basic_conv2(torch.cat(y, 1))


class SPPF(nn.Module):
    r"""Spatial Pyramid Pooling"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        r"""Initializes the SPPF layer with given input/output channels and kernel size.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int, optional): Size of the involving kernel. Defaults to 5.

        Note:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        hidden_channels = in_channels // 2  # hidden channels
        self.basic_conv1 = BasicConv2d(in_channels, hidden_channels, 1, 1)
        self.basic_conv2 = BasicConv2d(hidden_channels * 4, out_channels, 1, 1)
        self.module_list = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the SPPF layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the SPPF layer.
        """
        x = self.basic_conv1(x)
        y1 = self.module_list(x)
        y2 = self.module_list(y1)
        return self.basic_conv2(torch.cat((x, y1, y2, self.module_list(y2)), 1))
