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

from torch import nn, Tensor

SUPPORTED_ACTIVATION = {
    "silu": nn.SiLU(inplace=True),
    "relu": nn.ReLU(inplace=True),
    "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
    "hard_swish": nn.Hardswish(inplace=True)
}


class BasicConv2d(nn.Module):
    """A basic version of basic convolution, normalization and activation.

    Args:
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            bias: bool = False,
            activation_type: str = "silu",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        if activation_type is not None:
            self.act = SUPPORTED_ACTIVATION.get(activation_type)

        self.activation_type = activation_type

    def forward(self, x: Tensor) -> Tensor:
        if self.activation_type is None:
            return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: Tensor) -> Tensor:
        if self.activation_type is None:
            return self.conv(x)
        return self.act(self.conv(x))
