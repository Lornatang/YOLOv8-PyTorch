# Copyright 2023 AlphaBetter Corporation. All Rights Reserved.
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
import torch
from torch import nn, Tensor

__all__ = [
    "Concat",
]


class Concat(nn.Module):
    r"""Concatenate a list of tensors along dimension."""

    def __init__(self, dim=1):
        r"""Initializes the Concat module.

        Args:
            dim (int): The dimension along which the tensors will be concatenated. Default is 1.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: list) -> Tensor:
        r"""Forward pass for the Concat module.

        Args:
            x (list or Tensor): A list of tensors or a single tensor to be concatenated.

        Returns:
            Tensor: The concatenated tensor.
        """
        if not isinstance(x, list):
            x = [x]
        return torch.cat(x, self.dim)
