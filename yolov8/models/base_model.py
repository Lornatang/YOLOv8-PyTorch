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
from typing import Union, Callable, List

import thop
from torch import nn, Tensor

from .module import BasicConv2d, Detect
from yolov8.utils.misc import fuse_conv_and_bn, model_summary, time_sync
from yolov8.utils.visualizer import generate_feature_maps, save_feature_maps

logger = logging.getLogger(__name__)

__all__ = [
    "BaseModel",
]


class BaseModel(nn.Module):
    r"""The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x: Union[Tensor, dict], *args, **kwargs) -> Tensor:
        r"""Forward pass of the model on a single scale. Wrapper for `_forward_once` method.

        Args:
            x (Tensor | dict): The input image tensor or a dict including image tensor and ground truth labels.

        Returns:
            Tensor: The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, input_tensor: Tensor, profile: bool = False, visualize: bool = False, augment: bool = False) -> Tensor:
        r"""Perform a forward pass through the network.

        Args:
            input_tensor (Tensor): The input tensor to the model.
            profile (bool, optional):  Print the computation time of each layer if True. Defaults to False.
            visualize (bool, optional): Save the feature maps of the model if True. Defaults to False.
            augment (bool, optional): Augment image during prediction. Defaults to False.

        Returns:
            Tensor: The last output of the model.
        """
        if augment:
            return self._predict_augment(input_tensor)
        return self._predict_once(input_tensor, profile, visualize)

    def _predict_once(self, input_tensor: Tensor, profile: bool = False, visualize: bool = False) -> Tensor:
        r"""Perform a forward pass through the network.

        Args:
            input_tensor (Tensor): The input tensor to the model.
            profile (bool, optional):  Print the computation time of each layer if True. Defaults to False.
            visualize (bool, optional): Save the feature maps of the model if True. Defaults to False.

        Returns:
            Tensor: The last output of the model.
        """
        output_tensors, layer_times = [], []  # outputs and layer times
        for layer in self.model:
            if layer.f != -1:  # if not from previous layer
                input_tensor = output_tensors[layer.f] if isinstance(layer.f, int) else [input_tensor if index == -1 else output_tensors[index] for
                                                                                         index in layer.f]
            if profile:
                self._profile_one_layer(layer, input_tensor, layer_times)
            input_tensor = layer(input_tensor)  # run
            output_tensors.append(input_tensor if layer.i in self.save_list else None)  # save output
            if visualize:
                feature_maps = generate_feature_maps(input_tensor, layer.type, layer.i)
                save_feature_maps(feature_maps, layer.type, layer.i, visualize)
        return input_tensor

    def _predict_augment(self, input_tensor: Tensor) -> Tensor:
        r"""Perform augmentations on the input image and return the inference result.

        If the model does not support augmented inference, a warning will be logged,
        and the function will fall back to single-scale inference.

        Args:
            input_tensor (Tensor): The input tensor to the model.

        Returns:
            Tensor: The output of the model after augmentation.
        """
        if not hasattr(self, "augment"):
            logger.warning(f"{self.__class__.__name__} does not support augmented inference. "
                           "Falling back to single-scale inference.")
            return self._predict_once(input_tensor)

    def _profile_one_layer(self, layer: nn.Module, input_data: Tensor, computation_times: list) -> None:
        r"""Profile the computation time and FLOPs (Floating Point Operations Per Second) of a single layer of the model on a given input.
        Appends the computation time to the provided list.

        Args:
            layer (nn.Module): The layer to be profiled.
            input_data (Tensor): The input data to the layer.
            computation_times (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        # Check if the layer is the final layer and if the input data is a list
        is_final_layer = layer == self.model[-1] and isinstance(input_data, list)

        # Calculate the FLOPs of the layer
        flops = thop.profile(layer, inputs=[input_data.copy() if is_final_layer else input_data], verbose=False)[0] / 1E9 * 2 if thop else 0

        # Record the start time
        start_time = time_sync()

        # Run the layer 10 times to get an average computation time
        for _ in range(10):
            layer(input_data.copy() if is_final_layer else input_data)

        # Calculate the elapsed time and append it to the computation times list
        elapsed_time = (time_sync() - start_time) * 100
        computation_times.append(elapsed_time)

        # Log the computation time, FLOPs, and parameters of the first layer
        if layer == self.model[0]:
            logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")

        # Log the computation time, FLOPs, and parameters of the current layer
        logger.info(f'{computation_times[-1]:10.2f} {flops:10.2f} {layer.np:10.0f}  {layer.type}')

        # If it's the final layer, log the total computation time
        if is_final_layer:
            logger.info(f"{sum(computation_times):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose: bool = True) -> nn.Module:
        r"""Fuse the Conv2d and BatchNorm2d layers of the model into a single layer to improve computation efficiency.

        Args:
            verbose (bool, optional): If True, print the summary. Defaults to True.

        Returns:
            nn.Module: The fused model.
        """
        if not self.is_fused():
            for module in self.model.modules():
                if isinstance(module, BasicConv2d) and hasattr(module, "bn"):
                    module.conv = fuse_conv_and_bn(module.conv, module.bn)  # update convolution layer
                    delattr(module, "bn")  # remove batch normalization layer
                    module.forward = module.forward_fuse  # update forward method
            self.info(verbose=verbose)

        return self

    def is_fused(self, threshold: int = 10) -> bool:
        r"""Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            threshold (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            bool: True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        normalization_layers = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(layer, normalization_layers) for layer in self.modules()) < threshold

    def info(self, image_size: int = 640, verbose: bool = True) -> Union[tuple, None]:
        r"""Prints model information.

        Args:
            image_size (int, optional): The size of the input image. Defaults to 640.
            verbose (bool, optional): If True, print the summary. Defaults to True.

        Returns:
            Union[tuple, None]: The model summary if verbose is True, otherwise None.
        """
        return model_summary(self, image_size, verbose)

    def _apply(self, function: Callable) -> 'BaseModel':
        r"""Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            function (Callable): The function to apply to the model.

        Returns:
            BaseModel: An updated BaseModel object.
        """
        self = super()._apply(function)
        last_module = self.model[-1]  # Detect()
        if isinstance(last_module, Detect):
            last_module.stride = function(last_module.stride)
            last_module.anchors = function(last_module.anchors)
            last_module.strides = function(last_module.strides)
        return self

    def load(self, weights: Union[dict, nn.Module], verbose: bool = True) -> None:
        r"""Load the weights into the model.

        Args:
            weights (Union[dict, nn.Module]): The pre-trained weights to be loaded.
            verbose (bool, optional): If True, log the transfer progress. Defaults to True.
        """
        model_weights = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        checkpoint_state_dict = model_weights.float().state_dict()  # checkpoint state_dict as FP32
        intersect_state_dict = {k: v for k, v in checkpoint_state_dict.items() if
                                k in self.state_dict() and all(x not in k for x in ()) and v.shape == self.state_dict()[k].shape}
        self.load_state_dict(intersect_state_dict, strict=False)  # load
        if verbose:
            logger.info(f"Transferred {len(intersect_state_dict)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch: dict, predictions: Tensor | List[Tensor] = None):
        r"""Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            predictions (Tensor | List[Tensor], optional): Predictions. If None, predictions are computed by forwarding the batch. Defaults to None.

        Returns:
            Tensor: The computed loss.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        predictions = self.forward(batch["image"]) if predictions is None else predictions
        return self.criterion(predictions, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")
