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
import inspect
import logging
import sys
from pathlib import Path
from typing import Union, Callable, List

import thop
from omegaconf import OmegaConf
import os
from torch import nn, Tensor

from yolov8_pytorch_old.utils.callbacks import get_default_callbacks, default_callbacks
from yolov8_pytorch_old.utils.misc import fuse_conv_and_bn, model_summary, time_sync, get_save_dir
from yolov8_pytorch_old.utils.visualizer import generate_feature_maps, save_feature_maps
from .module import BasicConv2d, Detect

__all__ = [
    "BaseModel",
]

logger = logging.getLogger(__name__)


RANK = int(os.getenv('RANK', -1))

class Model(nn.Module):
    """
    A base class to unify APIs for all models.

    Args:
        model (str, Path): Path to the model file to load or create.
        task (Any, optional): Task type for the YOLO model. Defaults to None.

    Attributes:
        predictor (Any): The predictor object.
        model (Any): The model object.
        trainer (Any): The trainer object.
        task (str): The type of model task.
        ckpt (Any): The checkpoint object if the model loaded from *.pt file.
        cfg (str): The model configuration if loaded from *.yaml file.
        ckpt_path (str): The checkpoint file path.
        overrides (dict): Overrides for the trainer object.
        metrics (Any): The data for metrics.

    Methods:
        __call__(source=None, stream=False, **kwargs):
            Alias for the predict method.
        _new(cfg:str, verbose:bool=True) -> None:
            Initializes a new model and infers the task type from the model definitions.
        _load(weights:str, task:str='') -> None:
            Initializes a new model and infers the task type from the model head.
        _check_is_pytorch_model() -> None:
            Raises TypeError if the model is not a PyTorch model.
        reset() -> None:
            Resets the model modules.
        info(verbose:bool=False) -> None:
            Logs the model info.
        fuse() -> None:
            Fuses the model for faster inference.
        predict(source=None, stream=False, **kwargs) -> List[yolov8_pytorch.engine.results.Results]:
            Performs prediction using the YOLO model.

    Returns:
        list(yolov8_pytorch.engine.results.Results): The prediction results.
    """

    def __init__(self, model: Union[str, Path] = 'yolov8n.pt', task=None) -> None:
        """
        Initializes the YOLO model.

        Args:
            model (Union[str, Path], optional): Path or name of the model to load or create. Defaults to 'yolov8n.pt'.
            task (Any, optional): Task type for the YOLO model. Defaults to None.
        """
        super().__init__()
        self.callbacks = get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        self.task = task  # task type
        model = str(model).strip()  # strip spaces

        # Load or create new YOLO model
        if Path(model).suffix in ('.yaml', '.yml'):
            self._new(model, task)
        else:
            self._load(model, task)

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def _new(self, cfg: str, task=None, model=None, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str | None): model task
            model (BaseModel): Customized model.
            verbose (bool): display model info on load
        """
        self.cfg = cfg
        self.task = "detect"
        config = OmegaConf.load(cfg)
        self.config = OmegaConf.create(config)
        model_config = config.MODEL
        self.model = (model or self._smart_load('model'))(model_config, verbose=verbose)  # build model

        # Below added to allow export from YAMLs
        self.model.args = self.config
        self.model.task = self.task

    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.task = self.model.args['task']
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            self.model, self.ckpt = weights, None
            self.task = task
            self.ckpt_path = weights

    def _check_is_pytorch_model(self):
        """Raises TypeError is model is not a PyTorch model."""
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == '.pt'
        pt_module = isinstance(self.model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'")

    def reset_weights(self):
        """Resets the model modules parameters to randomly initialized values, losing all training information."""
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights='yolov8n.pt'):
        """Transfers parameters with matching names and shapes from 'weights' to model."""
        self.model.load(weights)
        return self

    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        """Fuse PyTorch Conv2d and BatchNorm2d layers."""
        self.model.fuse()

    def predict(self, source=None, stream=False, predictor=None, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            predictor (BasePredictor): Customized predictor.
            **kwargs : Additional keyword arguments passed to the predictor.
                Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.engine.results.Results]): The prediction results.
        """
        if source is None:
            logger.error(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('yolov8_pytorch')) and any(
            x in sys.argv for x in ('predict', 'track', 'mode=predict', 'mode=track'))

        custom = {'conf': 0.25, 'save': is_cli}  # method defaults
        args = {**self.overrides, **custom, **kwargs, 'mode': 'predict'}  # highest priority args on the right
        prompts = args.pop('prompts', None)  # for SAM-type models

        if not self.predictor:
            self.predictor = (predictor or self._smart_load('predictor'))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = self.args
            if 'project' in args or 'name' in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, 'set_prompts'):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(self, source=None, stream=False, persist=False, **kwargs):
        """
        Perform object tracking on the input source using the registered trackers.

        Args:
            source (str, optional): The input source for object tracking. Can be a file path or a video stream.
            stream (bool, optional): Whether the input source is a video stream. Defaults to False.
            persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
            **kwargs (optional): Additional keyword arguments for the tracking process.

        Returns:
            (List[ultralytics.engine.results.Results]): The tracking results.
        """
        if not hasattr(self.predictor, 'trackers'):
            from yolov8_pytorch.trackers import register_tracker
            register_tracker(self, persist)
        kwargs['conf'] = kwargs.get('conf') or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs['mode'] = 'track'
        return self.predict(source=source, stream=stream, **kwargs)

    def val(self, validator=None, **kwargs):
        """
        Validate a model on a given dataset.

        Args:
            validator (BaseValidator): Customized validator.
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        custom = {'rect': True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, 'mode': 'val'}  # highest priority args on the right

        validator = (validator or self._smart_load('validator'))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def train(self, trainer=None, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            trainer (BaseTrainer, optional): Customized trainer.
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        custom = {'data': self.cfg.data}  # method defaults
        if self.config.get('resume'):
            self.config.resume = self.ckpt_path

        self.trainer = (trainer or self._smart_load('trainer'))(overrides=self.config, _callbacks=self.callbacks)
        if not self.config.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in (-1, 0):
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP
        return self.metrics

    def tune(self, use_ray=True, iterations=10, *args, **kwargs):
        """
        Runs hyperparameter tuning, optionally using Ray Tune. See yolov8_pytorch.utils.tuner.run_ray_tune for Args.

        Returns:
            (dict): A dictionary containing the results of the hyperparameter search.
        """
        self._check_is_pytorch_model()
        if use_ray:
            from yolov8_pytorch.utils.tuner import run_ray_tune
            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)

    def _apply(self, fn):
        """Apply to(), cpu(), cuda(), half(), float() to model tensors that are not parameters or registered buffers."""
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # reset predictor as device may have changed
        self.overrides['device'] = self.device  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self):
        """Returns class names of the loaded model."""
        return self.model.names if hasattr(self.model, 'names') else None

    @property
    def device(self):
        """Returns device if PyTorch model."""
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        """Returns transform of the loaded model."""
        return self.model.transforms if hasattr(self.model, 'transforms') else None

    def add_callback(self, event: str, func):
        """Add a callback."""
        self.callbacks[event].append(func)

    def clear_callback(self, event: str):
        """Clear all event """
        self.callbacks[event] = []

    def reset_callbacks(self):
        """Reset all registered """
        for event in default_callbacks.keys():
            self.callbacks[event] = [default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _smart_load(self, key):
        """Load model/trainer/validator/predictor."""
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")

    @property
    def task_map(self):
        """
        Map head to model, trainer, validator, and predictor classes.

        Returns:
            task_map (dict): The map of model task to mode classes.
        """
        raise NotImplementedError('Please provide task map for your model!')


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
