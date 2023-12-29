# Ultralytics YOLO 🚀, AGPL-3.0 license

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from yolov8_pytorch.nn.tasks import nn
from yolov8_pytorch.utils import LOGGER, RANK, callbacks
from yolov8_pytorch.utils.benchmarks import benchmark
from yolov8_pytorch.utils.common import load_weights
from yolov8_pytorch.utils.tuner import run_ray_tune
from .tuner import Tuner

logger = logging.getLogger(__name__)


class ModelEngine(nn.Module):
    """The base class of all model engines, including training, verification, reasoning, fine-tuning, and deriving methods

    Attributes:
        model: The model to be used.
        inferencer: The inferencer to be used.
        trainer: The trainer to be used.
        checkpoint: The checkpoint to be used.
        checkpoint_path: The path of checkpoint to be used.
        metrics: The metrics to be used.
        callbacks: The callbacks to be used.
    """

    def __init__(self, config_dict: DictConfig, task: str = None, verbose: bool = False) -> None:
        r"""Initialize the model engine.

        Args:
            config_dict (DictConfig): The config_dict to be used.
            task (str, optional): The task to be used. Defaults to None.
            verbose (bool, optional): Whether to print the log information. Defaults to False.
        """
        super().__init__()
        self.config = config_dict
        self.verbose = verbose

        # Map head to model, trainer, validator, and predictor classes.
        self.callbacks = callbacks.get_default_callbacks()

        self.model = None
        self.inferencer = None
        self.trainer = None
        self.checkpoint = None
        self.checkpoint_path = None
        self.metrics = None

        # create model engine
        self.model = self._load_engine("model")(config_dict.MODEL, verbose=verbose and RANK == -1)
        self.model.config_dict = config_dict
        self.model.task = task

    def __call__(self, source: Any = None, stream: bool = False, **kwargs) -> Any:
        r"""As same as forward()

        Args:
            source (Any, optional): The source of the image to make predictions on. Defaults to None.
            stream (bool, optional): Whether to stream the predictions or not. Defaults to False.

        Returns:
            Any: The prediction results.
        """
        return self.inference(source, stream, **kwargs)

    def info(self, verbose: bool = True) -> None:
        r"""Print model information.

        Args:
            verbose (bool, optional): Whether to print the log information. Defaults to True.
        """
        return self.model.info(verbose)

    def load(self, checkpoint_path: str | Path) -> None:
        r"""Load the model.

        Args:
            checkpoint_path (str | Path): The path of weights to be loaded.
        """
        self.model = load_weights(checkpoint_path)
        self.checkpoint = True
        self.checkpoint_path = checkpoint_path

    def fuse(self) -> None:
        r"""Fuse Conv2d + BatchNorm2d layers throughout model for inference."""
        self.model.fuse()

    def embed(self, source: Any = None, stream: bool = False, **kwargs) -> Any:
        r"""Perform prediction using the YOLO model.

        Args:
            source (Any, optional): The source of the image to make predictions on. Defaults to None.
            stream (bool, optional): Whether to stream the predictions or not. Defaults to False.

        Returns:
            Any: The prediction results.
        """
        if not kwargs.get("embed"):
            # default to last layer
            kwargs["embed"] = [len(self.model.model) - 2]
        return self.inference(source, stream, **kwargs)

    def inference(self, source: Any = None, stream: bool = False, inferencer: Any = None, **kwargs) -> Any:
        r"""Perform prediction using the YOLO model.

        Args:
            source (Any, optional): The source of the image to make predictions on. Defaults to None.
            stream (bool, optional): Whether to stream the predictions or not. Defaults to False.
            inferencer (Any, optional): The inferencer to be used. Defaults to None.

        Returns:
            Any: The prediction results.
        """
        if source is None:
            LOGGER.error(f"'source' is missing.")

        self.inferencer = (inferencer or self._load_engine("inferencer"))(config_dict=self.config_dict, _callbacks=self.callbacks)

        # Add mode to kwargs
        self.config_dict["MODE"] = "INFERENCE"

        self.inferencer.config_dict = self.config_dict.INFERENCE

        # for SAM-type model
        prompts = self.inferencer.config_dict.get("PROMPTS")
        if prompts:
            self.inferencer.set_prompts(prompts)
        return self.inferencer(source=source, stream=stream)

    def train(self, trainer: Any = None, **kwargs) -> Any:
        r"""Trains the model on a given dataset.

        Args:
            trainer (BaseTrainer, optional): Customized trainer.
            **kwargs (Any): Any number of arguments representing the training configuration.

        Returns:
            Any: The metrics of the trained model.
        """
        self.trainer = (trainer or self._load_engine("trainer"))(config_dict=self.config_dict, _callbacks=self.callbacks)

        # Add mode to kwargs
        self.config_dict["MODE"] = "TRAIN"

        self.trainer.config_dict = self.config_dict.TRAIN

        if not self.config_dict.get("CHECKPOINT_PATH"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(config_dict=self.config_dict, weights=self.model if self.checkpoint else None)
            self.model = self.trainer.model

        self.trainer.train()

        # Update model and cfg after training
        if RANK in (-1, 0):
            checkpoint_path = self.trainer.best_checkpoint_path if self.trainer.best_checkpoint_path.exists() else self.trainer.last_checkpoint_path
            self.model = load_weights(checkpoint_path)
            self.metrics = getattr(self.trainer.validator, "metrics", None)
        return self.metrics

    def validate(self, validater: Any = None, **kwargs) -> Any:
        r"""Validate a model on a given dataset.

        Args:
            validater (BaseValidator): Customized validator.
            **kwargs : Any other args accepted by the validators. To see all args check "configuration" section in docs

        Returns:
            Any: The metrics of the trained model.
        """
        validater = (validater or self._load_engine("validator"))(config_dict=self.config_dict, _callbacks=self.callbacks)

        # Add mode to kwargs
        self.config_dict["MODE"] = "VALIDATE"

        validater.config_dict = self.config_dict.VALIDATE

        validater(model=self.model)
        self.metrics = validater.metrics
        return validater.metrics

    def tune(self, use_ray: bool = False, iterations: int = 10, *args, **kwargs) -> dict:
        r"""Runs hyperparameter tuning, optionally using Ray Tune. See yolov8_pytorch.utils.tuner.run_ray_tune for Args.

        Args:
            use_ray (bool): Whether to use Ray Tune for hyperparameter tuning. Defaults to False.
            iterations (int): Number of iterations to run. Defaults to 10.
            *args: Any other args accepted by the tuner. To see all args check "configuration" section in docs
            **kwargs: Any other args accepted by the tuner. To see all args check "configuration" section in docs

        Returns:
            (dict): A dictionary containing the results of the hyperparameter search.
        """
        if use_ray:
            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            # Add mode to kwargs
            self.config_dict["MODE"] = "TRAIN"
            return Tuner(config_dict=self.config_dict, _callbacks=self.callbacks)(model=self.model, iterations=iterations)

    def benchmark(self, verbose: bool = False, **kwargs):
        r"""Benchmark a model on all export formats.

        Args:
            verbose (bool): Print results to screen. Defaults to False.
            **kwargs : Any other args accepted by the validators. To see all args check "configuration" section in docs

        Args:
            **kwargs : Any other args accepted by the validators. To see all args check "configuration" section in docs
        """
        # Add mode to kwargs
        self.config_dict["MODE"] = "BENCHMARK"

        return benchmark(
            model=self.model,
            data=self.config_dict.BENCHMARK.get("DATASETS"),
            imgsz=self.config_dict.BENCHMARK.get("IMAGE_SIZE"),
            half=self.config_dict.BENCHMARK.get("HALF"),
            int8=self.config_dict.BENCHMARK.get("INT8"),
            device=self.config_dict.get("DEVICE"),
            verbose=verbose)

    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the Exporter. To see all args check "configuration" section in docs.
        """
        from .exporter import Exporter

        custom = {"imgsz": self.model.args["imgsz"], "batch": 1, "data": None, "verbose": False}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "MODE": "export"}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def _load_engine(self, key):
        return self.task_map[self.task][key]

    def _apply(self, func):
        """Apply to(), cpu(), cuda(), half(), float() to model tensors that are not parameters or registered buffers."""
        self = super()._apply(func)  # noqa
        self.inferencer = None  # reset inferencer as device may have changed
        self.config_dict["device"] = self.device  # was str(self.device) i.e. device(type="cuda", index=0) -> "cuda:0"
        return self

    @property
    def names(self):
        """Returns class names of the loaded model."""
        return self.config_dict.DATASETS.get("NAMES")

    @property
    def device(self):
        """Returns device if PyTorch model."""
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        """Returns transform of the loaded model."""
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func: Any) -> None:
        """Add a callback."""
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """Clear all event callbacks."""
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """Reset all registered callbacks."""
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]
