# Ultralytics YOLO 🚀, AGPL-3.0 license

import logging
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from yolov8_pytorch.cfg import get_cfg, get_save_dir
from yolov8_pytorch.nn.tasks import attempt_load_one_weight, nn
from yolov8_pytorch.utils import ASSETS, DEFAULT_CFG_DICT, LOGGER, RANK, callbacks
from yolov8_pytorch.utils.common import load_weights

logger = logging.getLogger(__name__)


class ModelEngine(nn.Module):
    def __init__(self, config_dict: DictConfig, task: str, verbose: bool = False) -> None:

        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.model = None
        self.inferencer = None
        self.trainer = None
        self.checkpoint = None
        self.checkpoint_path = None
        self.metrics = None
        self.overrides = {}

        self.config_dict = config_dict
        self.task = task  # task type

        # create YOLO model
        self.model = self._load_engine("model")(config_dict.MODEL, verbose=verbose and RANK == -1)
        self.model.config = config_dict
        self.model.task = task

    def __call__(self, source: Any = None, stream: bool = False, **kwargs):
        return self.predict(source, stream, **kwargs)

    def load(self, weights: str):
        """Transfers parameters with matching names and shapes from 'weights' to model."""
        if isinstance(weights, (str, Path)):
            self.model = load_weights(weights)
            self.checkpoint = True


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

    def embed(self, source=None, stream=False, **kwargs):
        """
        Calls the predict() method and returns image embeddings.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[torch.Tensor]): A list of image embeddings.
        """
        if not kwargs.get('embed'):
            kwargs['embed'] = [len(self.model.model) - 2]  # embed second-to-last layer if no indices passed
        return self.predict(source, stream, **kwargs)

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
            (List[yolov8_pytorch.engine.results.Results]): The prediction results.
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('yolov8_pytorch')) and any(
            x in sys.argv for x in ('predict', 'track', 'mode=predict', 'mode=track'))

        custom = {'conf': 0.25, 'save': is_cli}  # method defaults
        args = {**self.overrides, **custom, **kwargs, 'mode': 'predict'}  # highest priority args on the right
        prompts = args.pop('prompts', None)  # for SAM-type models

        if not self.inferencer:
            self.inferencer = (predictor or self._load_engine('predictor'))(overrides=args, _callbacks=self.callbacks)
            self.inferencer.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.inferencer.args = get_cfg(self.inferencer.args, args)
            if 'project' in args or 'name' in args:
                self.inferencer.save_dir = get_save_dir(self.inferencer.args)
        if prompts and hasattr(self.inferencer, 'set_prompts'):  # for SAM-type models
            self.inferencer.set_prompts(prompts)
        return self.inferencer.predict_cli(source=source) if is_cli else self.inferencer(source=source, stream=stream)

    def val(self, validator=None, **kwargs):
        """
        Validate a model on a given dataset.

        Args:
            validator (BaseValidator): Customized validator.
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        custom = {'rect': True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, 'mode': 'val'}  # highest priority args on the right

        validator = (validator or self._load_engine('validator'))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(self, **kwargs):
        """
        Benchmark a model on all export formats.

        Args:
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        from yolov8_pytorch.utils.benchmarks import benchmark

        custom = {'verbose': False}  # method defaults
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, 'mode': 'benchmark'}
        return benchmark(
            model=self,
            data=kwargs.get('data'),  # if no 'data' argument passed set data=None for default datasets
            imgsz=args['imgsz'],
            half=args['half'],
            int8=args['int8'],
            device=args['device'],
            verbose=kwargs.get('verbose'))

    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the Exporter. To see all args check 'configuration' section in docs.
        """
        from .exporter import Exporter

        custom = {'imgsz': self.model.args['imgsz'], 'batch': 1, 'data': None, 'verbose': False}  # method defaults
        args = {**self.overrides, **custom, **kwargs, 'mode': 'export'}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(self, trainer=None, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            trainer (BaseTrainer, optional): Customized trainer.
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        if self.config_dict.TRAIN.get('resume'):
            self.config_dict.TRAIN['resume'] = self.checkpoint_path

        self.trainer = (trainer or self._load_engine('trainer'))(cfg=self.config_dict, overrides=self.overrides, _callbacks=self.callbacks)
        if not self.config_dict.TRAIN.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.checkpoint else None, cfg=self.config_dict.MODEL)
            self.model = self.trainer.model
        self.trainer.train()
        # Update model and cfg after training
        if RANK in (-1, 0):
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP
        return self.metrics

    def tune(self, use_ray=False, iterations=10, *args, **kwargs):
        """
        Runs hyperparameter tuning, optionally using Ray Tune. See yolov8_pytorch.utils.tuner.run_ray_tune for Args.

        Returns:
            (dict): A dictionary containing the results of the hyperparameter search.
        """
        if use_ray:
            from yolov8_pytorch.utils.tuner import run_ray_tune
            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # method defaults
            args = {**self.overrides, **custom, **kwargs, 'mode': 'train'}  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _load_engine(self, key):
        return self.task_map[self.task][key]

    def _apply(self, fn):
        """Apply to(), cpu(), cuda(), half(), float() to model tensors that are not parameters or registered buffers."""
        self = super()._apply(fn)  # noqa
        self.inferencer = None  # reset predictor as device may have changed
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
        """Clear all event callbacks."""
        self.callbacks[event] = []

    def reset_callbacks(self):
        """Reset all registered callbacks."""
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]
