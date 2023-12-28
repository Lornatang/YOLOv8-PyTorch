# Ultralytics YOLO 🚀, AGPL-3.0 license

import inspect
import logging
import sys
from pathlib import Path
from typing import Union

from yolov8_pytorch.cfg import TASK2DATA, get_cfg, get_save_dir
from yolov8_pytorch.nn.tasks import attempt_load_one_weight, nn, yaml_model_load
from yolov8_pytorch.utils import ASSETS, DEFAULT_CFG_DICT, LOGGER, RANK, callbacks, checks, yaml_load
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    """
    A base class to unify APIs for all models.

    Args:
        config_dict (str, Path): Path to the model file to load or create.
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

    def __init__(self, config_dict: DictConfig, task: str, verbose: bool = True) -> None:
        """
        Initializes the YOLO model.

        Args:
            config_dict (Union[str, Path], optional): Path or name of the model to load or create. Defaults to 'yolov8n.pt'.
        """
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.task = task  # task type

        # create new YOLO model
        self.cfg = config_dict
        self.model = self._smart_load('model')(config_dict.MODEL, verbose=verbose and RANK == -1)  # build model
        self.overrides['model'] = self.cfg
        self.overrides['task'] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the predict() method with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def load(self, weights='yolov8n.pt'):
        """Transfers parameters with matching names and shapes from 'weights' to model."""
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
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

        if not self.predictor:
            self.predictor = (predictor or self._smart_load('predictor'))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if 'project' in args or 'name' in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, 'set_prompts'):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

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
        if self.cfg.TRAIN.get('resume'):
            self.cfg.TRAIN['resume'] = self.ckpt_path

        self.trainer = (trainer or self._smart_load('trainer'))(overrides=self.cfg.TRAIN, _callbacks=self.callbacks)
        if not self.cfg.TRAIN.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
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

    def _apply(self, fn):
        """Apply to(), cpu(), cuda(), half(), float() to model tensors that are not parameters or registered buffers."""
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
        """Clear all event callbacks."""
        self.callbacks[event] = []

    def reset_callbacks(self):
        """Reset all registered callbacks."""
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    def _smart_load(self, key):
        """Load model/trainer/validator/predictor."""
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                logger.warning(f"'{name}' model does not support '{mode}' mode for '{self.task}' task.")) from e

    @property
    def task_map(self):
        """
        Map head to model, trainer, validator, and predictor classes.

        Returns:
            task_map (dict): The map of model task to mode classes.
        """
        raise NotImplementedError('Please provide task map for your model!')
