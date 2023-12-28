# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
YOLO-NAS model interface.

Example:
    ```python
    from yolov8_pytorch import YOLONAS

    model = NAS('yolo_nas_s')
    results = model.predict('yolov8_pytorch/assets/bus.jpg')
    ```
"""

from pathlib import Path

import super_gradients
import torch

from yolov8_pytorch.engine.base_model import BaseModel
from yolov8_pytorch.utils.torch_utils import model_info, smart_inference_mode
from .predict import NASPredictor
from .val import NASValidator


class YOLONAS(BaseModel):
    """
    YOLO NAS model for object detection.

    This class provides an interface for the YOLO-NAS models and extends the `Model` class from Ultralytics engine.
    It is designed to facilitate the task of object detection using pre-trained or custom-trained YOLO-NAS models.

    Example:
        ```python
        from yolov8_pytorch import NAS

        model = NAS('yolo_nas_s')
        results = model.predict('yolov8_pytorch/assets/bus.jpg')
        ```

    Attributes:
        model (str): Path to the pre-trained model or model name. Defaults to 'yolo_nas_s.pt'.

    Note:
        YOLO-NAS models only support pre-trained models. Do not provide YAML configuration files.
    """

    def __init__(self, config_dict='yolo_nas_s.pt') -> None:
        """Initializes the NAS model with the provided or default 'yolo_nas_s.pt' model."""
        assert Path(config_dict).suffix not in ('.yaml', '.yml'), 'YOLO-NAS models only support pre-trained models.'
        super().__init__(config_dict, task='detect')

    @smart_inference_mode()
    def _load(self, weights: str, task: str):
        """Loads an existing NAS model weights or creates a new NAS model with pretrained weights if not provided."""
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model = torch.load(weights)
        elif suffix == '':
            self.model = super_gradients.training.models.get(weights, pretrained_weights='coco')
        # Standardize model
        self.model.fuse = lambda verbose=True: self.model
        self.model.stride = torch.tensor([32])
        self.model.names = dict(enumerate(self.model._class_names))
        self.model.is_fused = lambda: False  # for info()
        self.model.yaml = {}  # for info()
        self.model.pt_path = weights  # for export()
        self.model.task = 'detect'  # for export()

    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    @property
    def task_map(self):
        """Returns a dictionary mapping tasks to respective predictor and validator classes."""
        return {'detect': {'predictor': NASPredictor, 'validator': NASValidator}}
