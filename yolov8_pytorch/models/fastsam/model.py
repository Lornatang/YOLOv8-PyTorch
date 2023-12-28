# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from yolov8_pytorch.engine.base_model import BaseModel

from .predict import FastSAMPredictor
from .val import FastSAMValidator


class FastSAM(BaseModel):
    """
    FastSAM model interface.

    Example:
        ```python
        from yolov8_pytorch import FastSAM

        model = FastSAM('last.pt')
        results = model.predict('yolov8_pytorch/assets/bus.jpg')
        ```
    """

    def __init__(self, config_dict='FastSAM-x.pt'):
        """Call the __init__ method of the parent class (YOLO) with the updated default model."""
        if str(config_dict) == 'FastSAM.pt':
            config_dict = 'FastSAM-x.pt'
        assert Path(config_dict).suffix not in ('.yaml', '.yml'), 'FastSAM models only support pre-trained models.'
        super().__init__(config_dict=config_dict, task='segment')

    @property
    def task_map(self):
        """Returns a dictionary mapping segment task to corresponding predictor and validator classes."""
        return {'segment': {'predictor': FastSAMPredictor, 'validator': FastSAMValidator}}
