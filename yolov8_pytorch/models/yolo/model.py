# Ultralytics YOLO 🚀, AGPL-3.0 license

from yolov8_pytorch.engine.base_model import BaseModel
from yolov8_pytorch.models import yolo  # noqa
from yolov8_pytorch.nn.tasks import DetectionModel, SegmentationModel


class YOLO(BaseModel):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'detect': {
                'model': DetectionModel,
                'trainer': yolo.detect.DetectionTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': yolo.detect.DetectionPredictor, },
            'segment': {
                'model': SegmentationModel,
                'trainer': yolo.segment.SegmentationTrainer,
                'validator': yolo.segment.SegmentationValidator,
                'predictor': yolo.segment.SegmentationPredictor, },
        }
