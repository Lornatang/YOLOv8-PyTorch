# Ultralytics YOLO 🚀, AGPL-3.0 license

from yolov8_pytorch.engine.model import ModelEngine
from yolov8_pytorch.models import yolo  # noqa
from yolov8_pytorch.nn.tasks import DetectionModel, SegmentationModel


class YOLOEngine(ModelEngine):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionInferencer, },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor, },
        }
