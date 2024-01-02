# Ultralytics YOLO 🚀, AGPL-3.0 license

from yolov8_pytorch.models.yolo import classify, detect, pose, segment

from .model import YOLO

__all__ = 'classify', 'segment', 'detect', 'pose', 'YOLO'
