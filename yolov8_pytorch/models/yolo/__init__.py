# Ultralytics YOLO 🚀, AGPL-3.0 license

from yolov8_pytorch.models.yolo import detect, segment

from .model import YOLOEngine

__all__ = 'segment', 'detect', 'YOLOEngine'
