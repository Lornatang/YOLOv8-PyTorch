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
# Ultralytics YOLO 🚀, AGPL-3.0 license

from yolov8_pytorch_old.engine import DetectionPredictor, DetectionTrainer, DetectionValidator
from .base_model import Model
from .detect_model import DetectionModel


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {

            'detect': {
                'model': DetectionModel,
                'trainer': DetectionTrainer,
                'validator': DetectionValidator,
                'predictor': DetectionPredictor, },
        }
