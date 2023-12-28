# Copyright 2023 AlphaBetter Corporation. All Rights Reserved.
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
from omegaconf import OmegaConf

from yolov8_pytorch.models import YOLO
from yolov8_pytorch.nn.tasks import create_model_from_yaml

model_config_path = "../configs/COCO-Detection/yolov8n-ours.yaml"

model_config = OmegaConf.load(model_config_path)
model_config = OmegaConf.create(model_config)

if __name__ == "__main__":
    # model = create_model_from_yaml(model_config.MODEL, True)
    model = YOLO(model_config, task="detect", verbose=True).load("yolov8n.pt")
