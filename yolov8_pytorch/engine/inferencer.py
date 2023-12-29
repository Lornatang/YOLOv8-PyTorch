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
import platform
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from yolov8_pytorch.data import load_inference_source
from yolov8_pytorch.data.augment import LetterBox, classify_transforms
from yolov8_pytorch.nn.autobackend import AutoBackend
from yolov8_pytorch.utils import logger, MACOS, WINDOWS, callbacks, colorstr, ops
from yolov8_pytorch.utils.checks import check_imgsz, check_imshow
from yolov8_pytorch.utils.files import increment_path
from yolov8_pytorch.utils.torch_utils import select_device, smart_inference_mode


class InferencerEngine:
    def __init__(self, config_dict: DictConfig, _callbacks: Any = None):
        config_dict = config_dict.INFERENCE

        self.results_dir = config_dict.get("RESULTS_DIR")
        self.confidence_threshold = config_dict.get("CONFIDENCE_THRESHOLD")
        self.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.image_size = None
        self.device = None
        self.dataset = None
        self.video_path, self.video_writer, self.video_frame = None, None, None
        self.plotted_image = None
        self.data_path = None
        self.txt_path = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self._lock = threading.Lock()  # for automatic thread-safe inference
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        visualize = increment_path(self.results_dir / Path(self.batch[0][0]).stem,
                                   mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)

    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in im]

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.results_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.show_boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_image = result.plot(**plot_args)
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(results_dir=self.results_dir / 'crops',
                             file_name=self.data_path.stem + ('' if self.dataset.mode == 'image' else f'_{frame}'))

        return log_string

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions for an image and returns them."""
        return preds

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """
        Method used for CLI prediction.

        It uses always generator as outputs as not required by CLI mode.
        """
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None
        self.dataset = load_inference_source(source=source,
                                             imgsz=self.imgsz,
                                             video_stride=self.args.video_stride,
                                             buffer=self.args.stream_buffer)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'videoeo_flag', [False]))):  # videoeos
            logger.warning(".")
        self.video_path = [None] * self.dataset.bs
        self.video_writer = [None] * self.dataset.bs
        self.video_frame = [None] * self.dataset.bs

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            logger.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if results_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.results_dir / 'labels' if self.args.save_txt else self.results_dir).mkdir(parents=True, exist_ok=True)

            self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
            self.run_callbacks('on_predict_start')

            for batch in self.dataset:
                self.run_callbacks('on_predict_batch_start')
                self.batch = batch
                path, im0s, video_cap, s = batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)

                self.run_callbacks('on_predict_postprocess_end')
                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        'preprocess': profilers[0].dt * 1E3 / n,
                        'inference': profilers[1].dt * 1E3 / n,
                        'postprocess': profilers[2].dt * 1E3 / n}
                    p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                    p = Path(p)

                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s += self.write_results(i, self.results, (p, im, im0))
                    if self.args.save or self.args.save_txt:
                        self.results[i].results_dir = self.results_dir.__str__()
                    if self.args.show and self.plotted_image is not None:
                        self.show(p)
                    if self.args.save and self.plotted_image is not None:
                        self.save_preds(video_cap, i, str(self.results_dir / p.name))

                self.run_callbacks('on_predict_batch_end')
                yield from self.results

                # Print time (inference-only)
                if self.args.verbose:
                    logger.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(self.video_writer[-1], cv2.VideoWriter):
            self.video_writer[-1].release()  # release final videoeo writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in profilers)  # speeds per image
            logger.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *im.shape[2:])}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.results_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.results_dir / 'labels'}" if self.args.save_txt else ''
            logger.info(f"Results saved to {colorstr('bold', self.results_dir)}{s}")

        self.run_callbacks('on_predict_end')

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(model or self.args.model,
                                 device=select_device(self.args.device, verbose=verbose),
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()

    def show(self, p):
        """Display an image in a window using OpenCV imshow()."""
        im0 = self.plotted_image
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[3].startswith('image') else 1)  # 1 millisecond

    def save_preds(self, video_cap, idx, save_path):
        """Save videoeo predictions as mp4 at specified path."""
        im0 = self.plotted_image
        # Save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'videoeo' or 'stream'
            frames_path = f'{save_path.split(".", 1)[0]}_frames/'
            if self.video_path[idx] != save_path:  # new videoeo
                self.video_path[idx] = save_path
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                    self.video_frame[idx] = 0
                if isinstance(self.video_writer[idx], cv2.VideoWriter):
                    self.video_writer[idx].release()  # release previous videoeo writer
                if video_cap:  # videoeo
                    fps = int(video_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                suffix, fourcc = ('.mp4', 'avc1') if MACOS else ('.avi', 'WMV2') if WINDOWS else ('.avi', 'MJPG')
                self.video_writer[idx] = cv2.VideoWriter(str(Path(save_path).with_suffix(suffix)),
                                                       cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            # Write videoeo
            self.video_writer[idx].write(im0)

            # Write frame
            if self.args.save_frames:
                cv2.imwrite(f'{frames_path}{self.video_frame[idx]}.jpg', im0)
                self.video_frame[idx] += 1

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """Add callback."""
        self.callbacks[event].append(func)
