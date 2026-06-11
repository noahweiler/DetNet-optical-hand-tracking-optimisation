"""Concrete HandTracker implementations.

All backends share the MediaPipe HandLandmarker for the hand bounding box
(matches the existing webcam_detnet.py pipeline). The DetNet variants then run
the model's forward pass on the cropped bbox; MediaPipe-only returns the
landmarks directly.

Nothing here modifies webcam_detnet.py — load_model, load_quantized,
get_hand_detection, and preprocess are imported and reused unchanged.
"""
import os
import time
import urllib.request
from typing import Optional

import cv2
import numpy as np
import torch

from webcam_detnet import (
    HEATMAP_SIZE,
    get_hand_detection,
    load_model,
    load_quantized,
    preprocess,
)
from utils import smoother

from tracking.interface import HandResult, HandTracker


_MP_MODEL_PATH = 'hand_landmarker.task'
_MP_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
    'hand_landmarker/float16/1/hand_landmarker.task'
)


def _ensure_mp_model_file() -> str:
    if not os.path.isfile(_MP_MODEL_PATH):
        print(f"Downloading MediaPipe hand landmarker -> {_MP_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MP_MODEL_URL, _MP_MODEL_PATH)
    return _MP_MODEL_PATH


def _build_mp_landmarker():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    return mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_ensure_mp_model_file()),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
    )


class MediaPipeTracker(HandTracker):
    name = 'mediapipe'

    def __init__(self):
        self._hands = _build_mp_landmarker()

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        _, pts_xy = get_hand_detection(rgb, self._hands)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if pts_xy is None:
            return None
        return HandResult(
            landmarks_px=pts_xy.astype(np.int32, copy=False),
            inference_ms=elapsed_ms,
        )

    def close(self):
        self._hands.close()


class DetNetTracker(HandTracker):
    """DetNet wrapper. variant in {'baseline', 'pruned', 'quantized'}.

    - baseline:  checkpoint = path to state_dict (.pth); arch ignored.
    - pruned:    checkpoint = fine-tuned state_dict, arch = matching _noft.pth shell.
    - quantized: checkpoint = qquant tag ('qmm' or 'qmse'; default 'qmse');
                 arch ignored. ~3 min startup as qquant.quantize() runs.

    inference_ms records only the DetNet forward pass. The MediaPipe bbox call
    is bookkeeping overhead, like data loading.
    """

    def __init__(self, variant: str, checkpoint: Optional[str] = None,
                 arch: Optional[str] = None):
        device = torch.device('cpu')
        if variant == 'baseline':
            if not checkpoint:
                raise ValueError("variant='baseline' requires checkpoint=<path/to/state_dict.pth>")
            self.module = load_model(checkpoint, device, arch_path=None)
        elif variant == 'pruned':
            if not checkpoint or not arch:
                raise ValueError("variant='pruned' requires checkpoint=<ft_state_dict.pth> and arch=<_noft.pth>")
            self.module = load_model(checkpoint, device, arch_path=arch)
        elif variant == 'quantized':
            tag = checkpoint or 'qmse'
            if tag not in ('qmm', 'qmse'):
                raise ValueError(f"variant='quantized' expects tag 'qmm' or 'qmse', got {tag!r}")
            self.module = load_quantized(tag)
        else:
            raise ValueError(f"unknown variant: {variant!r}")
        self.module.eval()
        self.device = device
        self.variant = variant
        self.name = f'detnet-{variant}'
        self._hands = _build_mp_landmarker()
        self._smoother = smoother.OneEuroFilter(4.0, 0.0)

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bbox, _ = get_hand_detection(rgb, self._hands)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        region_w = x2 - x1
        region_h = y2 - y1
        if region_w <= 0 or region_h <= 0:
            return None

        input_tensor = preprocess(frame_bgr, self.device, crop=bbox)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.module(input_tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        uv = out['uv'].squeeze(0).cpu().numpy().astype(np.float32)  # (21, 2) in [0, HEATMAP_SIZE) — [row, col]
        uv = self._smoother.process(uv)
        pts = np.zeros((21, 2), dtype=np.int32)
        pts[:, 0] = (uv[:, 1] * region_w / HEATMAP_SIZE).astype(np.int32) + x1
        pts[:, 1] = (uv[:, 0] * region_h / HEATMAP_SIZE).astype(np.int32) + y1
        return HandResult(landmarks_px=pts, inference_ms=elapsed_ms)

    def close(self):
        self._hands.close()