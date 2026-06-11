"""Shared hand-tracking layer for bbt/cursor_control.py + bbt/box_block_test.py.

Wraps existing inference code from webcam_detnet.py without modifying it.
Provides:
  - HandResult dataclass + HandTracker abstract base
  - MediaPipeTracker, DetNetTracker(variant=...) adapters
  - make_tracker(model_name, ...) factory shared by both scripts
  - pinch_distance + PinchState helpers (single-threshold + debounce, per plan §1.2)

Mirroring is the caller's responsibility: pass an already-mirrored BGR frame
to detect(); landmarks come back in that frame's coord system. Both scripts
mirror before calling detect(), so all downstream coordinates align with the
on-screen pixels with no extra flipping.
"""
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

# Locate the repo root (parent of bbt/) and make sure it's on sys.path so
# `import webcam_detnet` works whether this module is imported as
# `bbt.hand_tracking` or run from inside bbt/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from webcam_detnet import (  # noqa: E402  (sys.path is set above)
    HEATMAP_SIZE,
    get_hand_detection,
    load_model,
    load_quantized,
    preprocess,
)
from utils import smoother  # noqa: E402


# ── Public types ─────────────────────────────────────────────────────────────
@dataclass
class HandResult:
    landmarks_px: np.ndarray    # (21, 2) int32 — pixel coords in the input (mirrored) frame
    landmarks_norm: np.ndarray  # (21, 2 or 3) raw model-space coords (heatmap [row,col] for DetNet, [x,y,z] normalized for MediaPipe)
    inference_ms: float         # model forward-pass time only


class HandTracker:
    name: str = "?"

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        raise NotImplementedError

    def close(self) -> None:
        pass


def pinch_distance(landmarks_px: np.ndarray) -> float:
    """Euclidean distance between thumb_tip (joint 4) and index_tip (joint 8)
    in pixels. Both backends use SNAP-ordered 21 landmarks, so this works
    uniformly."""
    return float(np.linalg.norm(landmarks_px[4] - landmarks_px[8]))


class PinchState:
    """Pinch detector with optional asymmetric thresholds and debounces.

    Engage: distance < engage_threshold (= `threshold`) for `debounce`
        consecutive frames -> click engages.
    Release: distance > release_threshold (default = engage_threshold) for
        `release_debounce` (default = debounce) consecutive frames -> click
        releases.

    With release_threshold < engage_threshold (e.g. 20 vs 30), the click
    engages on a loose pinch (d < 30) but only stays held while the pinch
    remains tight (d <= 20); any sustained loosening above 20 triggers
    release. With release_threshold > engage_threshold the click is sticky.
    """
    def __init__(self, threshold: float = 30.0, debounce: int = 2,
                 release_debounce: Optional[int] = None,
                 release_threshold: Optional[float] = None):
        self.engage_threshold = float(threshold)
        self.release_threshold = (
            float(release_threshold) if release_threshold is not None
            else self.engage_threshold
        )
        self.engage_debounce = max(1, int(debounce))
        self.release_debounce = (
            max(1, int(release_debounce))
            if release_debounce is not None
            else self.engage_debounce
        )
        # Back-compat: some early code reads `pinch.threshold`.
        self.threshold = self.engage_threshold
        self.grabbing = False
        self._below = 0
        self._above = 0

    def update(self, distance: float) -> Tuple[bool, bool, bool]:
        """Returns (grabbing, just_engaged_this_frame, just_released_this_frame)."""
        engaged = released = False
        if not self.grabbing:
            # Engagement watch — distance below engage_threshold.
            if distance < self.engage_threshold:
                self._below += 1
                self._above = 0
                if self._below >= self.engage_debounce:
                    self.grabbing = True
                    engaged = True
            else:
                self._below = 0
        else:
            # Release watch — distance above release_threshold.
            if distance > self.release_threshold:
                self._above += 1
                self._below = 0
                if self._above >= self.release_debounce:
                    self.grabbing = False
                    released = True
            else:
                self._above = 0
        return self.grabbing, engaged, released

    def force_release(self) -> bool:
        """Drop any held grab unconditionally (e.g. when hand goes out of frame)."""
        was = self.grabbing
        self.grabbing = False
        self._below = self._above = 0
        return was


# ── MediaPipe model file (auto-download once) ────────────────────────────────
_MP_MODEL_FILENAME = 'hand_landmarker.task'
_MP_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
    'hand_landmarker/float16/1/hand_landmarker.task'
)


def _ensure_mp_model_file() -> str:
    # Match webcam_detnet.py's behaviour: keeps the model at repo root so a
    # single copy serves both scripts.
    here_first = os.path.join(_REPO_ROOT, _MP_MODEL_FILENAME)
    if os.path.isfile(here_first):
        return here_first
    if os.path.isfile(_MP_MODEL_FILENAME):  # fallback to cwd
        return _MP_MODEL_FILENAME
    print(f"Downloading MediaPipe hand landmarker -> {here_first} ...")
    urllib.request.urlretrieve(_MP_MODEL_URL, here_first)
    return here_first


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


# ── Trackers ────────────────────────────────────────────────────────────────
class MediaPipeTracker(HandTracker):
    name = 'mediapipe'

    def __init__(self):
        self._hands = _build_mp_landmarker()

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        import mediapipe as mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t0 = time.perf_counter()
        results = self._hands.detect(mp_image)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not results.hand_landmarks:
            return None
        lms = results.hand_landmarks[0]
        norm = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
        pts_xy = np.column_stack([
            (norm[:, 0] * w).astype(np.int32),
            (norm[:, 1] * h).astype(np.int32),
        ])
        return HandResult(
            landmarks_px=pts_xy,
            landmarks_norm=norm,
            inference_ms=elapsed_ms,
        )

    def close(self):
        self._hands.close()


class DetNetTracker(HandTracker):
    """variant in {'baseline','pruned','quantized'}.

    Constructor expectations:
      - baseline:  weights = state_dict .pth path; arch ignored.
      - pruned:    weights = fine-tuned state_dict, arch = matching _noft.pth shell.
      - quantized: weights ignored; quant_tag in {'qmse','qmm'}. ~3 min startup
                   as quant/qquant.py rebuilds the INT8 model deterministically.

    MediaPipe is run internally to produce the hand bbox each frame (matches
    webcam_detnet.py). inference_ms records ONLY the DetNet forward pass —
    the MediaPipe bbox call is bookkeeping overhead.
    """

    def __init__(self, variant: str, weights: Optional[str] = None,
                 arch: Optional[str] = None, quant_tag: str = 'qmse'):
        device = torch.device('cpu')
        if variant == 'baseline':
            if not weights:
                raise ValueError("variant='baseline' requires weights=<state_dict.pth>")
            self.module = load_model(weights, device, arch_path=None)
        elif variant == 'pruned':
            if not weights or not arch:
                raise ValueError("variant='pruned' requires weights=<ft_state_dict.pth> and arch=<_noft.pth>")
            self.module = load_model(weights, device, arch_path=arch)
        elif variant == 'quantized':
            if quant_tag not in ('qmm', 'qmse'):
                raise ValueError(f"variant='quantized' expects quant_tag in {{'qmm','qmse'}}, got {quant_tag!r}")
            self.module = load_quantized(quant_tag)
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

        uv = out['uv'].squeeze(0).cpu().numpy().astype(np.float32)  # (21,2) [row,col] in [0, HEATMAP_SIZE)
        uv = self._smoother.process(uv)
        pts = np.zeros((21, 2), dtype=np.int32)
        pts[:, 0] = (uv[:, 1] * region_w / HEATMAP_SIZE).astype(np.int32) + x1
        pts[:, 1] = (uv[:, 0] * region_h / HEATMAP_SIZE).astype(np.int32) + y1
        return HandResult(
            landmarks_px=pts,
            landmarks_norm=uv,
            inference_ms=elapsed_ms,
        )

    def close(self):
        self._hands.close()


# ── Factory ─────────────────────────────────────────────────────────────────
_DEFAULT_BASELINE_WEIGHTS = os.path.join(_REPO_ROOT, 'new_check_point', 'ckp_detnet_71.pth')


def make_tracker(model_name: str, *, weights: Optional[str] = None,
                 arch: Optional[str] = None, quant_tag: str = 'qmse') -> HandTracker:
    """Resolve a tracker by name. Defaults to in-repo paths where sensible.

    model_name in {'mediapipe', 'detnet-baseline', 'detnet-pruned', 'detnet-quantized'}.
    For pruned, BOTH weights and arch are required (no silent fallback).
    Raises SystemExit with a clear message on any misconfiguration.
    """
    if model_name == 'mediapipe':
        return MediaPipeTracker()

    if model_name == 'detnet-baseline':
        ckp = weights or _DEFAULT_BASELINE_WEIGHTS
        if not os.path.isfile(ckp):
            raise SystemExit(f"detnet-baseline: weights not found at '{ckp}'")
        return DetNetTracker('baseline', weights=ckp)

    if model_name == 'detnet-pruned':
        if not weights or not arch:
            raise SystemExit(
                "detnet-pruned requires --weights <ft_state_dict.pth> and --arch <_noft.pth>"
            )
        for label, p in (('weights', weights), ('arch', arch)):
            if not os.path.isfile(p):
                raise SystemExit(f"detnet-pruned: {label} not found at '{p}'")
        return DetNetTracker('pruned', weights=weights, arch=arch)

    if model_name == 'detnet-quantized':
        return DetNetTracker('quantized', quant_tag=quant_tag)

    raise SystemExit(f"unknown --model: {model_name!r}")


MODEL_CHOICES = ('mediapipe', 'detnet-baseline', 'detnet-pruned', 'detnet-quantized')
