"""HandTracker abstract base + HandResult dataclass.

All backends (MediaPipe, DetNet baseline/pruned/quantized) return the same
HandResult so the BBT game loop is backend-agnostic. Landmarks are always in
the coordinate system of the (already-mirrored) frame the caller passed in.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HandResult:
    landmarks_px: np.ndarray  # (21, 2) int32 — pixel coords in the input frame
    inference_ms: float       # model forward-pass time only (excludes bbox / pre / post)


class HandTracker:
    name: str = "?"

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        raise NotImplementedError

    def close(self) -> None:
        pass