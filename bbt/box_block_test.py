"""Box and Block Test (BBT) — gamified single-camera hand-pose evaluation.

A 60-second game: pinch (thumb + index) to grab a block, drag it across the
centre partition, release to drop it on the target side. Score = blocks
delivered. Records the mirrored composite view to ``BBT recordings/`` along
with a per-frame CSV containing all 21 hand landmarks.

Backends — all CPU. Commands are single-line so they paste straight into
PowerShell (which uses backtick for line continuation, not backslash)::

    python bbt/box_block_test.py --model mediapipe
    python bbt/box_block_test.py --model detnet-baseline
    python bbt/box_block_test.py --model detnet-pruned --weights checkpoints_finetuned/l1_25pct/ft_l1_25pct_rhdbest.pth --arch pruned_architectures/detnet_l1_25pct_ep71_noft.pth
    python bbt/box_block_test.py --model detnet-quantized --quant-tag qmse

Common flags::

    --duration 60          seconds of active play (default 60)
    --infer-every 1        run detect() every Nth frame; >1 holds last result
                           between calls (mitigation for slow DetNet variants)
    --no-record            skip video + CSV
    --windowed             non-fullscreen window
    --pinch-threshold 40   pixels (plan §1.2)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2  # noqa: F401  -- Windows DLL load order
import numpy as np

# sys.path so direct `python bbt/box_block_test.py` and `python -m bbt.box_block_test`
# both find the package + webcam_detnet.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for p in (_REPO_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from bbt.hand_tracking import (
    MODEL_CHOICES, PinchState, make_tracker, pinch_distance,
)
from webcam_detnet import _draw_skeleton_at


# ── Tunables ─────────────────────────────────────────────────────────────────
# All pixel quantities are fractions of frame width / height, resolved once at
# startup so the layout adapts automatically to any camera resolution.
DURATION_SEC = 60.0
COUNTDOWN_SEC = 3.0
END_SCREEN_SEC = 5.0

PINCH_THRESHOLD_PX = 40.0          # engage threshold
PINCH_RELEASE_THRESHOLD_PX = 40.0  # release threshold (== engage: no hysteresis)
PINCH_DEBOUNCE_FRAMES = 2

PARTITION_X_FRAC = 0.50
PARTITION_WIDTH_FRAC = 0.012
BLOCK_SIZE_FRAC = 0.08
N_BLOCKS = 8
WALL_BLOCKS_TALL = 4  # partition acts as a physical barrier this many blocks tall
HUD_PADDING_FRAC = 0.02

GRAVITY_FRAC = 0.0018
MAX_VEL_FRAC = 0.05
BLOCK_FRICTION = 0.85

COLOUR_PARTITION = (200, 200, 200)
COLOUR_BLOCK_PALETTE = [
    (40, 80, 220), (40, 200, 220), (40, 200, 40), (220, 80, 40),
    (220, 40, 200), (40, 200, 200), (200, 100, 40), (40, 100, 220),
]
COLOUR_BLOCK_HELD = (50, 255, 255)
COLOUR_HUD_TEXT = (255, 255, 255)
COLOUR_HUD_ACCENT = (0, 255, 255)
COLOUR_HUD_TIME = (255, 255, 0)

RECORDINGS_DIR = os.path.join(_REPO_ROOT, 'BBT recordings')
CSV_FLUSH_EVERY_N = 30
VIDEO_FPS_NOMINAL = 30  # playback rate; CSV t_seconds is the ground truth for timing


# ── Blocks + physics ────────────────────────────────────────────────────────
@dataclass
class Block:
    id: int
    x: float
    y: float
    size: float
    color: tuple
    origin_x: float = 0.0
    origin_y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    held: bool = False
    scored: bool = False

    @property
    def cx(self) -> float: return self.x + self.size * 0.5

    @property
    def cy(self) -> float: return self.y + self.size * 0.5

    @property
    def x2(self) -> float: return self.x + self.size

    @property
    def y2(self) -> float: return self.y + self.size


def _resolve_pair(a: Block, b: Block) -> None:
    if a.held or b.held:
        return
    overlap_x = min(a.x2, b.x2) - max(a.x, b.x)
    overlap_y = min(a.y2, b.y2) - max(a.y, b.y)
    if overlap_x <= 0 or overlap_y <= 0:
        return
    if overlap_x < overlap_y:
        if a.cx < b.cx:
            a.x -= overlap_x * 0.5; b.x += overlap_x * 0.5
        else:
            a.x += overlap_x * 0.5; b.x -= overlap_x * 0.5
        a.vx = b.vx = 0.0
    else:
        if a.cy < b.cy:
            a.y -= overlap_y * 0.5; b.y += overlap_y * 0.5
            a.vy = min(a.vy, 0.0); b.vy = max(b.vy, 0.0)
        else:
            a.y += overlap_y * 0.5; b.y -= overlap_y * 0.5
            a.vy = max(a.vy, 0.0); b.vy = min(b.vy, 0.0)


def step_physics(blocks: List[Block], *, frame_w: int, frame_h: int,
                 gravity: float, max_vel: float, friction: float,
                 partition_x: float, partition_w: float,
                 wall_top: float) -> None:
    floor_y = frame_h - 1
    left_wall, right_wall = 0, frame_w - 1
    part_x1 = partition_x - partition_w * 0.5
    part_x2 = partition_x + partition_w * 0.5

    for b in blocks:
        if b.held:
            continue
        b.vy = min(b.vy + gravity, max_vel)
        b.x += b.vx
        b.y += b.vy
        if b.y2 > floor_y:
            b.y = floor_y - b.size; b.vy = 0.0; b.vx *= friction
        if b.y < 0:
            b.y = 0; b.vy = 0.0
        if b.x < left_wall:
            b.x = left_wall; b.vx = 0.0
        if b.x2 > right_wall:
            b.x = right_wall - b.size; b.vx = 0.0
        # Partition is a physical wall of finite height: only blocks whose
        # bottom edge sits below the wall's top edge are stopped. Lift higher
        # than wall_top to carry across.
        if b.y2 > wall_top and b.x2 > part_x1 and b.x < part_x2:
            if b.cx < partition_x:
                b.x = part_x1 - b.size
            else:
                b.x = part_x2
            b.vx = 0.0

    # Held block is positioned by the fingertip in the main loop just before
    # step_physics; apply the same wall constraint here so dragging at low
    # altitude is blocked too.
    for b in blocks:
        if not b.held:
            continue
        if b.y2 > wall_top and b.x2 > part_x1 and b.x < part_x2:
            if b.cx < partition_x:
                b.x = part_x1 - b.size
            else:
                b.x = part_x2

    # Two passes give noticeably more stable stacks than one on 8 blocks.
    n = len(blocks)
    for _ in range(2):
        for i in range(n):
            for j in range(i + 1, n):
                _resolve_pair(blocks[i], blocks[j])


# ── Scene layout + drawing ──────────────────────────────────────────────────
@dataclass
class Layout:
    frame_w: int
    frame_h: int
    partition_x: float
    partition_w: float
    block_size: float
    gravity: float
    max_vel: float
    wall_top: float  # y-coordinate of the top of the physical wall


def init_layout(frame_w: int, frame_h: int) -> Layout:
    block_size = BLOCK_SIZE_FRAC * frame_w
    return Layout(
        frame_w=frame_w,
        frame_h=frame_h,
        partition_x=PARTITION_X_FRAC * frame_w,
        partition_w=max(2.0, PARTITION_WIDTH_FRAC * frame_w),
        block_size=block_size,
        gravity=GRAVITY_FRAC * frame_h,
        max_vel=MAX_VEL_FRAC * frame_h,
        wall_top=(frame_h - 1) - WALL_BLOCKS_TALL * block_size,
    )


def spawn_blocks(layout: Layout) -> List[Block]:
    src_left = layout.partition_x * 0.05
    src_right = layout.partition_x - layout.partition_w * 0.5 - 1.0
    usable = max(layout.block_size, src_right - src_left)
    cols = max(1, int(usable // layout.block_size))
    blocks: List[Block] = []
    for i in range(N_BLOCKS):
        col, row = i % cols, i // cols
        x = src_left + col * layout.block_size
        y = (layout.frame_h - 1) - (row + 1) * layout.block_size + 1
        color = COLOUR_BLOCK_PALETTE[i % len(COLOUR_BLOCK_PALETTE)]
        blocks.append(Block(id=i, x=x, y=y, size=layout.block_size, color=color,
                            origin_x=x, origin_y=y))
    return blocks


def find_block_under_point(blocks: List[Block], x: float, y: float) -> Optional[Block]:
    hit = None
    for b in blocks:
        if not b.held and b.x <= x <= b.x2 and b.y <= y <= b.y2:
            hit = b  # last drawn wins (topmost)
    return hit


def draw_partition(img, layout: Layout) -> None:
    x1 = int(layout.partition_x - layout.partition_w * 0.5)
    x2 = int(layout.partition_x + layout.partition_w * 0.5)
    y_top = int(layout.wall_top)
    y_bot = layout.frame_h - 1
    cv2.rectangle(img, (x1, y_top), (x2, y_bot), COLOUR_PARTITION, -1)
    cv2.rectangle(img, (x1, y_top), (x2, y_bot), (255, 255, 255), 1)


def draw_blocks(img, blocks: List[Block], held_block_id: int) -> None:
    for b in blocks:
        x1, y1, x2, y2 = int(b.x), int(b.y), int(b.x2), int(b.y2)
        if b.id == held_block_id:
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOUR_BLOCK_HELD, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 3)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), b.color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)


def draw_skeleton_overlay(img, landmarks_px: np.ndarray, grabbing: bool) -> None:
    _draw_skeleton_at(img, landmarks_px)
    idx = landmarks_px[8]
    p = (int(idx[0]), int(idx[1]))
    if grabbing:
        cv2.circle(img, p, 13, (50, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(img, p, 5, (50, 255, 255), -1, cv2.LINE_AA)
    else:
        cv2.circle(img, p, 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, 7, (0, 0, 0), 1, cv2.LINE_AA)


# ── HUD ─────────────────────────────────────────────────────────────────────
def _shadow_text(img, text, org, scale, color, thick=2):
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hud(img, *, score: int, seconds_left: float, model_name: str,
             fps: float, inference_ms: float) -> None:
    h, w = img.shape[:2]
    pad = max(8, int(HUD_PADDING_FRAC * w))
    panel_w = min(340, w // 2)
    panel_h = 150
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)
    y = pad + 22
    _shadow_text(img, "Box & Block", (pad, y), 0.85, COLOUR_HUD_TEXT, 2);            y += 30
    _shadow_text(img, f"Score: {score}", (pad, y), 0.7, COLOUR_HUD_ACCENT, 2);       y += 26
    _shadow_text(img, f"Time:  {max(0.0, seconds_left):5.1f}s",
                 (pad, y), 0.7, COLOUR_HUD_TIME, 2);                                  y += 22
    _shadow_text(img, model_name, (pad, y), 0.5, (200, 200, 200), 1);                y += 18
    _shadow_text(img, f"FPS {fps:4.1f}   inf {inference_ms:5.1f} ms",
                 (pad, y), 0.5, (200, 200, 200), 1)


def draw_countdown(img, sec_remaining: float) -> None:
    h, w = img.shape[:2]
    text = str(int(sec_remaining) + 1) if sec_remaining > 0.05 else "GO!"
    scale = max(2.0, w / 200.0)
    thick = max(4, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cx, cy = w // 2 - tw // 2, h // 2 + th // 2
    cv2.putText(img, text, (cx + 4, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 255, 255), thick, cv2.LINE_AA)


def draw_endscreen(img, final_score: int) -> None:
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)
    scale = max(1.2, w / 480.0)
    thick = max(3, int(scale * 2))
    for text, dy, color in (("Time's up!", -20, (255, 255, 255)),
                            (f"Final Score: {final_score}", 50, (0, 255, 255))):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(img, text, (w // 2 - tw // 2, h // 2 + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ── Recorder + CSV (inline classes) ─────────────────────────────────────────
class Recorder:
    """Opens a cv2.VideoWriter immediately; on close, renames the file to embed
    the final score. If the process is killed mid-session, the temp file is
    left behind under an INPROGRESS name so the partial recording is recoverable."""
    def __init__(self, out_dir: str, model_label: str,
                 frame_w: int, frame_h: int, fps: int):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.model_label = model_label
        self.timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        self._tmp = os.path.join(out_dir, f"{self.timestamp}_{model_label}_INPROGRESS.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._w = cv2.VideoWriter(self._tmp, fourcc, fps, (frame_w, frame_h))
        if not self._w.isOpened():
            raise RuntimeError(f"VideoWriter failed at {self._tmp}")

    def write(self, frame_bgr) -> None:
        self._w.write(frame_bgr)

    def close(self, final_score: int) -> str:
        self._w.release(); self._w = None
        final = os.path.join(self.out_dir,
                             f"{self.timestamp}_{self.model_label}_score{final_score}.mp4")
        try:
            if os.path.exists(final):
                os.remove(final)
            os.rename(self._tmp, final)
            return final
        except OSError:
            return self._tmp


class FrameLogger:
    """Per-frame CSV. landmarks_px is a single JSON-encoded column of all 21
    (x,y) pixel coords, per plan §5. Flushes every CSV_FLUSH_EVERY_N rows so a
    crash mid-session still leaves usable data on disk."""
    FIELDS = (
        'frame', 't_seconds', 'model', 'hand_detected',
        'index_tip_x', 'index_tip_y', 'landmarks_px',
        'pinch_distance', 'grabbing', 'held_block_id',
        'score', 'fps', 'inference_ms',
    )

    def __init__(self, out_dir: str, model_label: str, timestamp: str,
                 flush_every: int = CSV_FLUSH_EVERY_N):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.model_label = model_label
        self.timestamp = timestamp
        self.flush_every = flush_every
        self._tmp = os.path.join(out_dir, f"{timestamp}_{model_label}_INPROGRESS.csv")
        self._f = open(self._tmp, 'w', newline='', encoding='utf-8')
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()
        self._n = 0

    def log(self, *, frame_idx: int, t_seconds: float, hand_detected: bool,
            landmarks_px: Optional[np.ndarray], pinch_distance: float,
            grabbing: bool, held_block_id: int, score: int, fps: float,
            inference_ms: float) -> None:
        row = {
            'frame': frame_idx,
            't_seconds': f'{t_seconds:.4f}',
            'model': self.model_label,
            'hand_detected': 1 if hand_detected else 0,
            'grabbing': 1 if grabbing else 0,
            'held_block_id': held_block_id,
            'score': score,
            'fps': f'{fps:.2f}',
            'inference_ms': f'{inference_ms:.2f}',
        }
        if hand_detected and landmarks_px is not None:
            row['index_tip_x'] = int(landmarks_px[8, 0])
            row['index_tip_y'] = int(landmarks_px[8, 1])
            row['landmarks_px'] = json.dumps(landmarks_px.tolist())
            row['pinch_distance'] = f'{pinch_distance:.2f}'
        else:
            row['index_tip_x'] = ''
            row['index_tip_y'] = ''
            row['landmarks_px'] = ''
            row['pinch_distance'] = ''
        self._w.writerow(row)
        self._n += 1
        if self._n % self.flush_every == 0:
            self._f.flush()

    def close(self, final_score: int) -> str:
        if self._f is not None:
            self._f.flush(); self._f.close(); self._f = None
        final = os.path.join(self.out_dir,
                             f"{self.timestamp}_{self.model_label}_score{final_score}.csv")
        try:
            if os.path.exists(final):
                os.remove(final)
            os.rename(self._tmp, final)
            return final
        except OSError:
            return self._tmp


# ── Main loop ───────────────────────────────────────────────────────────────
def compose_model_label(args) -> str:
    """Descriptive model label used for CSV/MP4 filenames and the CSV 'model' column.

    Examples:
        detnet-pruned + arch=...detnet_l1_25pct_ep71_noft.pth  -> detnet-pruned_L1-25pct
        detnet-pruned + arch=...detnet_taylor_10pct_ep71_noft.pth -> detnet-pruned_Taylor-10pct
        detnet-quantized + --quant-tag qmm  -> detnet-quantized_QMM
        detnet-quantized + --quant-tag qmse -> detnet-quantized_QMSE
        detnet-baseline / mediapipe -> unchanged
    """
    if args.model == 'detnet-pruned':
        for src in (args.arch, args.weights):
            if not src:
                continue
            m = re.search(r'(l1|taylor)[_-](\d+)\s*pct', src, re.IGNORECASE)
            if m:
                method = 'L1' if m.group(1).lower() == 'l1' else 'Taylor'
                return f'detnet-pruned_{method}-{int(m.group(2))}pct'
        return 'detnet-pruned_unknown'
    if args.model == 'detnet-quantized':
        return f'detnet-quantized_{args.quant_tag.upper()}'
    return args.model


def parse_args():
    p = argparse.ArgumentParser(description='Box and Block Test (BBT) — hand-pose game.')
    p.add_argument('--model', required=True, choices=MODEL_CHOICES)
    p.add_argument('--weights', type=str, default=None)
    p.add_argument('--arch', type=str, default=None,
                   help='pruned architecture shell (_noft.pth); required for detnet-pruned')
    p.add_argument('--quant-tag', type=str, default='qmse', choices=['qmm', 'qmse'])
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--duration', type=float, default=DURATION_SEC)
    p.add_argument('--pinch-threshold', type=float, default=PINCH_THRESHOLD_PX,
                   help=f'engage threshold in px (default {PINCH_THRESHOLD_PX:.0f})')
    p.add_argument('--pinch-release-threshold', type=float,
                   default=PINCH_RELEASE_THRESHOLD_PX,
                   help=f'release threshold in px '
                        f'(default {PINCH_RELEASE_THRESHOLD_PX:.0f})')
    p.add_argument('--pinch-debounce', type=int, default=PINCH_DEBOUNCE_FRAMES)
    p.add_argument('--infer-every', type=int, default=1,
                   help='run detect() every Nth frame; >1 holds last result between calls')
    p.add_argument('--no-record', action='store_true')
    p.add_argument('--windowed', dest='fullscreen', action='store_false')
    p.add_argument('--window-scale', type=float, default=2.0,
                   help='windowed-mode display scale relative to camera frame '
                        '(default 2.0; e.g. 640x480 camera -> 1280x960 window)')
    p.set_defaults(fullscreen=True)
    return p.parse_args()


def open_camera(index: int):
    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 60)
    return cap


PHASE_COUNTDOWN, PHASE_PLAY, PHASE_END = 'countdown', 'play', 'end'


def main():
    args = parse_args()
    if args.infer_every < 1:
        raise SystemExit('--infer-every must be >= 1')

    print(f"Building tracker: {args.model}")
    tracker = make_tracker(args.model, weights=args.weights, arch=args.arch,
                           quant_tag=args.quant_tag)
    print(f"Tracker ready: {tracker.name}")

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit(f'cannot open camera {args.camera}')
    ok, raw = cap.read()
    if not ok:
        raise SystemExit('cannot read initial frame from camera')
    frame_h, frame_w = raw.shape[:2]
    print(f"Camera: {frame_w}x{frame_h}")

    layout = init_layout(frame_w, frame_h)
    blocks = spawn_blocks(layout)

    recorder, logger = None, None
    if not args.no_record:
        model_label = compose_model_label(args)
        recorder = Recorder(RECORDINGS_DIR, model_label, frame_w, frame_h,
                            fps=VIDEO_FPS_NOMINAL)
        logger = FrameLogger(RECORDINGS_DIR, model_label, recorder.timestamp)

    win_name = 'BBT'
    if args.fullscreen:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name,
                         int(frame_w * args.window_scale),
                         int(frame_h * args.window_scale))

    pinch = PinchState(threshold=args.pinch_threshold,
                       debounce=args.pinch_debounce,
                       release_threshold=args.pinch_release_threshold)

    score = 0
    held_block_id = -1
    inference_ms = 0.0
    frame_idx = 0
    fps_smoothed = 0.0
    last_result = None  # cached for --infer-every N > 1

    session_start_t = time.time()
    now = session_start_t
    phase = PHASE_COUNTDOWN
    phase_started = now
    play_started = None
    prev_loop_t = now

    print(f"Starting — {COUNTDOWN_SEC:g}s countdown, then {args.duration:g}s play. "
          "Press 'q' or ESC to quit early.")
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            frame_idx += 1
            # DetNet (any variant) was trained on un-mirrored hand crops;
            # feeding it the flipped frame puts the thumb on the wrong side of
            # the crop and inverts its joint-index predictions
            # (index <-> pinky, middle <-> ring). Detect on the raw frame, then
            # mirror landmarks below so they align with the mirrored display.
            # MediaPipe is handedness-aware so this is correct for it too.
            if (frame_idx - 1) % args.infer_every == 0:
                last_result = tracker.detect(raw)
                if last_result is not None:
                    inference_ms = last_result.inference_ms
                    last_result.landmarks_px[:, 0] = (
                        (raw.shape[1] - 1) - last_result.landmarks_px[:, 0]
                    )
            result = last_result
            frame_bgr = cv2.flip(raw, 1)  # mirrored display; landmarks already mirrored above

            hand_detected = result is not None
            if hand_detected:
                landmarks = result.landmarks_px
                idx_tip = landmarks[8]
                d = pinch_distance(landmarks)
            else:
                landmarks = None
                d = float('nan')

            display = frame_bgr.copy()
            now = time.time()
            phase_elapsed = now - phase_started

            if phase == PHASE_COUNTDOWN:
                draw_partition(display, layout)
                draw_blocks(display, blocks, held_block_id)
                if hand_detected:
                    draw_skeleton_overlay(display, landmarks, grabbing=False)
                draw_countdown(display, COUNTDOWN_SEC - phase_elapsed)
                if phase_elapsed >= COUNTDOWN_SEC:
                    phase = PHASE_PLAY
                    phase_started = now
                    play_started = now

            elif phase == PHASE_PLAY:
                if hand_detected:
                    grabbing, engaged, released = pinch.update(d)
                else:
                    # Lost hand → drop anything held, force release
                    if pinch.force_release() and held_block_id >= 0:
                        blocks[held_block_id].held = False
                        held_block_id = -1
                    grabbing, engaged, released = False, False, False

                if engaged:
                    b = find_block_under_point(blocks,
                                               float(idx_tip[0]), float(idx_tip[1]))
                    if b is not None:
                        b.held = True
                        held_block_id = b.id
                elif released:
                    if held_block_id >= 0:
                        b = blocks[held_block_id]
                        b.held = False
                        if not b.scored and b.cx > layout.partition_x:
                            score += 1
                            b.scored = True
                            # Continuous replacement: a new block spawns at the
                            # placed block's original source-side position.
                            new_id = len(blocks)
                            new_color = COLOUR_BLOCK_PALETTE[new_id % len(COLOUR_BLOCK_PALETTE)]
                            blocks.append(Block(
                                id=new_id, x=b.origin_x, y=b.origin_y,
                                size=b.size, color=new_color,
                                origin_x=b.origin_x, origin_y=b.origin_y,
                            ))
                        held_block_id = -1

                # Held block follows the index fingertip; re-arm scoring if
                # the player carries a previously-scored block back to source.
                if held_block_id >= 0 and hand_detected:
                    b = blocks[held_block_id]
                    fingertip_x = float(idx_tip[0])
                    fingertip_y = float(idx_tip[1])
                    # If the player drags the fingertip past the partition
                    # while the held block is still below the wall's top
                    # edge, they're trying to push it through the wall.
                    # Drop the block on its current side instead of letting
                    # the constraint solver glitch it across.
                    block_blocked_by_wall = (b.y2 > layout.wall_top)
                    block_on_left = (b.cx < layout.partition_x)
                    finger_on_left = (fingertip_x < layout.partition_x)
                    if block_blocked_by_wall and (block_on_left != finger_on_left):
                        b.held = False
                        held_block_id = -1
                        pinch.force_release()
                    else:
                        b.x = fingertip_x - b.size * 0.5
                        b.y = fingertip_y - b.size * 0.5
                        b.vx = b.vy = 0.0
                        if b.scored and b.cx < layout.partition_x:
                            b.scored = False
                        # Keep fully on-screen even while held
                        b.x = max(0.0, min(b.x, frame_w - b.size))
                        b.y = max(0.0, min(b.y, frame_h - b.size))

                step_physics(blocks, frame_w=frame_w, frame_h=frame_h,
                             gravity=layout.gravity, max_vel=layout.max_vel,
                             friction=BLOCK_FRICTION,
                             partition_x=layout.partition_x,
                             partition_w=layout.partition_w,
                             wall_top=layout.wall_top)

                draw_partition(display, layout)
                draw_blocks(display, blocks, held_block_id)
                if hand_detected:
                    draw_skeleton_overlay(display, landmarks, grabbing=grabbing)

                play_elapsed = now - play_started
                seconds_left = args.duration - play_elapsed
                draw_hud(display, score=score, seconds_left=seconds_left,
                         model_name=tracker.name, fps=fps_smoothed,
                         inference_ms=inference_ms)
                if seconds_left <= 0:
                    phase = PHASE_END
                    phase_started = now

            else:  # PHASE_END
                draw_partition(display, layout)
                draw_blocks(display, blocks, held_block_id)
                if hand_detected:
                    draw_skeleton_overlay(display, landmarks, grabbing=False)
                draw_hud(display, score=score, seconds_left=0.0,
                         model_name=tracker.name, fps=fps_smoothed,
                         inference_ms=inference_ms)
                draw_endscreen(display, score)
                if phase_elapsed >= END_SCREEN_SEC:
                    break

            if recorder is not None:
                recorder.write(display)
            if logger is not None:
                t_play = (now - play_started) if play_started else 0.0
                logger.log(
                    frame_idx=frame_idx,
                    t_seconds=t_play,
                    hand_detected=hand_detected,
                    landmarks_px=landmarks,
                    pinch_distance=(d if hand_detected else 0.0),
                    grabbing=pinch.grabbing,
                    held_block_id=held_block_id,
                    score=score,
                    fps=fps_smoothed,
                    inference_ms=inference_ms,
                )

            dt = now - prev_loop_t
            prev_loop_t = now
            if dt > 0:
                inst = 1.0 / dt
                fps_smoothed = inst if fps_smoothed == 0 else 0.9 * fps_smoothed + 0.1 * inst

            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        video_path = recorder.close(score) if recorder is not None else None
        csv_path = logger.close(score) if logger is not None else None
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print()
        print("=" * 60)
        print(f"  Box & Block Test — {tracker.name}")
        print("=" * 60)
        print(f"  Final score : {score}")
        print(f"  Frames      : {frame_idx}")
        session_elapsed = time.time() - session_start_t
        session_fps = (frame_idx / session_elapsed) if session_elapsed > 0 else 0.0
        print(f"  Mean FPS    : {session_fps:5.2f}  (over {session_elapsed:.1f}s)")
        if video_path:
            print(f"  Video       : {video_path}")
        if csv_path:
            print(f"  CSV         : {csv_path}")
        print("=" * 60)


if __name__ == '__main__':
    main()
