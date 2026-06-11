"""Hand -> OS mouse cursor via PyAutoGUI. Pinch = click-and-hold.

Same model dispatch as bbt/box_block_test.py — both scripts go through
bbt/hand_tracking.py::make_tracker, so the model loaders and pinch logic exist
once. Index fingertip is the cursor control point.

Usage (commands are single-line so they paste straight into PowerShell,
which uses backtick for line continuation, not backslash):
    python bbt/cursor_control.py --model mediapipe
    python bbt/cursor_control.py --model detnet-baseline
    python bbt/cursor_control.py --model detnet-pruned --weights checkpoints_finetuned/l1_25pct/ft_l1_25pct_rhdbest.pth --arch pruned_architectures/detnet_l1_25pct_ep71_noft.pth
    python bbt/cursor_control.py --model detnet-quantized --quant-tag qmse

Keys:
    q / ESC   quit
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys

import cv2  # noqa: F401  -- Windows DLL load order
import numpy as np

# Repo root on sys.path so both `python bbt/cursor_control.py` and
# `python -m bbt.cursor_control` find the package and webcam_detnet alike.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for p in (_REPO_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from bbt.hand_tracking import (
    MODEL_CHOICES, PinchState, make_tracker, pinch_distance,
)
from webcam_detnet import _draw_skeleton_at  # reuse skeleton draw


# ── Stability tunables (cursor-control only — BBT game keeps its own values) ─
# Tolerate brief MediaPipe dropouts: don't release the click on a single missed
# frame, only after the hand has been gone for this many consecutive frames.
# Fixes the "many short strokes" pattern seen during fast drag motion.
HAND_LOSS_TOLERANCE_FRAMES = 10

# Cursor position EMA: smoothed = α * previous + (1-α) * current_raw.
# Higher α = smoother but laggier. 0.6 = ~2-3 frame settling time on a step
# input; suppresses jitter without noticeable lag at MediaPipe FPS.
CURSOR_EMA_ALPHA = 0.6

# Pinch-distance EMA: same shape, applied to the thumb-index distance BEFORE
# threshold check. Stops transient landmark noise spikes (~10-20 px during
# fast drag motion) from popping above the release threshold even when the
# user is still pinched. Combined with release_debounce (below), this gives
# true click-and-hold.
PINCH_DIST_EMA_ALPHA = 0.5

# Asymmetric debounce on the pinch state: engage fast, release slowly.
# release-debounce defaults to 10 frames (~333 ms at 30 FPS); release requires
# sustained finger separation, ignoring noise / motion-blur spikes.
PINCH_RELEASE_DEBOUNCE_FRAMES = 10

# Standard hysteresis: engage when smoothed d < 20 px (tight pinch starts
# the click), release when smoothed d > 30 px (sustained finger separation
# releases). The 20-30 px band is a sticky dead-zone: once engaged, you
# stay engaged inside it; once released, you stay released inside it. This
# prevents chattering when the user's pinch hovers near the threshold.
PINCH_RELEASE_THRESHOLD_PX = 30.0

# Cursor-move interpolation. PyAutoGUI's moveTo to (x, y) is INSTANT — the
# cursor teleports there in one OS event. With a click held, Paint draws a
# straight line between consecutive frame positions, which looks awful when
# the hand path between frames is curved. Splitting each frame's move into
# several small intermediate moveTo calls makes Paint see a connected drag
# path rather than discrete teleports.
MOUSE_INTERP_MAX_STEPS = 12
MOUSE_INTERP_PIXELS_PER_STEP = 25  # one intermediate position per ~N screen px


def parse_args():
    p = argparse.ArgumentParser(description='Hand -> OS mouse via PyAutoGUI; pinch = click-and-hold.')
    p.add_argument('--model', required=True, choices=MODEL_CHOICES)
    p.add_argument('--weights', type=str, default=None,
                   help='state_dict .pth (baseline / pruned). Defaults to repo baseline ckp.')
    p.add_argument('--arch', type=str, default=None,
                   help='pruned architecture shell (_noft.pth); required for detnet-pruned.')
    p.add_argument('--quant-tag', type=str, default='qmse', choices=['qmm', 'qmse'])
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--pinch-threshold', type=float, default=20.0,
                   help='engage threshold in pixels (default 20; click engages '
                        'when smoothed pinch distance falls below this)')
    p.add_argument('--pinch-debounce', type=int, default=3,
                   help='engage debounce: frames distance < threshold must hold '
                        'to engage the click (default 3)')
    p.add_argument('--pinch-release-debounce', type=int,
                   default=PINCH_RELEASE_DEBOUNCE_FRAMES,
                   help=f'release debounce: frames distance > threshold must '
                        f'hold to release the click '
                        f'(default {PINCH_RELEASE_DEBOUNCE_FRAMES}; longer than '
                        f'engage = sticky click-and-hold)')
    p.add_argument('--pinch-release-threshold', type=float,
                   default=PINCH_RELEASE_THRESHOLD_PX,
                   help=f'release threshold in pixels: distance > this releases '
                        f'the click after release-debounce '
                        f'(default {PINCH_RELEASE_THRESHOLD_PX:.0f}; higher than '
                        f'pinch-threshold = sticky dead-zone between the two)')
    p.add_argument('--no-mouse', action='store_true',
                   help='preview only — do not move the OS mouse or click')
    p.add_argument('--window-scale', type=float, default=2.0,
                   help='display scale relative to camera frame '
                        '(default 2.0; e.g. 640x480 camera -> 1280x960 window)')
    return p.parse_args()


def open_camera(index: int):
    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 60)
    return cap


def _smooth_drag_move(pyautogui_mod, last_pos, target_pos):
    """Move cursor from last_pos to target_pos via N intermediate moveTo calls.

    PyAutoGUI's moveTo with duration=0 fires one absolute-position OS event;
    paint apps then draw a straight line between consecutive frame positions.
    Splitting the move into several short hops generates one WM_MOUSEMOVE per
    hop, which Paint renders as a chain of short segments approximating the
    actual hand path. The step count is bounded by both the screen distance
    (one hop per ~MOUSE_INTERP_PIXELS_PER_STEP px) and the hard cap, so a
    near-stationary frame still costs ~1 move and a long jump costs at most
    MOUSE_INTERP_MAX_STEPS. last_pos=None on the first frame just teleports.
    """
    tx, ty = target_pos
    if last_pos is None:
        pyautogui_mod.moveTo(tx, ty, _pause=False)
        return
    lx, ly = last_pos
    dx, dy = tx - lx, ty - ly
    distance = (dx * dx + dy * dy) ** 0.5
    steps = min(MOUSE_INTERP_MAX_STEPS,
                max(1, int(distance / MOUSE_INTERP_PIXELS_PER_STEP)))
    for i in range(1, steps + 1):
        t = i / steps
        pyautogui_mod.moveTo(int(lx + t * dx), int(ly + t * dy), _pause=False)


def main():
    args = parse_args()

    pyautogui = None
    if not args.no_mouse:
        try:
            import pyautogui as _pyautogui
            pyautogui = _pyautogui
            # Disable PyAutoGUI's corner-failsafe so the cursor reaching (0,0)
            # doesn't terminate the script mid-session; rely on q/ESC instead.
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.0
        except ImportError as e:
            raise SystemExit(
                "pyautogui not installed. `pip install pyautogui` or use --no-mouse."
            ) from e

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

    screen_w, screen_h = (None, None)
    if pyautogui is not None:
        screen_w, screen_h = pyautogui.size()
        print(f"Screen: {screen_w}x{screen_h}")

    pinch = PinchState(threshold=args.pinch_threshold,
                       debounce=args.pinch_debounce,
                       release_debounce=args.pinch_release_debounce,
                       release_threshold=args.pinch_release_threshold)

    win_name = 'Cursor Control'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name,
                     int(frame_w * args.window_scale),
                     int(frame_h * args.window_scale))

    missed_frames = 0
    smoothed_idx = None  # EMA state for the index-fingertip pixel position
    smoothed_d = None    # EMA state for the thumb-index pinch distance
    last_screen_pos = None  # previous frame's screen-space cursor, for drag interpolation

    print(f"Running — q / ESC to quit.  "
          f"(engage at d<{args.pinch_threshold:.0f}px, "
          f"release at d>{args.pinch_release_threshold:.0f}px, "
          f"engage debounce {args.pinch_debounce} frames, "
          f"release debounce {args.pinch_release_debounce} frames, "
          f"hand-loss tolerance {HAND_LOSS_TOLERANCE_FRAMES} frames, "
          f"cursor EMA α={CURSOR_EMA_ALPHA}, "
          f"distance EMA α={PINCH_DIST_EMA_ALPHA})")
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            # DetNet (any variant) was trained on un-mirrored hand crops;
            # feeding it the flipped frame puts the thumb on the wrong side of
            # the crop and inverts its joint-index predictions (index ↔ pinky,
            # middle ↔ ring). Detect on the raw frame; MediaPipe is
            # handedness-aware so this is correct for it too. Mirror the
            # landmarks below so they align with the mirrored display.
            result = tracker.detect(raw)
            frame_bgr = cv2.flip(raw, 1)
            if result is not None:
                result.landmarks_px[:, 0] = (raw.shape[1] - 1) - result.landmarks_px[:, 0]

            display = frame_bgr.copy()

            if result is not None:
                missed_frames = 0
                lms = result.landmarks_px
                raw_idx = lms[8].astype(np.float32)
                if smoothed_idx is None:
                    smoothed_idx = raw_idx.copy()
                else:
                    smoothed_idx = (CURSOR_EMA_ALPHA * smoothed_idx
                                    + (1.0 - CURSOR_EMA_ALPHA) * raw_idx)
                raw_d = pinch_distance(lms)
                if smoothed_d is None:
                    smoothed_d = raw_d
                else:
                    smoothed_d = (PINCH_DIST_EMA_ALPHA * smoothed_d
                                  + (1.0 - PINCH_DIST_EMA_ALPHA) * raw_d)
                d = smoothed_d  # threshold + HUD both use the smoothed value
                grabbing, engaged, released = pinch.update(d)

                if pyautogui is not None:
                    sx = int(smoothed_idx[0] * (screen_w - 1) / max(1, frame_w - 1))
                    sy = int(smoothed_idx[1] * (screen_h - 1) / max(1, frame_h - 1))
                    _smooth_drag_move(pyautogui, last_screen_pos, (sx, sy))
                    last_screen_pos = (sx, sy)
                    if engaged:
                        pyautogui.mouseDown(_pause=False)
                    elif released:
                        pyautogui.mouseUp(_pause=False)

                _draw_skeleton_at(display, lms)
                p = (int(smoothed_idx[0]), int(smoothed_idx[1]))
                if grabbing:
                    cv2.circle(display, p, 14, (50, 255, 255), 3, cv2.LINE_AA)
                    cv2.circle(display, p, 5, (50, 255, 255), -1, cv2.LINE_AA)
                else:
                    cv2.circle(display, p, 7, (0, 255, 255), -1, cv2.LINE_AA)

                cv2.putText(display, f'd={d:5.1f}px  {"GRAB" if grabbing else "----"}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255) if grabbing else (200, 200, 200), 2, cv2.LINE_AA)
            else:
                missed_frames += 1
                if missed_frames >= HAND_LOSS_TOLERANCE_FRAMES:
                    # Tolerated dropout window exceeded — release click + reset
                    # smoothers so the next detection initialises fresh.
                    if pinch.force_release() and pyautogui is not None:
                        pyautogui.mouseUp(_pause=False)
                    smoothed_idx = None
                    smoothed_d = None
                    last_screen_pos = None
                    label, colour = 'No hand detected', (0, 0, 255)
                else:
                    label = (f'Hand briefly lost ({missed_frames}/'
                             f'{HAND_LOSS_TOLERANCE_FRAMES}) — click held')
                    colour = (0, 165, 255)
                cv2.putText(display, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

            cv2.putText(display, f'{tracker.name}', (10, frame_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        # Make sure we don't exit while still holding a click.
        if pinch.grabbing and pyautogui is not None:
            try:
                pyautogui.mouseUp(_pause=False)
            except Exception:
                pass
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
