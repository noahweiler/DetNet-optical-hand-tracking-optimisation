"""
Real-time hand tracking from webcam — DetNet (any variant) or MediaPipe alone.

Four model paths, all sharing the same camera + drawing + FPS instrumentation
so you can compare end-to-end FPS apples-to-apples:

  - Unpruned baseline   : --checkpoint <ckp_detnet_*.pth>   --device {cpu,cuda}
  - Pruned + fine-tuned : --checkpoint <ft_*pct_*.pth>      --device {cpu,cuda}
                          --arch       <pruned_architectures/*_noft.pth>
  - Quantized (INT8)    : --quant {qmm,qmse}   (CPU forced; FBGEMM x86)
  - MediaPipe only      : --mediapipe          (skips DetNet entirely; draws
                                                MediaPipe's 21 landmarks
                                                directly. CPU tflite.)

Pipeline tweaks (always on) so the loop isn't bottlenecked by camera I/O:
  - DSHOW backend on Windows  (default MSMF adds ~30 ms / frame of buffering)
  - cap.set(BUFFERSIZE=1)     (drop stale frames -> cap.read() returns latest)
  - cap.set(FPS=60)           (the camera silently caps to what it supports)
  - --mp_every_n N            : run the MediaPipe detector every N frames and
                                reuse the cached bbox/landmarks between. DetNet
                                still runs every frame. Bigger N -> faster.

A session-wide mean FPS (every frame between launch and quit, incl. warmup) is
printed when you exit. Press 'q' to quit.

Usage:
    python webcam_detnet.py                                          # baseline CPU (default)
    python webcam_detnet.py --device cuda                            # baseline GPU
    python webcam_detnet.py --checkpoint checkpoints_finetuned/l1_25pct/ft_l1_25pct_rhdbest.pth ^
                            --arch       pruned_architectures/detnet_l1_25pct_ep71_noft.pth ^
                            --device cuda                            # pruned GPU
    python webcam_detnet.py --quant qmse                             # INT8 (CPU)
    python webcam_detnet.py --mediapipe                              # MediaPipe alone
    python webcam_detnet.py --mp_every_n 3                           # MP bbox every 3rd frame
    python webcam_detnet.py --device cuda --seconds 30               # 30 s timed benchmark run
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

from model.detnet import detnet
from utils import func, smoother
from config import SNAP_BONES, JOINT_COLORS

HEATMAP_SIZE = 32
INPUT_SIZE = 128


def load_model(checkpoint_path, device, arch_path=None):
    """Load a DetNet state_dict. Mirrors evaluate_detnet.load_model exactly:

    1. Unpruned (default)  - fresh detnet() + state_dict from checkpoint.
    2. Pruned + fine-tuned - pass arch_path pointing at the matching _noft.pth
                             so the pruned architecture is rebuilt before the
                             state_dict is loaded.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: '{checkpoint_path}'")

    if arch_path is not None:
        if not os.path.isfile(arch_path):
            raise FileNotFoundError(f"Architecture file not found: '{arch_path}'")
        arch_obj = torch.load(arch_path, map_location=device, weights_only=False)
        if not isinstance(arch_obj, torch.nn.Module):
            raise ValueError(
                f"--arch must point to a whole-model file (e.g. _noft.pth), "
                f"got a state_dict at '{arch_path}'."
            )
        model = arch_obj.to(device)
        arch_label = f"pruned arch from {arch_path}"
    else:
        model = detnet().to(device)
        arch_label = "unpruned DetNet"

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Strip 'module.' prefix added by DataParallel if present
    if next(iter(ckpt)).startswith('module.'):
        ckpt = {k[len('module.'):]: v for k, v in ckpt.items()}

    state   = model.state_dict()
    matched = {k: v for k, v in ckpt.items()
               if k in state and v.shape == state[k].shape}
    state.update(matched)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded {len(matched)}/{len(ckpt)} keys into {arch_label}.")
    return model


def load_quantized(tag):
    """Re-quantize the INT8 DetNet in-process via quant/qquant.py.

    INT8 whole-model pickles don't round-trip through torch.load, but the
    quantization pipeline is deterministic given the fixed fork weights +
    quant/calib_tensors.pt — so re-running qquant.quantize() reproduces the
    same model that 07_quant_to_evalresults.py / method[23].py saved. CPU
    only (FBGEMM x86 backend).
    """
    quant_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quant')
    if quant_dir not in sys.path:
        sys.path.insert(0, quant_dir)
    import qquant   # type: ignore[reportMissingImports]  (added to sys.path above)
    qconfig = {'qmm': qquant.M2_QCONFIG, 'qmse': qquant.M3_QCONFIG}[tag]
    print(f"Re-quantizing {tag.upper()} via qquant.quantize() "
          f"(deterministic, ~3 min: calibrate -> convert -> bias-correct)...")
    model, n_corr = qquant.quantize(qconfig)
    model.eval()
    print(f"  INT8 ready (bias-corrected layers: {n_corr})")
    return model


def get_hand_detection(frame_rgb, hands_detector, pad=0.3):
    """One MediaPipe call -> (square bbox, (21,2) pixel landmarks) or (None, None).

    Returns BOTH so DetNet mode can take the bbox and discard landmarks while
    --mediapipe mode can take the landmarks and skip cropping/DetNet entirely.
    """
    import mediapipe as mp
    h, w = frame_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = hands_detector.detect(mp_image)
    if not results.hand_landmarks:
        return None, None

    lms = results.hand_landmarks[0]
    pts_xy = np.array([[int(lm.x * w), int(lm.y * h)] for lm in lms],
                      dtype=np.int32)

    xs, ys = pts_xy[:, 0].astype(float), pts_xy[:, 1].astype(float)
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())

    # Expand by padding factor
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad
    y2 += bh * pad

    # Make square by expanding the shorter side
    bw, bh = x2 - x1, y2 - y1
    if bw > bh:
        diff = bw - bh
        y1 -= diff / 2
        y2 += diff / 2
    else:
        diff = bh - bw
        x1 -= diff / 2
        x2 += diff / 2

    # Clamp to frame bounds
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return (x1, y1, x2, y2), pts_xy


def preprocess(frame_bgr, device, crop=None):
    """
    Prepare a 128x128 input tensor for DetNet.
    If crop=(x1,y1,x2,y2) is given, use that region.
    Otherwise fall back to centre square crop of the full frame.
    """
    if crop is not None:
        x1, y1, x2, y2 = crop
        img = frame_bgr[y1:y2, x1:x2].copy()
    else:
        img = frame_bgr.copy()
        fh, fw = img.shape[:2]
        if fh > fw:
            m = (fh - fw) // 2
            img = img[m:m + fw]
        else:
            m = (fw - fh) // 2
            img = img[:, m:m + fh]

    img = np.flip(img, -1).copy()  # BGR -> RGB
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    # Divide by 255 to match training: func.to_tensor gives [0,1], then
    # normalize([0.5,0.5,0.5], [1,1,1]) shifts to [-0.5, 0.5]
    tensor = torch.tensor(img.transpose([2, 0, 1]), dtype=torch.float, device=device) / 255.0
    tensor = func.normalize(tensor, [0.5, 0.5, 0.5], [1, 1, 1])
    return tensor.unsqueeze(0)


def _draw_skeleton_at(img, pts):
    """Draw the SNAP 21-joint skeleton at (21, 2) integer pixel coords.

    MediaPipe Hands and SNAP share the same joint numbering (wrist=0, thumb 1-4,
    index 5-8, middle 9-12, ring 13-16, pinky 17-20), so the same SNAP_BONES /
    JOINT_COLORS work for both the DetNet uv output and MediaPipe landmarks.
    """
    for finger_idx, bone_chain in enumerate(SNAP_BONES):
        r, g, b = JOINT_COLORS[finger_idx]
        bgr = (b, g, r)
        for i in range(len(bone_chain) - 1):
            j1, j2 = bone_chain[i], bone_chain[i + 1]
            cv2.line(img, tuple(pts[j1]), tuple(pts[j2]), bgr, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), -1)
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def draw_skeleton(img, uv_joints, region_h, region_w, origin=(0, 0)):
    """DetNet heatmap (32x32) coords -> image pixels, then draw.

    uv_joints : (21, 2) array — [row, col] in [0, HEATMAP_SIZE) space
    region_h/w: pixel size of the region DetNet ran on
    origin    : (ox, oy) pixel offset of that region within img
    """
    ox, oy = origin
    pts = np.zeros((21, 2), dtype=np.int32)
    pts[:, 0] = (uv_joints[:, 1] * region_w / HEATMAP_SIZE).astype(np.int32) + ox
    pts[:, 1] = (uv_joints[:, 0] * region_h / HEATMAP_SIZE).astype(np.int32) + oy
    return _draw_skeleton_at(img, pts)


def open_camera(index):
    """Open the camera with the fastest backend + smallest buffer + high FPS.

    On Windows the default MSMF backend adds ~30 ms / frame of buffering
    latency; DSHOW is significantly snappier. BUFFERSIZE=1 drops stale frames
    so cap.read() returns the LATEST, not the back of a queue. cap.set(FPS=60)
    is a request — the camera silently caps to what it actually supports.
    """
    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 60)
    return cap


def main():
    parser = argparse.ArgumentParser(description='Real-time DetNet / MediaPipe hand tracking')
    parser.add_argument('--checkpoint', type=str,
                        default='new_check_point/ckp_detnet_71.pth',
                        help='DetNet state_dict (.pth). For the unpruned '
                             'baseline use it alone; for a pruned + fine-tuned '
                             'model also pass --arch. Ignored with --quant '
                             'or --mediapipe.')
    parser.add_argument('--arch', type=str, default=None,
                        help='Pruned architecture file (an _noft.pth from '
                             'pruned_architectures/). Required when --checkpoint '
                             'is a pruned + fine-tuned state_dict.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='Device for FP32 DetNet (baseline or pruned). '
                             'Default cpu to match evaluate_detnet. --quant is '
                             'always CPU; --mediapipe is always CPU tflite.')
    parser.add_argument('--quant', choices=['qmm', 'qmse'], default=None,
                        help='Use the INT8 quantized DetNet (re-quantized '
                             'in-process via quant/qquant.py). Overrides '
                             '--checkpoint / --arch / --device.')
    parser.add_argument('--mediapipe', action='store_true',
                        help='Use MediaPipe Hands AS THE PREDICTOR; skip DetNet '
                             'entirely. Lets you compare end-to-end FPS '
                             'against the DetNet variants on the same loop.')
    parser.add_argument('--mp_every_n', type=int, default=1,
                        help='Run the MediaPipe detector every N frames '
                             '(default 1 = every frame). The cached bbox / '
                             'landmarks are reused between detections. Bigger '
                             'N -> faster FPS at the cost of bbox / landmark lag.')
    parser.add_argument('--seconds', type=float, default=None,
                        help='Auto-quit after this many seconds of loop time '
                             '(default: run until "q" pressed). Useful for '
                             'benchmarking variants for exactly comparable '
                             'durations. Timer starts when the loop begins, '
                             'so model load + camera init are NOT included.')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Disable One Euro smoothing filter on DetNet joints')
    args = parser.parse_args()

    if args.mp_every_n < 1:
        parser.error('--mp_every_n must be >= 1')
    if args.seconds is not None and args.seconds <= 0:
        parser.error('--seconds must be > 0')

    # ── Model dispatch ────────────────────────────────────────────────────────
    if args.mediapipe:
        module = None
        device = None
        model_label = 'MediaPipe Hands (no DetNet, CPU tflite)'
        print('Mode: MediaPipe-only (no DetNet)')
    elif args.quant is not None:
        device = torch.device('cpu')
        print(f'Device: {device}   (forced — INT8 FBGEMM is CPU-only)')
        module = load_quantized(args.quant)
        model_label = f'{args.quant.upper()} (INT8, CPU)'
    else:
        want_cuda = (args.device == 'cuda') and torch.cuda.is_available()
        if args.device == 'cuda' and not want_cuda:
            print('WARNING: --device cuda requested but CUDA unavailable — using CPU.')
        device = torch.device('cuda:0' if want_cuda else 'cpu')
        print(f'Device: {device}')
        module = load_model(args.checkpoint, device, arch_path=args.arch)
        which = 'CUDA' if want_cuda else 'CPU'
        if args.arch:
            model_label = (f'Pruned {os.path.basename(args.arch)} '
                           f'(+ {os.path.basename(args.checkpoint)}, {which})')
        else:
            model_label = f'Baseline {os.path.basename(args.checkpoint)} ({which})'

    # MediaPipe smooths internally; OneEuro is for DetNet joints. Skip for MP-only.
    joint_filter = None if (args.no_filter or args.mediapipe) else smoother.OneEuroFilter(4.0, 0.0)

    # ── MediaPipe (used for bbox in DetNet modes, full landmarks in --mediapipe mode) ─
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import urllib.request

    mp_model_path = 'hand_landmarker.task'
    if not os.path.isfile(mp_model_path):
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        print(f'Downloading MediaPipe hand landmarker model to {mp_model_path} ...')
        urllib.request.urlretrieve(url, mp_model_path)
        print('Download complete.')

    hands_detector = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=mp_model_path),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
    )

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = open_camera(args.camera)
    if not cap.isOpened():
        print(f'Error: Cannot open camera {args.camera}')
        return
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ {cam_fps:.0f} FPS  "
          f"(MP detection every {args.mp_every_n} frame(s))")

    if args.seconds is not None:
        print(f"Running — auto-quit after {args.seconds:g} s (or press 'q').")
    else:
        print("Running — press 'q' to quit.")
    prev_time = time.time()
    crop = None    # last known DetNet bbox
    mp_pts = None  # last known MediaPipe landmarks

    # Session-wide FPS: count every webcam frame and divide by elapsed time at
    # quit. try/finally so a Ctrl+C still prints the summary + releases the cam.
    session_start = time.time()
    total_frames = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if args.seconds is not None and (time.time() - session_start) >= args.seconds:
                break

            # --- MediaPipe: every Nth frame; cached bbox/landmarks reused otherwise ---
            if (total_frames - 1) % args.mp_every_n == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                new_crop, new_pts = get_hand_detection(frame_rgb, hands_detector)
                if new_crop is not None:
                    crop = new_crop
                    mp_pts = new_pts

            # --- Predict + draw on the RAW frame (which we then mirror) ---
            # DetNet was trained on un-mirrored hand crops; feeding it a flipped
            # frame would invert its handedness assumptions (thumb on the wrong
            # side, joints relabelled). So we detect on the raw frame here,
            # draw the skeleton/bbox on it, and flip the whole image once
            # afterwards. The skeleton overlay therefore flips together with
            # the live feed, staying aligned with the user's actual hand.
            if args.mediapipe:
                # MediaPipe-only — landmarks ARE the prediction; no DetNet.
                if mp_pts is not None:
                    _draw_skeleton_at(frame, mp_pts)
                    label  = 'MediaPipe landmarks'
                    colour = (0, 255, 0)
                else:
                    label  = 'No hand detected'
                    colour = (0, 0, 255)
            elif crop is not None:
                x1, y1, x2, y2 = crop
                region_w = x2 - x1
                region_h = y2 - y1

                # --- DetNet inference on the hand crop ---
                # Run preprocess BEFORE drawing on `frame` so the bbox + skeleton
                # pixels never leak into the model input.
                input_tensor = preprocess(frame, device, crop=crop)
                with torch.no_grad():
                    result = module(input_tensor)

                uv = result['uv'].squeeze(0).cpu().numpy().astype(np.float32)  # (21, 2)

                hmap = result['h_map'].squeeze(0)
                confidence = hmap.view(21, -1).max(dim=1).values.mean().item()

                if joint_filter is not None:
                    uv = joint_filter.process(uv)

                # Now safe to draw on frame — preprocess has already consumed it.
                if confidence > 0.2:
                    draw_skeleton(frame, uv, region_h, region_w, origin=(x1, y1))

                label  = f'conf: {confidence:.2f}' if confidence > 0.2 else 'No hand detected'
                colour = (0, 255, 0) if confidence > 0.2 else (0, 0, 255)
            else:
                # No hand acquired yet (DetNet mode) — reset filter for first capture
                joint_filter = None if args.no_filter else smoother.OneEuroFilter(4.0, 0.0)
                label  = 'No hand detected'
                colour = (0, 0, 255)

            # Mirror the raw frame (with skeleton + bbox already drawn on it)
            # so the live feed acts like a mirror — the user's right hand
            # appears on the right side of the window.
            display = cv2.flip(frame, 1)

            # FPS + status overlay drawn AFTER the flip so text reads normally
            # (otherwise putText output would be horizontally reversed).
            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-9)
            prev_time = now
            cv2.putText(display, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display, label, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

            cv2.imshow('DetNet Hand Tracking', display)
    finally:
        duration = time.time() - session_start
        mean_fps = (total_frames / duration) if duration > 0 else 0.0
        print('\n' + '=' * 50)
        print('  Session statistics')
        print('=' * 50)
        print(f'  Model         : {model_label}')
        print(f'  MP every N    : {args.mp_every_n}')
        if args.seconds is not None:
            print(f'  Auto-quit     : {args.seconds:g} s')
        print(f'  Total frames  : {total_frames}')
        print(f'  Session time  : {duration:.2f} s')
        print(f'  Mean FPS      : {mean_fps:.2f}    (includes warmup)')
        print('=' * 50)

        hands_detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
