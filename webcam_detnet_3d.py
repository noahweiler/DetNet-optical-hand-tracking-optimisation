"""
Real-time hand tracking with 2D skeleton + 3D skeleton side by side.

Left panel  : webcam feed with 2D skeleton overlay  (DetNet heatmap argmax)
Right panel : 3D hand skeleton from MediaPipe world landmarks (metres)

DetNet is 2D-only (3D heads disabled in model/detnet/detnet.py — see the
commented-out dmap_0/lmap_0 there), so the 3D panel is driven by MediaPipe's
hand_world_landmarks. That output is already produced by the same
HandLandmarker call used to crop the input, so adding the 3D panel costs no
extra MediaPipe pass and no change to DetNet.

Press 'q' to quit.

Usage:
    python webcam_detnet_3d.py
    python webcam_detnet_3d.py --checkpoint path/to/ckp_detnet.pth
    python webcam_detnet_3d.py --camera 1
    python webcam_detnet_3d.py --no-filter
"""
import argparse
import os
import time

import cv2
import matplotlib
matplotlib.use('Agg')   # off-screen backend — no separate window
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from model.detnet import detnet
from utils import func, smoother
from config import SNAP_BONES, JOINT_COLORS

HEATMAP_SIZE = 32
INPUT_SIZE   = 128
SHAPE_PANEL  = 400   # pixel size of the 3D panel (square)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'\n"
            "Please pass the correct path with --checkpoint."
        )
    module = detnet().to(device)
    print(f'Loading checkpoint: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location=device)
    first_key = next(iter(ckpt))
    if first_key.startswith('module.'):
        print('Stripping DataParallel "module." prefix.')
        ckpt = {k[len('module.'):]: v for k, v in ckpt.items()}
    model_state = module.state_dict()
    state = {k: v for k, v in ckpt.items() if k in model_state}
    print(f'Loaded {len(state)}/{len(ckpt)} keys from checkpoint.')
    model_state.update(state)
    module.load_state_dict(model_state)
    module.eval()
    return module


# ── MediaPipe hand crop ───────────────────────────────────────────────────────

def get_hand_detection(frame_rgb, hands_detector, pad=0.3):
    """One MediaPipe call -> ((x1,y1,x2,y2) square bbox, (21,3) world xyz in m).

    Both outputs come from the same HandLandmarker.detect call: the 2D image
    landmarks build the bbox DetNet crops on, and hand_world_landmarks are the
    root-relative 3D joints (metres) that drive the right-hand 3D panel.
    MediaPipe and DetNet share the same 21-joint SNAP ordering.

    Returns (None, None) if no hand is detected.
    """
    import mediapipe as mp
    h, w = frame_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = hands_detector.detect(mp_image)
    if not results.hand_landmarks:
        return None, None
    lms = results.hand_landmarks[0]
    xs = [lm.x * w for lm in lms]
    ys = [lm.y * h for lm in lms]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad;  x2 += bw * pad
    y1 -= bh * pad;  y2 += bh * pad
    bw, bh = x2 - x1, y2 - y1
    if bw > bh:
        diff = bw - bh; y1 -= diff / 2; y2 += diff / 2
    else:
        diff = bh - bw; x1 -= diff / 2; x2 += diff / 2
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    wlms = results.hand_world_landmarks[0]
    xyz = np.array([[lm.x, lm.y, lm.z] for lm in wlms], dtype=np.float32)
    return (x1, y1, x2, y2), xyz


# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess(frame_bgr, device, crop=None):
    if crop is not None:
        x1, y1, x2, y2 = crop
        img = frame_bgr[y1:y2, x1:x2].copy()
    else:
        img = frame_bgr.copy()
        fh, fw = img.shape[:2]
        if fh > fw:
            m = (fh - fw) // 2; img = img[m:m + fw]
        else:
            m = (fw - fh) // 2; img = img[:, m:m + fh]
    img = np.flip(img, -1).copy()                          # BGR → RGB
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    tensor = torch.tensor(img.transpose([2, 0, 1]),
                          dtype=torch.float, device=device) / 255.0
    tensor = func.normalize(tensor, [0.5, 0.5, 0.5], [1, 1, 1])
    return tensor.unsqueeze(0)


# ── 2-D skeleton overlay ──────────────────────────────────────────────────────

def draw_skeleton_2d(img, uv_joints, region_h, region_w, origin=(0, 0)):
    ox, oy = origin
    pts = np.zeros((21, 2), dtype=np.int32)
    pts[:, 0] = (uv_joints[:, 1] * region_w / HEATMAP_SIZE).astype(np.int32) + ox
    pts[:, 1] = (uv_joints[:, 0] * region_h / HEATMAP_SIZE).astype(np.int32) + oy
    for finger_idx, bone_chain in enumerate(SNAP_BONES):
        r, g, b = JOINT_COLORS[finger_idx]
        bgr = (b, g, r)
        for i in range(len(bone_chain) - 1):
            j1, j2 = bone_chain[i], bone_chain[i + 1]
            cv2.line(img, tuple(pts[j1]), tuple(pts[j2]), bgr, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(img, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)


# ── 3-D skeleton panel ────────────────────────────────────────────────────────

class Shape3DRenderer:
    """
    Maintains a matplotlib 3D figure.
    Call render(xyz) each frame to get a BGR numpy image.
    """

    def __init__(self, size=SHAPE_PANEL):
        dpi = 100
        px = size / dpi
        self.fig = plt.figure(figsize=(px, px), dpi=dpi)
        self.ax  = self.fig.add_subplot(111, projection='3d')
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.set_facecolor('#1a1a1a')
        self._init_lines()
        self._style_axes()
        self.fig.tight_layout(pad=0)

    def _init_lines(self):
        self.bone_lines = []
        colors = JOINT_COLORS
        for finger_idx, bone_chain in enumerate(SNAP_BONES):
            r, g, b = colors[finger_idx]
            col = (r / 255, g / 255, b / 255)
            for _ in range(len(bone_chain) - 1):
                ln, = self.ax.plot([], [], [], '-', color=col,
                                   linewidth=2, solid_capstyle='round')
                self.bone_lines.append((ln, bone_chain))
        self.dots, = self.ax.plot([], [], [], 'o', color='white',
                                  markersize=4, markeredgecolor='black',
                                  markeredgewidth=0.5)

    def _style_axes(self):
        ax = self.ax
        ax.set_xlabel('X', color='#888888', labelpad=2, fontsize=7)
        ax.set_ylabel('Z', color='#888888', labelpad=2, fontsize=7)
        ax.set_zlabel('Y', color='#888888', labelpad=2, fontsize=7)
        ax.tick_params(colors='#666666', labelsize=6)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor('#333333')
        ax.grid(True, color='#333333', linewidth=0.5)
        ax.view_init(elev=20, azim=-60)
        ax.set_title('3D skeleton', color='white', fontsize=8, pad=2)
        # MediaPipe world landmarks are in metres, root-relative; a full hand
        # spans ~0.18 m corner-to-corner, so ±0.12 m gives a small framing margin.
        ax.set_xlim(-0.12, 0.12)
        ax.set_ylim(-0.12, 0.12)
        ax.set_zlim(-0.12, 0.12)

    def render(self, xyz):
        """
        xyz : (21,3) MediaPipe world landmarks — root-relative, metres, y-down.
              We flip y → y-up for display.
        Returns : (H,W,3) BGR uint8 image.
        """
        pts = xyz.copy()
        pts[:, 1] *= -1   # y-down → y-up

        line_idx = 0
        colors = JOINT_COLORS
        for finger_idx, bone_chain in enumerate(SNAP_BONES):
            r, g, b = colors[finger_idx]
            col = (r / 255, g / 255, b / 255)
            for i in range(len(bone_chain) - 1):
                j1, j2 = bone_chain[i], bone_chain[i + 1]
                ln = self.bone_lines[line_idx][0]
                ln.set_data([pts[j1, 0], pts[j2, 0]],
                            [pts[j1, 2], pts[j2, 2]])
                ln.set_3d_properties([pts[j1, 1], pts[j2, 1]])
                ln.set_color(col)
                line_idx += 1

        self.dots.set_data(pts[:, 0], pts[:, 2])
        self.dots.set_3d_properties(pts[:, 1])

        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = self.fig.canvas.get_width_height()
        img_rgba = buf.reshape(h, w, 4)
        return cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

    def blank(self):
        """Return a dark placeholder when no hand is detected."""
        blank = np.full((SHAPE_PANEL, SHAPE_PANEL, 3), 26, dtype=np.uint8)
        cv2.putText(blank, 'No hand', (SHAPE_PANEL // 2 - 45, SHAPE_PANEL // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1, cv2.LINE_AA)
        return blank


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DetNet webcam demo — 2D tracking + 3D skeleton')
    parser.add_argument('--checkpoint', type=str,
                        default='new_check_point/ckp_detnet_83.pth')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--no-filter', action='store_true',
                        help='Disable One Euro smoothing')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    module = load_model(args.checkpoint, device)
    uv_filter  = None if args.no_filter else smoother.OneEuroFilter(4.0, 0.0)
    xyz_filter = None if args.no_filter else smoother.OneEuroFilter(4.0, 0.0)

    renderer = Shape3DRenderer(size=SHAPE_PANEL)

    # MediaPipe setup
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import urllib.request

    model_path = 'hand_landmarker.task'
    if not os.path.isfile(model_path):
        url = ('https://storage.googleapis.com/mediapipe-models/'
               'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
        print(f'Downloading MediaPipe model to {model_path} ...')
        urllib.request.urlretrieve(url, model_path)

    hands_detector = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'Error: cannot open camera {args.camera}')
        return

    print("Running — press 'q' to quit.")
    prev_time = time.time()
    crop = None
    xyz_world = None     # latest MediaPipe (21,3) world landmarks, metres
    shape_panel = renderer.blank()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        display   = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Single MediaPipe call -> bbox for DetNet AND 3D world joints for the panel
        new_crop, new_xyz = get_hand_detection(frame_rgb, hands_detector)
        if new_crop is not None:
            crop = new_crop
            xyz_world = new_xyz

        if crop is not None:
            x1, y1, x2, y2 = crop
            region_w, region_h = x2 - x1, y2 - y1

            cv2.rectangle(display, (x1, y1), (x2, y2), (200, 200, 200), 1)

            input_tensor = preprocess(frame, device, crop=crop)
            with torch.no_grad():
                result = module(input_tensor)

            uv   = result['uv'].squeeze(0).cpu().numpy().astype(np.float32)
            conf = result['h_map'].squeeze(0).view(21, -1).max(dim=1).values.mean().item()
            xyz  = xyz_world.copy()    # MediaPipe; DetNet no longer produces xyz

            if uv_filter is not None:
                uv = uv_filter.process(uv)
            if xyz_filter is not None:
                xyz = xyz_filter.process(xyz)

            if conf > 0.2:
                draw_skeleton_2d(display, uv, region_h, region_w, origin=(x1, y1))
                shape_panel = renderer.render(xyz)

            label  = f'conf: {conf:.2f}' if conf > 0.2 else 'No hand detected'
            colour = (0, 255, 0) if conf > 0.2 else (0, 0, 255)
        else:
            uv_filter  = None if args.no_filter else smoother.OneEuroFilter(4.0, 0.0)
            xyz_filter = None if args.no_filter else smoother.OneEuroFilter(4.0, 0.0)
            shape_panel = renderer.blank()
            label  = 'No hand detected'
            colour = (0, 0, 255)

        # FPS + status overlay on left panel
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now
        cv2.putText(display, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

        # Resize 3D panel to match frame height
        fh = display.shape[0]
        panel_resized = cv2.resize(shape_panel, (SHAPE_PANEL * fh // SHAPE_PANEL, fh))

        # Combine left (webcam) + right (3D skeleton) side by side
        combined = np.hstack([display, panel_resized])
        cv2.imshow('DetNet  |  left: 2D tracking   right: 3D skeleton', combined)

    hands_detector.close()
    cap.release()
    cv2.destroyAllWindows()
    plt.close('all')


if __name__ == '__main__':
    main()
