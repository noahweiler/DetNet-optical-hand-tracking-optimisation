"""
Visualise dataset samples with GT skeleton and (optionally) DetNet predictions.

Draws the same coloured-bone skeleton as the webcam script overlaid on the
original dataset images so you can inspect the annotation quality and
see how well DetNet predicts on real samples.

FreiHAND, GANeratedHands, and RHD have full 21-joint annotations → full skeleton.
Dexter+Object and EgoDexter annotate only visible fingertips → dots drawn for GT,
but the DetNet prediction panel still shows the full 21-joint skeleton.

Controls
--------
  any key  : next sample
  'q'      : quit
  's'      : save current image to disk

Usage
-----
  python visualise_dataset.py --dataset gan
  python visualise_dataset.py --dataset freihand --checkpoint new_check_point/ckp_detnet_106.pth
  python visualise_dataset.py --dataset rhd --split train
  python visualise_dataset.py --dataset dexter
  python visualise_dataset.py --dataset egodexter --checkpoint new_check_point/ckp_detnet_106.pth

  # Start at sample 50, show 20 samples
  python visualise_dataset.py --dataset gan --skip 50 --n_samples 20
"""

import argparse
import glob
import json
import os
import pickle

import cv2
import numpy as np
import torch
from PIL import Image

from config import SNAP_BONES, JOINT_COLORS
from model.detnet import detnet
from utils import func, handutils

HEATMAP_SIZE = 32
INPUT_SIZE   = 128

# Dexter+Object and EgoDexter sequence names
_DEXTER_SEQS    = ['Grasp1', 'Grasp2', 'Occlusion', 'Pinch', 'Rigid', 'Rotate']
_EGODEXTER_SEQS = ['Desk', 'Fruits', 'Kitchen', 'Rotunda']


# ── Skeleton drawing ──────────────────────────────────────────────────────────

def draw_skeleton_px(img, pts_xy, color_override=None, radius=5, thickness=2):
    """
    Draw skeleton on img.
    pts_xy : (21, 2) float array of (x, y) pixel coordinates.
    color_override : if set, use this BGR colour for all bones/joints.
    """
    pts = pts_xy.astype(np.int32)
    for finger_idx, bone_chain in enumerate(SNAP_BONES):
        if color_override:
            bgr = color_override
        else:
            r, g, b = JOINT_COLORS[finger_idx]
            bgr = (b, g, r)
        for i in range(len(bone_chain) - 1):
            j1, j2 = bone_chain[i], bone_chain[i + 1]
            p1, p2 = tuple(pts[j1]), tuple(pts[j2])
            h, w = img.shape[:2]
            if (-w < p1[0] < 2*w and -h < p1[1] < 2*h and
                    -w < p2[0] < 2*w and -h < p2[1] < 2*h):
                cv2.line(img, p1, p2, bgr, thickness, cv2.LINE_AA)

    for x, y in pts:
        if -50 < x < img.shape[1] + 50 and -50 < y < img.shape[0] + 50:
            cv2.circle(img, (x, y), radius, (255, 255, 255), -1)
            cv2.circle(img, (x, y), radius, (0, 0, 0), 1, cv2.LINE_AA)


def draw_fingertips_only(img, pts_xy):
    """
    Draw circles at fingertip positions only (no bone chains).
    Used for Dexter+Object and EgoDexter which only annotate visible fingertips.
    """
    pts = pts_xy.astype(np.int32)
    for x, y in pts:
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 8, (0, 220, 0), -1)
            cv2.circle(img, (x, y), 8, (0, 0, 0), 1, cv2.LINE_AA)


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: '{checkpoint_path}'")
    module = detnet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    first_key = next(iter(ckpt))
    if first_key.startswith('module.'):
        ckpt = {k[len('module.'):]: v for k, v in ckpt.items()}
    state = module.state_dict()
    matched = {k: v for k, v in ckpt.items() if k in state}
    state.update(matched)
    module.load_state_dict(state)
    module.eval()
    print(f'Loaded {len(matched)}/{len(ckpt)} keys from checkpoint.')
    return module


def predict_on_image(model, img_bgr, kp2d_gt, device):
    """Run DetNet and return predicted (21,2) pixel coordinates."""
    center = handutils.get_annot_center(kp2d_gt)
    scale  = handutils.get_ori_crop_scale(mask=None, mask_flag=False,
                                          side=None, kp2d=kp2d_gt)
    affinetrans, _ = handutils.get_affine_transform_test(
        center, scale, [INPUT_SIZE, INPUT_SIZE], rot=0)

    pil_img  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil_crop = handutils.transform_img(pil_img, affinetrans, [INPUT_SIZE, INPUT_SIZE])
    pil_crop = pil_crop.crop((0, 0, INPUT_SIZE, INPUT_SIZE))

    tensor = func.to_tensor(pil_crop).float().to(device)
    tensor = func.normalize(tensor, [0.5, 0.5, 0.5], [1, 1, 1])
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        result = model(tensor)
    uv = result['uv'].squeeze(0).cpu().numpy().astype(np.float32)   # (21,2) [row,col]

    pts = np.zeros((21, 2), dtype=np.float32)
    pts[:, 0] = uv[:, 1] * scale / HEATMAP_SIZE + (center[0] - scale / 2)  # x
    pts[:, 1] = uv[:, 0] * scale / HEATMAP_SIZE + (center[1] - scale / 2)  # y
    return pts


# ── Dataset iterators ─────────────────────────────────────────────────────────

def iter_ganerated(data_root, skip, n_samples, subset='all'):
    """Yield (img_bgr, kp2d_gt (21,2), label) for GANeratedHands."""
    gan_data = os.path.join(data_root, 'GANeratedHands_Release', 'data')
    subsets  = ('noObject', 'withObject') if subset == 'all' else (subset,)
    count = 0
    total_seen = 0
    for sub in subsets:
        subset_path = os.path.join(gan_data, sub)
        if not os.path.isdir(subset_path):
            print(f'Warning: subset not found: {subset_path}')
            continue
        for part in sorted(os.listdir(subset_path)):
            part_path = os.path.join(subset_path, part)
            if not os.path.isdir(part_path):
                continue
            for img_path in sorted(glob.glob(
                    os.path.join(part_path, '*_color_composed.png'))):
                j2d_path = img_path.replace('_color_composed.png', '_joint2D.txt')
                if not os.path.isfile(j2d_path):
                    continue
                total_seen += 1
                if total_seen <= skip:
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue
                kp2d = np.loadtxt(j2d_path, delimiter=',').reshape(21, 2)
                label = f'GAN {sub}/{part} #{total_seen}'
                yield img, kp2d, label
                count += 1
                if count >= n_samples:
                    return


def iter_freihand(data_root, skip, n_samples, split='eval'):
    """Yield (img_bgr, kp2d_gt (21,2), label) for FreiHAND."""
    root = os.path.join(data_root, 'FreiHAND')
    if split == 'train':
        img_dir  = os.path.join(root, 'training', 'rgb')
        xyz_file = os.path.join(root, 'training_xyz.json')
        K_file   = os.path.join(root, 'training_K.json')
    else:
        img_dir  = os.path.join(root, 'evaluation', 'rgb')
        xyz_file = os.path.join(root, 'evaluation_xyz.json')
        K_file   = os.path.join(root, 'evaluation_K.json')

    with open(xyz_file) as f:
        xyz_list = json.load(f)
    with open(K_file) as f:
        K_list = json.load(f)
    img_files = sorted(
        glob.glob(os.path.join(img_dir, '*.jpg')) +
        glob.glob(os.path.join(img_dir, '*.png'))
    )
    n = min(skip + n_samples, len(xyz_list), len(img_files))
    for i in range(skip, n):
        img = cv2.imread(img_files[i])
        if img is None:
            continue
        xyz = np.array(xyz_list[i], dtype=np.float32)
        K   = np.array(K_list[i],   dtype=np.float32)
        uvw  = K @ xyz.T
        kp2d = (uvw[:2] / uvw[2]).T   # (21,2) [x,y]
        label = f'FreiHAND {split} #{i}'
        yield img, kp2d, label


# RHD joint ordering is tip→MCP (distal→proximal) per finger, opposite to SNAP.
# This index array reorders RHD joints into SNAP (MCP→tip) order so that
# SNAP_BONES draws the correct palm→tip bone chains.
_RHD_TO_SNAP = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]


def iter_rhd(data_root, skip, n_samples, split='eval'):
    """
    Yield (img_bgr, kp2d_gt (21,2), label) for RHD.

    Annotations have 42 joints: 0-20 = left hand, 21-41 = right hand.
    Whichever hand has more visible joints is selected. If the left hand
    dominates the image is flipped horizontally and x-coords are mirrored
    so DetNet (right-hand model) sees a consistent right hand.

    RHD joint order within each finger is tip→MCP; joints are reordered to
    SNAP's MCP→tip convention so SNAP_BONES draws correctly.
    """
    if split == 'train':
        data_dir  = os.path.join(data_root, 'RHD', 'RHD_published_v2', 'training')
        anno_file = os.path.join(data_dir, 'anno_training.pickle')
    else:
        data_dir  = os.path.join(data_root, 'RHD', 'RHD_published_v2', 'evaluation')
        anno_file = os.path.join(data_dir, 'anno_evaluation.pickle')

    img_dir = os.path.join(data_dir, 'color')

    with open(anno_file, 'rb') as f:
        anno = pickle.load(f, encoding='latin1')

    img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    n = min(skip + n_samples, len(img_files), len(anno))
    count = 0

    for i in range(skip, n):
        img = cv2.imread(img_files[i])
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        uv_vis    = anno[i]['uv_vis']           # (42, 3): [u, v, visibility]
        vis_left  = uv_vis[:21, 2].sum()
        vis_right = uv_vis[21:, 2].sum()

        if vis_right >= vis_left:
            kp2d  = uv_vis[21:, :2].copy()     # right hand, original image
            hand  = 'R'
        else:
            kp2d           = uv_vis[:21, :2].copy()
            kp2d[:, 0]     = img_w - kp2d[:, 0]   # mirror x
            img            = cv2.flip(img, 1)       # flip image to match
            hand           = 'L→flipped'

        # Reorder from RHD (tip→MCP) to SNAP (MCP→tip) joint convention
        kp2d = kp2d[_RHD_TO_SNAP]

        label = f'RHD {split} #{i} ({hand})'
        yield img, kp2d, label
        count += 1
        if count >= n_samples:
            return


def iter_dexter(data_root, skip, n_samples):
    """
    Yield (img_bgr, kp2d_valid (n,2), label) for Dexter+Object.

    GT is fingertips only (up to 5 per frame). Frames with no valid
    fingertips (z == 32001 in the 3D file) are skipped.
    kp2d_valid contains only the visible fingertips, not all 5.
    """
    data_dir  = os.path.join(data_root, 'dexter+object', 'data')
    total_seen = 0
    count      = 0

    for seq in _DEXTER_SEQS:
        seq_dir      = os.path.join(data_dir, seq)
        anno_dir     = os.path.join(seq_dir, 'annotations')
        anno_2d_file = os.path.join(anno_dir, f'{seq}2D.txt')
        anno_3d_file = os.path.join(anno_dir, f'{seq}3D.txt')

        if not os.path.isfile(anno_2d_file):
            print(f'Warning: not found: {anno_2d_file}')
            continue

        with open(anno_2d_file) as f: lines_2d = f.readlines()
        with open(anno_3d_file) as f: lines_3d = f.readlines()

        img_files = sorted(glob.glob(os.path.join(seq_dir, 'color', '*.bmp')))
        n_frames  = min(len(lines_2d), len(lines_3d), len(img_files))

        for i in range(n_frames):
            if count >= n_samples:
                return

            # First 5 semicolon-separated entries are fingertips; rest are cuboid corners
            pairs_2d   = [p.strip() for p in lines_2d[i].split(';')][:5]
            kp2d_all   = np.array([[float(v) for v in p.split(',')]
                                    for p in pairs_2d], dtype=np.float32)  # (5, 2)
            pairs_3d   = [p.strip() for p in lines_3d[i].split(';')][:5]
            valid_mask = np.array(
                [float(p.split(',')[2]) != 32001 for p in pairs_3d], dtype=bool
            )

            if valid_mask.sum() == 0:
                continue

            total_seen += 1
            if total_seen <= skip:
                continue

            img = cv2.imread(img_files[i])
            if img is None:
                continue

            kp2d_valid = kp2d_all[valid_mask]   # (n_valid, 2) visible fingertips
            label = f'Dexter {seq} #{i}  ({int(valid_mask.sum())}/5 tips)'
            yield img, kp2d_valid, label
            count += 1


def iter_egodexter(data_root, skip, n_samples):
    """
    Yield (img_bgr, kp2d_valid (n,2), label) for EgoDexter.

    GT is fingertips only (up to 5 per frame). Entries with x < 0 or y < 0
    are unannotated and excluded. Frames with no valid tips are skipped.
    """
    data_dir   = os.path.join(data_root, 'EgoDexter', 'data')
    total_seen = 0
    count      = 0

    for seq in _EGODEXTER_SEQS:
        seq_dir   = os.path.join(data_dir, seq)
        anno_file = os.path.join(seq_dir, 'annotation.txt')

        if not os.path.isfile(anno_file):
            print(f'Warning: not found: {anno_file}')
            continue

        with open(anno_file) as f:
            anno_lines = f.readlines()

        img_files = sorted(glob.glob(os.path.join(seq_dir, 'color', '*.png')))
        n_frames  = min(len(anno_lines), len(img_files))

        for i in range(n_frames):
            if count >= n_samples:
                return

            pairs = [p.strip() for p in anno_lines[i].split(';') if p.strip()]
            kp2d_valid = []
            for p in pairs[:5]:
                vals = [float(v) for v in p.split(',')]
                if vals[0] >= 0 and vals[1] >= 0:   # -1,-1 = unannotated
                    kp2d_valid.append(vals)

            if not kp2d_valid:
                continue

            total_seen += 1
            if total_seen <= skip:
                continue

            img = cv2.imread(img_files[i])
            if img is None:
                continue

            kp2d_arr = np.array(kp2d_valid, dtype=np.float32)   # (n_valid, 2)
            label = f'EgoDexter {seq} #{i}  ({len(kp2d_valid)}/5 tips)'
            yield img, kp2d_arr, label
            count += 1


# ── Rendering ─────────────────────────────────────────────────────────────────

SCALE = 2   # upscale factor for display (256→512 or 224→448)

def render(img_bgr, kp2d_gt, kp2d_pred, label, sample_idx, fingertips_only=False):
    """
    Build a two-panel display image — always side by side:
      left  panel : original image + GT annotation
      right panel : original image + DetNet prediction (or grey placeholder)

    fingertips_only : if True, GT is (n, 2) visible fingertip coords — draw
                      dots instead of bone chains (Dexter+Object / EgoDexter).
    """
    h, w = img_bgr.shape[:2]
    disp_h, disp_w = h * SCALE, w * SCALE

    # ── Left panel: ground truth ───────────────────────────────────────────────
    left = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    gt_scaled = kp2d_gt * SCALE
    if fingertips_only:
        draw_fingertips_only(left, gt_scaled)
    else:
        draw_skeleton_px(left, gt_scaled)
    cv2.putText(left, 'GT', (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(left, 'GT', (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 1, cv2.LINE_AA)

    # ── Right panel: DetNet prediction (or placeholder) ────────────────────────
    right = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    if kp2d_pred is not None:
        pred_scaled = kp2d_pred * SCALE
        draw_skeleton_px(right, pred_scaled)
        cv2.putText(right, 'DetNet', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right, 'DetNet', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 1, cv2.LINE_AA)
    else:
        # Dim the right panel and show a message when no model is loaded
        right = (right * 0.35).astype(np.uint8)
        cv2.putText(right, 'No checkpoint loaded', (10, disp_h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(right, '--checkpoint <path>', (10, disp_h // 2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    canvas = np.hstack([left, right])
    cv2.putText(canvas, f'#{sample_idx}  {label}', (8, canvas.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, f'#{sample_idx}  {label}', (8, canvas.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualise dataset with skeleton overlay')
    parser.add_argument('--dataset',
                        choices=['freihand', 'gan', 'rhd', 'dexter', 'egodexter'],
                        required=True)
    parser.add_argument('--data_root', type=str,
                        default='C:\Imperial\FYP Data')
    parser.add_argument('--checkpoint', type=str,
                        default='new_check_point/ckp_detnet_83.pth',
                        help='DetNet checkpoint — if given, also shows prediction panel')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip the first N samples (default: 0)')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Maximum samples to browse (default: 200)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Save annotated images to this folder instead of displaying')
    parser.add_argument('--split', choices=['eval', 'train'], default='eval',
                        help='Split: "eval" or "train". Applies to FreiHAND and RHD. '
                             'Ignored for GANeratedHands.')
    parser.add_argument('--gan_subset', choices=['all', 'noObject', 'withObject'],
                        default='all',
                        help='GANeratedHands subset: "all", "noObject", or "withObject". '
                             'Ignored for other datasets.')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = None
    if args.checkpoint:
        model = load_model(args.checkpoint, device)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Fingertip-only datasets: GT has only visible fingertips, not 21 joints
    fingertips_only = args.dataset in ('dexter', 'egodexter')

    if args.dataset == 'gan':
        it = iter_ganerated(args.data_root, args.skip, args.n_samples,
                            subset=args.gan_subset)
    elif args.dataset == 'rhd':
        it = iter_rhd(args.data_root, args.skip, args.n_samples,
                      split=args.split)
    elif args.dataset == 'dexter':
        it = iter_dexter(args.data_root, args.skip, args.n_samples)
    elif args.dataset == 'egodexter':
        it = iter_egodexter(args.data_root, args.skip, args.n_samples)
    else:
        it = iter_freihand(args.data_root, args.skip, args.n_samples,
                           split=args.split)

    win = 'Dataset visualisation  (any key = next | q = quit | s = save)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for idx, (img, kp2d_gt, label) in enumerate(it):
        kp2d_pred = None
        if model is not None:
            # For fingertip datasets, GT points are used to compute the crop
            # centre/scale; the prediction panel shows the full 21-joint skeleton.
            kp2d_pred = predict_on_image(model, img, kp2d_gt, device)

        canvas = render(img, kp2d_gt, kp2d_pred, label, idx + args.skip,
                        fingertips_only=fingertips_only)

        if args.save_dir:
            out_path = os.path.join(args.save_dir, f'sample_{idx + args.skip:05d}.png')
            cv2.imwrite(out_path, canvas)
            print(f'Saved {out_path}')
            continue

        cv2.imshow(win, canvas)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            save_path = f'vis_sample_{idx + args.skip:05d}.png'
            cv2.imwrite(save_path, canvas)
            print(f'Saved {save_path}')
            cv2.waitKey(200)

    cv2.destroyAllWindows()
    print('Done.')


if __name__ == '__main__':
    main()
