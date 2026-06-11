"""
visualise_overlay.py — single FreiHAND frame, single model, keypoint overlay PNG.

Picks one FreiHAND image (by filename or numeric index, on either split) and
saves a clean keypoint overlay as a raster PNG — no matplotlib axes, no titles,
no figure margins. The overlay is drawn directly onto the 128x128 evaluation
crop with cv2 (upscaled for crispness), so the output is a flat image you can
drop straight into a slide / doc.

Model selection — same source list as Fig 14:
  default          unpruned baseline (ckp_detnet_71.pth)
  --ckpt + --arch  any pruned + fine-tuned model
  --quant qmm      re-quantized INT8 MinMax (deterministic, ~30 s)
  --quant qmse     re-quantized INT8 Hist-MSE
  --gt_only        skip the model entirely, draw the GT keypoints

By default the model's predictions are drawn. Pass --gt to additionally overlay
the GT for direct visual comparison (GT bones get a thinner dashed style and
white-edged cyan dots so they don't blend into the prediction).

Usage:
  python visualise_overlay.py --image 00000047.jpg
  python visualise_overlay.py --image 00000047.jpg --gt
  python visualise_overlay.py --split train --image 00050000.jpg
  python visualise_overlay.py --image 00000047.jpg --quant qmse --upscale 8
  python visualise_overlay.py --image 00000047.jpg --gt_only
  python visualise_overlay.py --image 00000047.jpg ^
       --ckpt checkpoints_finetuned/l1_55pct/ft_l1_55pct_rhdbest.pth ^
       --arch pruned_architectures/detnet_l1_55pct_ep71_noft.pth
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Windows cv2+torch OMP Error #15

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import argparse

import cv2                       # noqa: F401 — keep ABOVE torch (Windows DLL order)
import numpy as np
import torch

from evaluate_detnet import (
    load_model, training_crop_and_transform, INPUT_SIZE, HEATMAP_SIZE,
)
from visualise_heatmaps import load_freihand_frame, crop_to_display
from visualise_qualitative import predict_kp_crop, gt_in_crop
from plot_pruning_results import SNAP_BONES, COLORS

# Per-finger colours (same palette as the Fig 14 grid). _FINGER_COLORS[b // 4]
# picks the colour for the b-th bone in SNAP_BONES (4 bones per finger).
_FINGER_HEX = [COLORS['red'], COLORS['blue'], COLORS['green'],
               COLORS['orange'], COLORS['purple']]


def _hex_to_bgr(s):
    """'#RRGGBB' -> (B, G, R) ints in [0, 255] for cv2."""
    s = s.lstrip('#')
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return (b, g, r)


_FINGER_BGR = [_hex_to_bgr(c) for c in _FINGER_HEX]


# ── Model loading ───────────────────────────────────────────────────────────

def load_quantized(tag, repo_root):
    """Re-quantize INT8 model in-process — same path Fig 14 uses."""
    quant_dir = os.path.join(repo_root, 'quant')
    if quant_dir not in sys.path:
        sys.path.insert(0, quant_dir)
    import qquant
    qcfg = {'qmm': qquant.M2_QCONFIG, 'qmse': qquant.M3_QCONFIG}[tag]
    print(f'  re-quantizing PTQ {tag.upper()} (deterministic, ~30 s) ...',
          flush=True)
    model, _ = qquant.quantize(qcfg)
    model.eval()
    return model


# ── Drawing ─────────────────────────────────────────────────────────────────

def _draw_skeleton(canvas, kp, *, bone_thick, dot_radius,
                   dot_fill, dot_edge, bone_colors, dashed=False):
    """Draw SNAP-bones skeleton onto canvas IN PLACE.

    kp        : (21, 2) pixel coords [x, y] in canvas space
    bone_thick: integer line thickness
    dot_*     : circle styling for the 21 joints
    bone_colors: list[5] of per-finger BGR (4 bones per finger via b // 4)
    dashed    : draw bones as short dashes instead of solid (used for GT-on-pred)
    """
    pts = np.round(kp).astype(np.int32)
    for b, (p, q) in enumerate(SNAP_BONES):
        col = bone_colors[b // 4]
        x1, y1 = pts[p]
        x2, y2 = pts[q]
        if dashed:
            # 4 px on / 4 px off — readable even at small bones
            length = max(1, int(np.hypot(x2 - x1, y2 - y1)))
            n_seg  = max(2, length // 8)
            on     = True
            for k in range(n_seg):
                a = k / n_seg
                b_ = (k + 1) / n_seg
                if on:
                    cv2.line(canvas,
                             (int(x1 + a  * (x2 - x1)), int(y1 + a  * (y2 - y1))),
                             (int(x1 + b_ * (x2 - x1)), int(y1 + b_ * (y2 - y1))),
                             col, bone_thick, cv2.LINE_AA)
                on = not on
        else:
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                     col, bone_thick, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(canvas, (int(x), int(y)), dot_radius, dot_fill, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(x), int(y)), dot_radius, dot_edge,  1, cv2.LINE_AA)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Single-frame keypoint overlay (no matplotlib)')
    ap.add_argument('--data_root', default='C:/Imperial/FYP Data')
    ap.add_argument('--split', choices=['eval', 'train'], default='eval')
    ap.add_argument('--index', type=int, default=0,
                    help='Numeric position in the sorted file list (default 0). '
                         'Ignored if --image is given.')
    ap.add_argument('--image', type=str, default=None,
                    help="FreiHAND filename (e.g. '00000047.jpg', or full path "
                         '— only the basename is used). The directory is '
                         'decided by --split. Overrides --index.')
    ap.add_argument('--ckpt', default='new_check_point/ckp_detnet_71.pth',
                    help='Checkpoint .pth (state_dict). Default: unpruned '
                         'baseline. For a pruned + fine-tuned model also pass '
                         '--arch. Ignored under --quant / --gt_only.')
    ap.add_argument('--arch', default=None,
                    help='Pruned architecture _noft.pth from '
                         'pruned_architectures/. Pair with the matching '
                         '--ckpt.')
    ap.add_argument('--quant', choices=['qmm', 'qmse'], default=None,
                    help='Use an INT8 PTQ model instead of --ckpt (re-quantized '
                         'in-process, deterministic). CPU-only (FBGEMM/x86).')
    ap.add_argument('--gt', action='store_true',
                    help='Also overlay the GT keypoints (dashed bones + cyan '
                         'dots) on top of the model prediction.')
    ap.add_argument('--gt_only', action='store_true',
                    help='Skip the model and draw ONLY the GT keypoints.')
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cpu',
                    help='cpu (default) or cuda. INT8 PTQ is forced CPU.')
    ap.add_argument('--upscale', type=int, default=4,
                    help='Upscale the 128x128 crop by this factor before '
                         'drawing for crisp bones/dots (default 4 -> 512x512).')
    ap.add_argument('--out_dir', default='figures')
    ap.add_argument('--out_name', default=None,
                    help='Output PNG basename (no extension). Default builds '
                         'one from the chosen frame + model tag.')
    args = ap.parse_args()

    if args.gt and args.gt_only:
        ap.error('--gt and --gt_only are mutually exclusive.')
    if args.quant is not None and args.gt_only:
        ap.error('--quant and --gt_only are mutually exclusive '
                 '(the quant model never runs in --gt_only mode).')

    device = torch.device('cuda:0' if (args.device == 'cuda'
                                       and torch.cuda.is_available()) else 'cpu')
    if args.quant is not None and device.type == 'cuda':
        print('PTQ INT8 forces CPU (FBGEMM/x86).')
        device = torch.device('cpu')
    print(f'Device : {device}')

    # ── Pick the frame + run the crop pipeline ─────────────────────────────
    img_bgr, kp2d_gt, frame_id = load_freihand_frame(
        args.data_root, args.split, index=args.index, image=args.image
    )
    tensor, center, scale = training_crop_and_transform(img_bgr, kp2d_gt, device)
    crop_rgb = crop_to_display(tensor)                       # (128, 128, 3) RGB
    print(f'Frame  : FreiHAND {args.split} {frame_id}')

    # ── Decide model + tag for the output filename ─────────────────────────
    pred_kp = None
    if args.gt_only:
        model_tag = 'gt'
    elif args.quant is not None:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        model = load_quantized(args.quant, repo_root)
        pred_kp = predict_kp_crop(model, tensor.cpu())
        model_tag = args.quant
    else:
        model = load_model(args.ckpt, device, arch_path=args.arch)
        pred_kp = predict_kp_crop(model, tensor)
        # Tag: 'baseline' if unpruned, else stem of the arch file
        if args.arch is None:
            model_tag = 'baseline'
        else:
            model_tag = os.path.splitext(os.path.basename(args.arch))[0]
        print(f"Model  : {model_tag}  (ckpt={args.ckpt})")

    gt_kp_crop = gt_in_crop(kp2d_gt, center, scale)

    # ── Compose the overlay canvas ─────────────────────────────────────────
    # crop_to_display returns RGB; cv2 writes BGR — flip for drawing + save.
    canvas = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    if args.upscale != 1:
        canvas = cv2.resize(canvas,
                            (INPUT_SIZE * args.upscale,
                             INPUT_SIZE * args.upscale),
                            interpolation=cv2.INTER_CUBIC)

    # Bone/dot sizes scale with the upscale factor so they look the same as
    # the matplotlib figure at any output resolution.
    bone_thick = max(1, 2 * args.upscale)
    dot_r      = max(2, 2 * args.upscale)   # ~= bone thickness — no oversized blobs

    # Predicted keypoints — solid bones, white dots with black edge.
    if pred_kp is not None:
        _draw_skeleton(canvas, pred_kp * args.upscale,
                       bone_thick=bone_thick, dot_radius=dot_r,
                       dot_fill=(255, 255, 255), dot_edge=(0, 0, 0),
                       bone_colors=_FINGER_BGR, dashed=False)

    # GT overlay — only when explicitly requested (--gt or --gt_only).
    # Dashed bones + cyan dots so they read as a "reference" layer on top of
    # the prediction in --gt mode, and stand alone naturally in --gt_only.
    if args.gt or args.gt_only:
        gt_bones = _FINGER_BGR if args.gt_only else [(220, 220, 220)] * 5
        gt_thick = bone_thick if args.gt_only else max(1, args.upscale)
        gt_dot_fill = (255, 255, 255) if args.gt_only else (255, 255, 0)  # cyan
        gt_dot_edge = (0, 0, 0)
        _draw_skeleton(canvas, gt_kp_crop * args.upscale,
                       bone_thick=gt_thick, dot_radius=dot_r,
                       dot_fill=gt_dot_fill, dot_edge=gt_dot_edge,
                       bone_colors=gt_bones, dashed=not args.gt_only)

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    if args.out_name is None:
        stem = os.path.splitext(frame_id)[0]            # '00000047'
        suffix = '_gt' if (args.gt and not args.gt_only) else ''
        args.out_name = f'overlay_{args.split}_{stem}_{model_tag}{suffix}'
    out_path = os.path.join(args.out_dir, args.out_name + '.png')
    cv2.imwrite(out_path, canvas)
    print(f'Wrote {out_path}  ({canvas.shape[1]}x{canvas.shape[0]} px)')


if __name__ == '__main__':
    main()
