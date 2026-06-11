"""
visualise_heatmaps.py — Fig 15: DetNet heatmap precision-loss comparison.

Takes ONE FreiHAND image, runs it through the exact evaluate_detnet.py crop +
forward pipeline for two models — the unpruned baseline and the most
aggressively pruned model (70% L1) — and renders, per model:

  1. the cropped 128x128 input the network actually sees;
  2. the max-across-21-channels heatmap overlaid on that crop, with the
     argmax-decoded predicted keypoints (the Fig-15-spec panel);
  3. all 21 joint heatmaps concatenated into ONE 3x7 montage (one space),
     so the per-joint sharpness vs diffuseness is directly visible.

Baseline vs 70% L1 share one colour scale, so the spatial-precision loss at
extreme compression is an honest visual comparison (not per-panel re-scaled).

It also writes figures/fig15_heatmap_data.npz in the exact contract
plot_pruning_results.plot_heatmap_comparison() expects (image / hmap_base /
hmap_pruned / label_pruned), plus the full 21-channel tensors, so the existing
figure15() keeps working — point its HEATMAP_DATA at this .npz.

Checkpoint shapes (same rules as evaluate_detnet.py):
  baseline : --checkpoint new_check_point/ckp_detnet_71.pth        (no --arch)
  pruned   : --checkpoint checkpoints_finetuned/l1_70pct/ft_l1_70pct_rhdbest.pth
             --arch       pruned_architectures/detnet_l1_70pct_ep71_noft.pth

Usage:
  python visualise_heatmaps.py
  python visualise_heatmaps.py --index 7 --device cpu
  python visualise_heatmaps.py --split train --index 1234
  python visualise_heatmaps.py --image 00000047.jpg            # pick by filename
  python visualise_heatmaps.py --split train --image 00050000.jpg
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Windows cv2+torch OMP Error #15

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')  # tee/redirect-safe on Windows
except Exception:
    pass

import argparse
import glob
import json

import cv2                       # noqa: F401 — keep ABOVE torch (Windows DLL order)
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')            # headless — no display needed
import matplotlib.pyplot as plt

# Reuse the *exact* crop + load pipeline so the heatmaps match what
# evaluate_detnet.py scores (importing also sets KMP / cv2-before-torch).
from evaluate_detnet import (
    load_model, training_crop_and_transform, project_3d_to_2d,
    INPUT_SIZE, HEATMAP_SIZE,
)

# Publication style — matches plot_pruning_results.py.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'DejaVu Serif'],
    'font.size':   9,
})

# Defaults mirror build_results_csv.py / the eval_results run names.
BASELINE_CKPT = 'new_check_point/ckp_detnet_71.pth'
PRUNED_CKPT   = 'checkpoints_finetuned/l1_70pct/ft_l1_70pct_rhdbest.pth'
PRUNED_ARCH   = 'pruned_architectures/detnet_l1_70pct_ep71_noft.pth'
PRUNED_LABEL  = 'L1 70%'


# ── Data ────────────────────────────────────────────────────────────────────

def load_freihand_frame(data_root, split, index=None, image=None):
    """Return (img_bgr, kp2d_gt, frame_id) for one FreiHAND frame — same GT
    projection evaluate_detnet.run_freihand uses (so center/scale match exactly).

    Pick the frame by EITHER --image (filename like '00000047.jpg', or full
    path — only the basename is used; the directory is decided by --split) or
    --index (numeric position in the sorted list). FreiHAND training has 4×
    composites of n_unique=32560 poses, so the GT lookup uses
    label_index = pos % n_unique — matching qcommon.run_freihand_with_latency.
    """
    root = os.path.join(data_root, 'FreiHAND')
    if split == 'train':
        img_dir  = os.path.join(root, 'training', 'rgb')
        xyz_file = os.path.join(root, 'training_xyz.json')
        K_file   = os.path.join(root, 'training_K.json')
    else:
        img_dir  = os.path.join(root, 'evaluation', 'rgb')
        xyz_file = os.path.join(root, 'evaluation_xyz.json')
        K_file   = os.path.join(root, 'evaluation_K.json')

    with open(xyz_file) as f: xyz_list = json.load(f)
    with open(K_file)   as f: K_list   = json.load(f)
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                       glob.glob(os.path.join(img_dir, '*.png')))
    n_unique = len(xyz_list)
    n_files  = len(img_files)

    if image is not None:
        target = os.path.basename(image)
        pos = next((i for i, f in enumerate(img_files)
                    if os.path.basename(f) == target), None)
        if pos is None:
            raise FileNotFoundError(
                f"--image '{target}' not found in FreiHAND {split} ({img_dir}). "
                f"Available range: {os.path.basename(img_files[0])} … "
                f"{os.path.basename(img_files[-1])} ({n_files} files)."
            )
    else:
        if not 0 <= index < n_files:
            raise IndexError(
                f'--index {index} out of range (0..{n_files - 1}) for FreiHAND {split}'
            )
        pos = index

    label_index = pos % n_unique     # 4× train composites share GT with first n_unique

    img = cv2.imread(img_files[pos])
    if img is None:
        raise FileNotFoundError(f'could not read {img_files[pos]}')
    kp2d_gt = project_3d_to_2d(xyz_list[label_index], K_list[label_index]).astype(np.float32)
    return img, kp2d_gt, os.path.basename(img_files[pos])


def crop_to_display(tensor):
    """Invert evaluate_detnet's normalise (x = x/255 - 0.5) → RGB uint8 crop,
    so we show exactly the 128x128 the network saw."""
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 0.5
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


# ── Forward ─────────────────────────────────────────────────────────────────

def run_model(ckpt, arch, device, tensor):
    """Load a model (baseline or pruned+finetuned) and return its
    (21, 32, 32) heatmap stack + (21, 2) heatmap-space predicted joints."""
    model = load_model(ckpt, device, arch_path=arch)
    with torch.no_grad():
        out = model(tensor)
    hm = out['h_map'].squeeze(0).cpu().numpy().astype(np.float32)  # (21,32,32)
    uv = out['uv'].squeeze(0).cpu().numpy().astype(np.float32)     # (21,2) row,col
    return hm, uv


def montage_21(hm, gap=1):
    """Concatenate the 21 joint heatmaps into ONE 3x7 image.
    Gap pixels are NaN so the colormap renders them as the 'bad' colour."""
    rows, cols, S = 3, 7, HEATMAP_SIZE
    H = rows * S + (rows - 1) * gap
    W = cols * S + (cols - 1) * gap
    canvas = np.full((H, W), np.nan, dtype=np.float32)
    for j in range(21):
        r, c = divmod(j, cols)
        y0, x0 = r * (S + gap), c * (S + gap)
        canvas[y0:y0 + S, x0:x0 + S] = hm[j]
    return canvas, rows, cols, S, gap


# ── Figure ──────────────────────────────────────────────────────────────────

def render(crop_rgb, models, vmax, cmap, out_base):
    """models: list of dicts {label, hm (21,32,32), uv (21,2)}.
    One row per model: [input crop | max-heatmap overlay | 21-montage]."""
    nrow = len(models)
    # Col 3 holds the 3x7 montage (~2.3:1), so give it more width than the
    # square input/overlay panels — otherwise it renders tiny.
    fig, axes = plt.subplots(nrow, 3, figsize=(11.0, 3.0 * nrow),
                             constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.0, 1.05, 1.75]})
    if nrow == 1:
        axes = axes[None, :]

    col_titles = ['Input (128×128 crop)',
                  'Max-across-channels heatmap',
                  'All 21 joint heatmaps (3×7)']
    bad_cmap = plt.get_cmap(cmap).copy()
    bad_cmap.set_bad('white')
    im_ref = None

    for i, m in enumerate(models):
        hm, uv = m['hm'], m['uv']

        # Col 0 — the exact crop the network saw.
        ax = axes[i, 0]
        ax.imshow(crop_rgb)
        ax.set_ylabel(m['label'], fontsize=11, fontweight='bold',
                      rotation=90, labelpad=8)

        # Col 1 — max over 21 channels, upsampled, overlaid; argmax keypoints.
        ax = axes[i, 1]
        hmax  = hm.max(axis=0)                                  # (32,32)
        big   = cv2.resize(hmax, (INPUT_SIZE, INPUT_SIZE),
                           interpolation=cv2.INTER_CUBIC)
        alpha = np.clip(big / (vmax + 1e-9), 0.0, 1.0) ** 0.7   # peaks opaque
        ax.imshow(crop_rgb)
        ax.imshow(big, cmap=cmap, vmin=0.0, vmax=vmax, alpha=alpha)
        scale = INPUT_SIZE / HEATMAP_SIZE
        ax.scatter(uv[:, 1] * scale, uv[:, 0] * scale, s=14,
                   facecolors='none', edgecolors='#00E5FF', linewidths=0.9)

        # Col 2 — the 21 heatmaps concatenated into one space.
        ax = axes[i, 2]
        canvas, rrows, rcols, S, gap = montage_21(hm)
        im_ref = ax.imshow(canvas, cmap=bad_cmap, vmin=0.0, vmax=vmax)
        for j in range(21):
            r, c = divmod(j, rcols)
            ax.text(c * (S + gap) + 2, r * (S + gap) + 5, str(j),
                    color='white', fontsize=6, va='top', ha='left')

        for c in range(3):
            axes[i, c].set_xticks([]); axes[i, c].set_yticks([])
            if i == 0:
                axes[i, c].set_title(col_titles[c], fontsize=10)

    cbar = fig.colorbar(im_ref, ax=axes[:, 2], shrink=0.85, pad=0.02)
    cbar.set_label('Heatmap confidence', fontsize=9)

    png, pdf = out_base + '.png', out_base + '.pdf'
    fig.savefig(png, dpi=800, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(pdf,            bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Wrote {png}')
    print(f'Wrote {pdf}')


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Fig 15 — DetNet heatmap precision-loss comparison')
    ap.add_argument('--data_root', default='C:/Imperial/FYP Data')
    ap.add_argument('--split', choices=['eval', 'train'], default='eval',
                    help='FreiHAND split to pull the frame from (default: eval)')
    ap.add_argument('--index', type=int, default=0,
                    help='Numeric position in the sorted file list of --split '
                         '(default: 0). Ignored if --image is given.')
    ap.add_argument('--image', type=str, default=None,
                    help="FreiHAND filename to visualise, e.g. '00000047.jpg' "
                         "(or a full path — only the basename is used). The "
                         "image is looked up in the directory for --split; "
                         "overrides --index.")
    ap.add_argument('--baseline_ckpt', default=BASELINE_CKPT)
    ap.add_argument('--pruned_ckpt',   default=PRUNED_CKPT)
    ap.add_argument('--pruned_arch',   default=PRUNED_ARCH)
    ap.add_argument('--pruned_label',  default=PRUNED_LABEL)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cpu',
                    help='cpu (default — single frame, avoids GPU contention) or cuda')
    ap.add_argument('--cmap', default='inferno',
                    help='matplotlib colormap for the heatmaps (default: inferno)')
    ap.add_argument('--out_dir', default='figures')
    args = ap.parse_args()

    device = torch.device('cuda:0' if (args.device == 'cuda'
                                       and torch.cuda.is_available()) else 'cpu')
    print(f'Device : {device}')

    img_bgr, kp2d_gt, frame_id = load_freihand_frame(
        args.data_root, args.split, index=args.index, image=args.image
    )
    tensor, _center, _scale = training_crop_and_transform(img_bgr, kp2d_gt, device)
    crop_rgb = crop_to_display(tensor)
    print(f'Frame  : FreiHAND {args.split} {frame_id}')

    specs = [
        dict(label='Baseline',           ckpt=args.baseline_ckpt, arch=None),
        dict(label=args.pruned_label,    ckpt=args.pruned_ckpt,   arch=args.pruned_arch),
    ]
    models = []
    for s in specs:
        print(f'-> {s["label"]}: {s["ckpt"]}')
        hm, uv = run_model(s['ckpt'], s['arch'], device, tensor)
        models.append(dict(label=s['label'], hm=hm, uv=uv))

    # Shared colour scale → honest baseline-vs-pruned comparison.
    vmax = max(float(m['hm'].max()) for m in models)
    print(f'Shared heatmap vmax = {vmax:.4f}')

    os.makedirs(args.out_dir, exist_ok=True)
    out_base = os.path.join(args.out_dir, 'fig15_heatmap_comparison')
    render(crop_rgb, models, vmax, args.cmap, out_base)

    # Also emit the npz plot_pruning_results.plot_heatmap_comparison() expects
    # (+ full 21-channel tensors), so figure15() works off this file too.
    npz_path = os.path.join(args.out_dir, 'fig15_heatmap_data.npz')
    np.savez_compressed(
        npz_path,
        image=crop_rgb,
        hmap_base=models[0]['hm'].max(axis=0),
        hmap_pruned=models[1]['hm'].max(axis=0),
        label_pruned=args.pruned_label,
        hm21_base=models[0]['hm'],
        hm21_pruned=models[1]['hm'],
    )
    print(f'Wrote {npz_path}')
    print(f"  (set HEATMAP_DATA = '{npz_path}' in plot_pruning_results.py to "
          f'reuse it in figure15())')


if __name__ == '__main__':
    main()
