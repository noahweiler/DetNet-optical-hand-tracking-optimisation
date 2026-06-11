"""
visualise_qualitative.py — Fig 14: qualitative keypoint-overlay grid.

6 rows {GT, DetNet FP32, L1 55%, Taylor 55%, PTQ MinMax, PTQ Hist-MSE} x
4 FreiHAND frames, arranged as a 6x4 grid. Keypoints are drawn as white
dots connected by the SNAP_BONES skeleton (one colour per finger). Row 0
shows the GT-keypoint overlay; rows 1-5 show each model's predicted overlay
with its FreiHAND AUC in the left-column row label. Hardest frame flagged
in the top-column title.

The 4 frames are auto-chosen to span the difficulty range — easiest, ~33rd,
~66th percentile, and the WORST baseline frame (guarantees the spec's "≥1
challenging (occlusion / unusual pose)" image). Override with --indices.

PTQ models (QMM, QMSE) are re-quantized in-process from the FP32 fork
(deterministic — same as quant/07_quant_to_evalresults.py), since a converted
INT8 model can't be torch.save/load'd as a whole-model pickle.

Same crop + forward pipeline as evaluate_detnet.py, so the overlaid
predictions are exactly what the scored models produce. Also writes
figures/fig14_qualitative_data.npz with the expanded
(images / models / aucs / preds) contract — preds[:, 0] is the GT overlay.

Usage:
  python visualise_qualitative.py
  python visualise_qualitative.py --indices 0 12 480 1900
  python visualise_qualitative.py --l1_run ft_l1_70pct_rhdbest_both --taylor_run ft_taylor_70pct_rhdbest_both
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Windows cv2+torch OMP Error #15

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import argparse
import json

import cv2  # noqa: F401 — keep ABOVE torch (Windows DLL load order)
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from evaluate_detnet import (
    load_model, training_crop_and_transform, INPUT_SIZE, HEATMAP_SIZE,
)
from visualise_heatmaps import load_freihand_frame, crop_to_display
from plot_pruning_results import SNAP_BONES, COLORS, _auc_from_errors

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'DejaVu Serif'],
    'font.size':   9,
})

_UV2CROP = INPUT_SIZE / HEATMAP_SIZE          # heatmap (32) -> crop (128) px
_FINGER_COLORS = [COLORS['red'], COLORS['blue'], COLORS['green'],
                  COLORS['orange'], COLORS['purple']]

# Fig 14 layout (6 rows x 4 cols). AUCs are hardcoded bbox-norm FreiHAND
# eval+train numbers from eval_results_MPJPE/ — keep these in sync with that
# dir if re-running. Two decimals to match the summary table / CSV precision.
#
# Row-label AUC vs per-cell MPJPE distinction (thesis caption must spell this
# out, otherwise readers conflate them): row-label AUC is the DATASET-LEVEL
# integrated PCK curve over the full FreiHAND test set. Per-cell MPJPE is a
# PER-FRAME bbox-normalised mean joint error for THAT individual image —
# different quantities, not commensurable.
ROW_LABELS = [
    "Ground Truth",
    "DetNet FP32\nAUC 68.47%",
    "L1 – 55%\nAUC 66.76%",
    "Taylor – 55%\nAUC 66.52%",
    "PTQ MinMax\nAUC 62.17%",
    "PTQ Hist-MSE\nAUC 67.51%",
]
COL_TITLES = ["Open hand", "Side view", "Bent fingers", "Partly hidden"]
# Skeleton colour key — must match _FINGER_COLORS order. SNAP_BONES grouping:
# thumb (b 0-3), index (4-7), middle (8-11), ring (12-15), pinky (16-19).
FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']


# ── Model discovery from eval_results ────────────────────────────────────────

def _pooled_freihand_auc(run_dir, tau_min=0, tau_max=30):
    """FreiHAND eval∪train AUC pooled from errors.npz — identical scheme to
    build_results_csv.py / plot_pruning_results.py (tau=[0,30] px)."""
    npz = os.path.join(run_dir, 'errors.npz')
    if not os.path.isfile(npz):
        return None
    with np.load(npz) as z:
        parts = [z[k] for k in z.files if k.startswith('freihand')]
    if not parts:
        return None
    return _auc_from_errors(np.concatenate(parts), tau_min, tau_max)


def _spec_from_run(results_dir, run_name, label):
    """Read a run's summary.json → (label, checkpoint, arch, auc)."""
    run_dir = os.path.join(results_dir, run_name)
    with open(os.path.join(run_dir, 'summary.json'), encoding='utf-8') as f:
        s = json.load(f)
    return {
        'label': label,
        'ckpt':  s['checkpoint'],
        'arch':  s.get('arch'),
        'auc':   _pooled_freihand_auc(run_dir),
    }


def discover_fp32_models(results_dir, args):
    """baseline + L1 55% + Taylor 55% (CLI overrides win).

    Fig 14 columns are pinned to specific ratios; no "best AUC" auto-pick.
    """
    base_run   = args.baseline_run or 'ckp_detnet_71_both'
    l1_run     = args.l1_run       or 'ft_l1_55pct_rhdbest_both'
    taylor_run = args.taylor_run   or 'ft_taylor_55pct_rhdbest_both'
    for nm, r in [('baseline', base_run), ('L1 55%', l1_run),
                  ('Taylor 55%', taylor_run)]:
        if not os.path.isdir(os.path.join(results_dir, r)):
            sys.exit(f'No {nm} run dir {r!r} in {results_dir}.')
    return [
        _spec_from_run(results_dir, base_run,   'DetNet FP32'),
        _spec_from_run(results_dir, l1_run,     'L1 – 55%'),
        _spec_from_run(results_dir, taylor_run, 'Taylor – 55%'),
    ]


def load_quant_models():
    """Re-quantize MinMax (M2) + Hist-MSE (M3) PTQ models from the FP32 fork.

    Deterministic — same pipeline as quant/07_quant_to_evalresults.py. Returns
    eval-mode INT8 models running on the FBGEMM/x86 CPU backend.
    """
    quant_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quant')
    if quant_dir not in sys.path:
        sys.path.insert(0, quant_dir)
    import qquant

    print('  re-quantizing PTQ MinMax (M2) ...', flush=True)
    qmm, _ = qquant.quantize(qquant.M2_QCONFIG)
    qmm.eval()
    print('  re-quantizing PTQ Hist-MSE (M3) ...', flush=True)
    qmse, _ = qquant.quantize(qquant.M3_QCONFIG)
    qmse.eval()
    return qmm, qmse


# ── Forward / geometry ───────────────────────────────────────────────────────

def predict_kp_crop(model, tensor):
    """Model → predicted joints in 128×128 crop pixels, (21,2) [x,y]."""
    with torch.no_grad():
        uv = model(tensor)['uv'].squeeze(0).cpu().numpy().astype(np.float32)
    # uv is heatmap-space (row, col); evaluate_detnet maps col→x, row→y.
    return np.stack([uv[:, 1] * _UV2CROP, uv[:, 0] * _UV2CROP], axis=1)


def gt_in_crop(kp2d_gt, center, scale):
    """Forward crop transform: original-image GT → 128×128 crop pixels."""
    x = (kp2d_gt[:, 0] - (center[0] - scale / 2)) * INPUT_SIZE / scale
    y = (kp2d_gt[:, 1] - (center[1] - scale / 2)) * INPUT_SIZE / scale
    return np.stack([x, y], axis=1)


def pick_indices(baseline, data_root, split, device, scan_pool):
    """Scan the first `scan_pool` frames with the baseline; return 4 indices
    spanning the error range: easiest, ~33%, ~66%, hardest (the challenging
    occlusion/unusual-pose frame the spec requires)."""
    errs = []
    for i in range(scan_pool):
        try:
            img, gt, _ = load_freihand_frame(data_root, split, i)
        except (IndexError, FileNotFoundError):
            break
        tensor, c, s = training_crop_and_transform(img, gt, device)
        pred = predict_kp_crop(baseline, tensor)
        e = float(np.linalg.norm(pred - gt_in_crop(gt, c, s), axis=1).mean())
        errs.append((e, i))
        if (i + 1) % 50 == 0:
            print(f'  difficulty scan: {i + 1}/{scan_pool}', flush=True)
    errs.sort()
    n = len(errs)
    chosen = [errs[0][1], errs[n // 3][1], errs[2 * n // 3][1], errs[-1][1]]
    print(f'Auto-selected frames (easy→hard): {chosen}  '
          f'(baseline mean-err {errs[0][0]:.1f}→{errs[-1][0]:.1f}px)')
    return chosen, len(chosen) - 1   # hardest frame is the last grid column


# ── Figure ───────────────────────────────────────────────────────────────────

def render(images, preds, out_base):
    """6x4 grid. images:(4,128,128,3); preds:(4,6,21,2)[x,y].

    Row order (ROW_LABELS): GT, DetNet FP32, L1 55%, Taylor 55%, PTQ MinMax,
    PTQ Hist-MSE. Column order (COL_TITLES): Open hand, Side view,
    Bent fingers, Partly hidden. preds[frame, model] feeds axes[model, frame]
    (transpose of the data layout). Column titles go on the top row only;
    row labels on the left column only.

    Each non-GT cell is annotated with that frame's bbox-normalised MPJPE
    (white text in a semi-transparent black rounded box, bottom-left) so
    the per-frame error is quantitative, not just visually inferred from
    the keypoint overlay. The GT row is unlabelled (MPJPE would be 0).
    A skeleton colour key is drawn at the bottom of the figure.
    """
    n_frames, n_models = len(images), len(ROW_LABELS)
    assert preds.shape[1] == n_models, \
        f'preds has {preds.shape[1]} model slots, expected {n_models}'
    assert preds.shape[0] == n_frames, \
        f'preds has {preds.shape[0]} frame slots, expected {n_frames}'

    # Manual margins (not constrained_layout) so the small inter-cell gap
    # from gridspec_kw is honoured exactly. Left margin holds the horizontal
    # row labels; bottom margin reserves room for the skeleton colour key.
    fig, axes = plt.subplots(n_models, n_frames, figsize=(16, 22),
                             gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
    fig.subplots_adjust(left=0.17, right=0.98, top=0.96, bottom=0.05)

    mpjpe_grid = np.full((n_models, n_frames), np.nan, dtype=float)
    for m in range(n_models):
        for f in range(n_frames):
            ax = axes[m, f]
            ax.imshow(images[f])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():       # thin dark-grey cell border
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor('#333333')
            kp = preds[f, m]
            for b, (p, q) in enumerate(SNAP_BONES):
                ax.plot([kp[p, 0], kp[q, 0]], [kp[p, 1], kp[q, 1]], '-',
                        color=_FINGER_COLORS[b // 4], linewidth=1.6, zorder=2)
            ax.scatter(kp[:, 0], kp[:, 1], s=18, c='white',
                       edgecolor='k', linewidth=0.6, zorder=3)
            if m == 0:
                ax.set_title(COL_TITLES[f], fontsize=22, fontweight='bold',
                             pad=8)
            if f == 0:
                ax.set_ylabel(ROW_LABELS[m], fontsize=18, fontweight='bold',
                              rotation=0, ha='right', va='center', labelpad=12)
            # Per-frame MPJPE annotation (skip the GT row, which would be 0).
            # bbox-normalised: max(bbox_w, bbox_h) of the GT keypoints in crop
            # pixels — same convention as evaluate_detnet.per_sample_metrics
            # so the per-cell number is in the same units as the row-label AUC
            # tau axis.
            if m > 0:
                gt = preds[f, 0]
                bbox_w = float(gt[:, 0].max() - gt[:, 0].min())
                bbox_h = float(gt[:, 1].max() - gt[:, 1].min())
                norm   = max(bbox_w, bbox_h)
                mpjpe  = float(np.linalg.norm(kp - gt, axis=1).mean()) / norm
                mpjpe_grid[m, f] = mpjpe
                ax.text(0.03, 0.03, f'MPJPE {mpjpe:.2f}',
                        transform=ax.transAxes,
                        fontsize=9, color='white',
                        ha='left', va='bottom', zorder=4,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='black', alpha=0.6,
                                  edgecolor='none'))

    # Verification dump: print per-cell bbox-norm MPJPE so the numbers in
    # the figure annotations can be eyeballed against the console before
    # committing the PDF.
    print('Per-cell bbox-norm MPJPE (matches figure annotations, GT row omitted):')
    col_hdr = ' ' * 24 + ''.join(f'{c:>16}' for c in COL_TITLES)
    print(col_hdr)
    for mi in range(1, n_models):
        row_name = ROW_LABELS[mi].split('\n')[0]
        cells = ''.join(f'{mpjpe_grid[mi, fi]:>16.2f}' for fi in range(n_frames))
        print(f'{row_name:<24}{cells}')

    # Skeleton colour key — five short coloured line segments, one per
    # finger, drawn as a figure-level legend at the bottom margin.
    key_handles = [
        Line2D([0], [0], color=col, linewidth=4, label=name)
        for col, name in zip(_FINGER_COLORS, FINGER_NAMES)
    ]
    fig.legend(handles=key_handles, loc='lower center', ncol=5,
               fontsize=14, frameon=False,
               bbox_to_anchor=(0.5, 0.012),
               handlelength=2.2, columnspacing=2.5,
               title='Skeleton', title_fontsize=14)

    # Thin rectangular border around the entire figure grid. Left edge pulled
    # in (0.04 instead of 0.01) so it sits just outside the row labels, not
    # half a column away from them.
    border = FancyBboxPatch(
        (0.04, 0.01), 0.95, 0.98,
        boxstyle="square,pad=0",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor="none",
        transform=fig.transFigure,
        zorder=10,
        clip_on=False,
    )
    fig.add_artist(border)

    png, pdf = out_base + '.png', out_base + '.pdf'
    fig.savefig(png, dpi=1000, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(pdf, dpi=1000, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Wrote {png}')
    print(f'Wrote {pdf}')


def render_v2(images, preds, out_base):
    """V2 styling for the qualitative overlay grid (additive — does not
    modify or replace render()):
      * MPJPE annotation in opaque white rounded box, black bold text.
      * Larger MPJPE font (14pt) for legibility at thesis \\textwidth.
      * Per-cell corner placement: annotation is placed at whichever of
        BL/TR/TL/BR has the fewest keypoints inside the planned box, so
        the box doesn't sit on top of the hand.
      * Enlarged skeleton colour key at the bottom; the 'Skeleton' title
        is dropped, leaving the five finger labels on their own.
    Produces fig_overlay_mpjpe_v2.{pdf,png} for side-by-side comparison
    with the v1 render.
    """
    n_frames, n_models = len(images), len(ROW_LABELS)
    assert preds.shape[1] == n_models
    assert preds.shape[0] == n_frames
    H, W = images.shape[1], images.shape[2]

    fig, axes = plt.subplots(n_models, n_frames, figsize=(16, 22),
                             gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
    fig.subplots_adjust(left=0.17, right=0.98, top=0.96, bottom=0.06)

    # Estimated annotation box footprint in axes coords for the corner
    # picker. Sized for fontsize=14 bold + boxstyle='round,pad=0.45'.
    BW, BH = 0.42, 0.13
    corner_specs = [
        (0.03, 0.03, 'left',  'bottom'),  # BL
        (0.97, 0.97, 'right', 'top'),     # TR
        (0.03, 0.97, 'left',  'top'),     # TL
        (0.97, 0.03, 'right', 'bottom'),  # BR
    ]

    def pick_corner(kp_axes):
        best_anchor, best_count = None, None
        for x_a, y_a, ha, va in corner_specs:
            x0 = x_a if ha == 'left' else x_a - BW
            y0 = y_a if va == 'bottom' else y_a - BH
            in_box = ((kp_axes[:, 0] >= x0) & (kp_axes[:, 0] <= x0 + BW) &
                      (kp_axes[:, 1] >= y0) & (kp_axes[:, 1] <= y0 + BH))
            n_inside = int(in_box.sum())
            if best_count is None or n_inside < best_count:
                best_anchor, best_count = (x_a, y_a, ha, va), n_inside
            if n_inside == 0:
                break
        return best_anchor

    mpjpe_grid = np.full((n_models, n_frames), np.nan, dtype=float)
    for m in range(n_models):
        for f in range(n_frames):
            ax = axes[m, f]
            ax.imshow(images[f])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor('#333333')
            kp = preds[f, m]
            for b, (p, q) in enumerate(SNAP_BONES):
                ax.plot([kp[p, 0], kp[q, 0]], [kp[p, 1], kp[q, 1]], '-',
                        color=_FINGER_COLORS[b // 4], linewidth=1.6, zorder=2)
            ax.scatter(kp[:, 0], kp[:, 1], s=18, c='white',
                       edgecolor='k', linewidth=0.6, zorder=3)
            if m == 0:
                ax.set_title(COL_TITLES[f], fontsize=22, fontweight='bold',
                             pad=8)
            if f == 0:
                ax.set_ylabel(ROW_LABELS[m], fontsize=18, fontweight='bold',
                              rotation=0, ha='right', va='center', labelpad=12)
            if m > 0:
                gt = preds[f, 0]
                bbox_w = float(gt[:, 0].max() - gt[:, 0].min())
                bbox_h = float(gt[:, 1].max() - gt[:, 1].min())
                norm   = max(bbox_w, bbox_h)
                mpjpe  = float(np.linalg.norm(kp - gt, axis=1).mean()) / norm
                mpjpe_grid[m, f] = mpjpe
                # imshow uses origin='upper' (y inverted) — flip y so
                # axes-coord keypoints match the on-screen layout.
                kp_all = np.vstack([gt, kp])
                kp_axes = np.column_stack([
                    kp_all[:, 0] / W,
                    1.0 - kp_all[:, 1] / H,
                ])
                x_a, y_a, ha, va = pick_corner(kp_axes)
                ax.text(x_a, y_a, f'MPJPE {mpjpe:.2f}',
                        transform=ax.transAxes,
                        fontsize=14, color='black', fontweight='bold',
                        ha=ha, va=va, zorder=4,
                        bbox=dict(boxstyle='round,pad=0.45',
                                  facecolor='white', alpha=1.0,
                                  edgecolor='black', linewidth=0.8))

    print('Per-cell bbox-norm MPJPE (V2 render, GT row omitted):')
    col_hdr = ' ' * 24 + ''.join(f'{c:>16}' for c in COL_TITLES)
    print(col_hdr)
    for mi in range(1, n_models):
        row_name = ROW_LABELS[mi].split('\n')[0]
        cells = ''.join(f'{mpjpe_grid[mi, fi]:>16.2f}' for fi in range(n_frames))
        print(f'{row_name:<24}{cells}')

    # Skeleton colour key, enlarged and with the 'Skeleton' title dropped.
    key_handles = [
        Line2D([0], [0], color=col, linewidth=5, label=name)
        for col, name in zip(_FINGER_COLORS, FINGER_NAMES)
    ]
    fig.legend(handles=key_handles, loc='lower center', ncol=5,
               fontsize=20, frameon=False,
               bbox_to_anchor=(0.5, 0.012),
               handlelength=2.4, columnspacing=2.8)

    border = FancyBboxPatch(
        (0.04, 0.01), 0.95, 0.98,
        boxstyle="square,pad=0",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor="none",
        transform=fig.transFigure,
        zorder=10,
        clip_on=False,
    )
    fig.add_artist(border)

    png, pdf = out_base + '.png', out_base + '.pdf'
    fig.savefig(png, dpi=1000, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(pdf, dpi=1000, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Wrote {png}')
    print(f'Wrote {pdf}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Fig 14 — qualitative keypoint-overlay grid')
    ap.add_argument('--results_dir', default='eval_results')
    ap.add_argument('--data_root', default='C:/Imperial/FYP Data')
    ap.add_argument('--split', choices=['eval', 'train'], default='eval')
    ap.add_argument('--indices', type=int, nargs=4, default=None,
                    help='4 explicit FreiHAND indices (skips the difficulty scan)')
    ap.add_argument('--scan_pool', type=int, default=150,
                    help='# frames scanned to auto-pick the 4 (default: 150)')
    ap.add_argument('--baseline_run', default=None,
                    help='override baseline run dir (default: ckp_detnet_71_both)')
    ap.add_argument('--l1_run',       default=None,
                    help='override L1 run dir (default: ft_l1_55pct_rhdbest_both)')
    ap.add_argument('--taylor_run',   default=None,
                    help='override Taylor run dir (default: ft_taylor_55pct_rhdbest_both)')
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
    ap.add_argument('--out_dir', default='figures')
    args = ap.parse_args()

    device = torch.device('cuda:0' if (args.device == 'cuda'
                                       and torch.cuda.is_available()) else 'cpu')
    print(f'Device : {device}')

    fp32_specs = discover_fp32_models(args.results_dir, args)
    for s in fp32_specs:
        a = f"{s['auc']:.2f}%" if s['auc'] is not None else 'n/a'
        print(f"  {s['label']:<22} AUC={a:<7} ckpt={s['ckpt']}")
    fp32_models = [load_model(s['ckpt'], device, arch_path=s['arch'])
                   for s in fp32_specs]

    print('Loading PTQ INT8 models (deterministic re-quantization):')
    qmm, qmse = load_quant_models()

    # Choose the 4 frames using the FP32 baseline (models[0]).
    if args.indices:
        indices = list(args.indices)
        print(f'Using explicit frames: {indices}')
    else:
        indices, _ = pick_indices(fp32_models[0], args.data_root,
                                  args.split, device, args.scan_pool)

    # Model slots after GT (5 models, in figure row order).
    fp32_models_in_order = fp32_models                 # baseline, L1, Taylor
    quant_models_in_order = [qmm, qmse]                # PTQ MinMax, PTQ Hist-MSE

    n_frames, n_models = len(indices), len(ROW_LABELS)
    images = np.zeros((n_frames, INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
    preds  = np.zeros((n_frames, n_models, 21, 2), np.float32)
    for r, idx in enumerate(indices):
        img, gt, _ = load_freihand_frame(args.data_root, args.split, idx)
        tensor, c, s = training_crop_and_transform(img, gt, device)
        images[r] = crop_to_display(tensor)

        preds[r, 0] = gt_in_crop(gt, c, s)             # row 0: GT

        slot = 1
        for m in fp32_models_in_order:                 # rows 1-3
            preds[r, slot] = predict_kp_crop(m, tensor)
            slot += 1

        tensor_cpu = tensor.cpu()                      # PTQ models are CPU-only
        for m in quant_models_in_order:                # rows 4-5
            preds[r, slot] = predict_kp_crop(m, tensor_cpu)
            slot += 1

    os.makedirs(args.out_dir, exist_ok=True)
    out_base = os.path.join(args.out_dir, 'fig14_qualitative_overlay')
    render(images, preds, out_base)
    out_base_v2 = os.path.join(args.out_dir, 'fig_overlay_mpjpe_v2')
    render_v2(images, preds, out_base_v2)

    # Expanded npz contract: 6 model slots, slot 0 is GT (AUC = nan).
    model_labels = ['Ground Truth', 'DetNet FP32', 'L1 – 55%',
                    'Taylor – 55%', 'PTQ MinMax', 'PTQ Hist-MSE']
    model_aucs = np.array([np.nan, 68.5, 66.8, 66.5, 62.2, 67.5], dtype=float)
    npz_path = os.path.join(args.out_dir, 'fig14_qualitative_data.npz')
    np.savez_compressed(
        npz_path,
        images=images,
        models=np.array(model_labels),
        aucs=model_aucs,
        preds=preds,
    )
    print(f'Wrote {npz_path}')


if __name__ == '__main__':
    main()
