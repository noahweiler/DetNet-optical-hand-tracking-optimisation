"""
Score the 5 original DetNet checkpoints on the RHD test set using the
canonical PCK-AUC range τ ∈ [0, 30] px.

This is a tau-range-corrected redo of
``Setting_up_fine_tuning/find_best_checkpoint.py`` restricted to the RHD test
split only, to verify whether epoch 71 remains the best-performing checkpoint
under the current convention (canonical τ range now [0, 30] px — see memory
note ``feedback_auc_tau_range``).

Methodology
-----------
For each checkpoint:
  1. Iterate every RHD-test sample via the project's own ``RhdDataset`` loader
     (located in ``Old_training/datasets/rhd.py``). This yields one PIL image
     and one set of GT 2D keypoints per frame, both in original-image
     coordinates. Left-hand frames are mirror-flipped to right-hand orientation
     by the loader itself (``hand_side='right'``), matching the convention
     DetNet was trained under.
  2. Apply ``evaluate_detnet.training_crop_and_transform`` to crop a
     128×128 region centred on the GT-keypoint bounding box (scale = max
     bbox side × 2), identical to the FreiHAND evaluation protocol.
  3. Forward through the model.
  4. Map heatmap-space ``uv`` predictions back to original image coordinates
     via ``evaluate_detnet.uv_to_pixels``.
  5. Compute per-joint Euclidean errors in original image pixels.
  6. Aggregate via ``evaluate_detnet.compute_summary_stats`` with
     ``tau_min=0`` and ``tau_max=30``.

The errors are in original-image pixels and the τ range matches the
canonical convention used throughout the rest of the thesis (Section / Table X
of the main text), so the AUC values reported here are directly comparable
to the FreiHAND AUC values in ``eval_results_cpu/*_both/`` modulo dataset-
distribution differences (RHD hands tend to be smaller in pixel terms than
FreiHAND hands, so absolute AUC will be lower on RHD even for an equally
accurate model).

The ``my_scale`` convention in the loader and the bbox-norm MPJPE convention
in the metric path are both inherited unchanged from ``evaluate_detnet.py``;
this script adds no new convention.

Usage
-----
    python Setting_up_fine_tuning/score_checkpoints_rhd_tau030.py

Output is a table of (checkpoint, AUC, MPJPE_bbox_norm, n_frames). The best
checkpoint by AUC is printed at the end.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Windows cv2+torch OMP Error #15

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')   # tee/redirect-safe on Windows
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
# Make repo root and the old_training/ package both importable
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_OLD_TRAINING = os.path.join(_REPO, 'old_training')
if _OLD_TRAINING not in sys.path:
    sys.path.insert(0, _OLD_TRAINING)

import argparse
import time

import cv2  # noqa: F401 — keep above torch (Windows DLL load order)
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:                # tqdm is optional; degrade gracefully
    def tqdm(iterable, **kwargs):
        return iterable

# Project imports (read-only — no edits required to existing files)
from datasets.rhd import RHDDataset
from evaluate_detnet import (
    load_model,
    training_crop_and_transform,
    uv_to_pixels,
    predict,
    per_sample_metrics,
    compute_summary_stats,
)

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = os.path.join(_REPO, 'new_check_point')
CHECKPOINTS = [
    ('ep68',  'ckp_detnet_68.pth'),
    ('ep71',  'ckp_detnet_71.pth'),
    ('ep83',  'ckp_detnet_83.pth'),
    ('ep101', 'ckp_detnet_101.pth'),
    ('ep106', 'ckp_detnet_106.pth'),
]

DEFAULT_DATA_ROOT = 'C:/Imperial/FYP Data'   # same as the rest of the project
DEFAULT_TAU_MIN   = 0                        # canonical bbox-norm tau range
DEFAULT_TAU_MAX   = 30
DEFAULT_DEVICE    = 'cuda'                   # falls back to CPU if unavailable


# ── Evaluation ────────────────────────────────────────────────────────────────

def _to_bgr_numpy(pil_rgb):
    """PIL RGB image -> BGR numpy array, the format training_crop_and_transform
    expects (it then converts back to RGB internally — this dance preserves the
    exact byte order evaluate_detnet uses for FreiHAND)."""
    rgb = np.array(pil_rgb)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def evaluate_on_rhd(model, device, data_root, max_samples=None,
                    show_progress=True):
    """Iterate RHD test, return list of per_sample_metrics dicts (one per frame).

    Each dict has keys 'errors_px', 'norm_px', 'mpjpe' (see
    evaluate_detnet.per_sample_metrics for the precise semantics — errors in
    original image pixels, norm in original image pixels = max GT bbox side).
    """
    ds = RHDDataset(
        data_split='test',
        hand_side='right',            # mirror left -> right (training convention)
        data_root=data_root,
    )
    n = len(ds)
    if max_samples is not None:
        n = min(n, max_samples)

    results = []
    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, desc='  RHD test', unit='img')

    model.eval()
    for i in iterator:
        sample = ds.get_sample(i)
        img_bgr = _to_bgr_numpy(sample['clr'])
        kp2d_gt = np.asarray(sample['kp2d'], dtype=np.float32)   # (21, 2) in original image px

        # Identical pre-processing path to evaluate_detnet.run_freihand: crop
        # 2×-padded square around the GT-keypoint bbox, resample to 128×128.
        tensor, center, scale = training_crop_and_transform(img_bgr, kp2d_gt, device)

        # Forward + map heatmap UV back to original image pixels
        with torch.no_grad():
            uv, _conf = predict(model, tensor)        # (21, 2) heatmap (row, col)
        kp2d_pred = uv_to_pixels(uv, center, scale)   # (21, 2) original image px

        # bbox-norm convention: norm = scale/2 (tight GT bbox side) — verbatim from
        # evaluate_detnet.run_freihand:678-682. Keeps AUC math and MPJPE math
        # byte-identical to the FreiHAND path.
        norm = float(scale / 2.0)
        results.append(per_sample_metrics(kp2d_pred, kp2d_gt, norm))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Score original DetNet checkpoints on RHD test, '
                    'AUC PCK over τ ∈ [0, 30] px.',
    )
    parser.add_argument('--data_root',  type=str, default=DEFAULT_DATA_ROOT,
                        help='Root folder containing the RHD dataset folder')
    parser.add_argument('--tau_min',    type=int, default=DEFAULT_TAU_MIN)
    parser.add_argument('--tau_max',    type=int, default=DEFAULT_TAU_MAX)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Cap frames per checkpoint (default: full RHD test split)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=DEFAULT_DEVICE)
    args = parser.parse_args()

    device = torch.device('cuda:0' if (args.device == 'cuda'
                                       and torch.cuda.is_available()) else 'cpu')
    print(f'Device          : {device}')
    print(f'Data root       : {args.data_root}')
    print(f'PCK τ range     : [{args.tau_min}, {args.tau_max}] px (original-image scale)')
    print(f'Convention      : bbox-norm MPJPE (norm = scale/2), AUC on raw px errors')
    print(f'Checkpoints     : {len(CHECKPOINTS)} candidates')
    if args.max_samples is not None:
        print(f'Sample cap      : {args.max_samples} per checkpoint (smoke mode)')
    print()

    results_table = []   # list of (tag, auc, mpjpe, n_frames, latency_s)
    for tag, ckpt_file in CHECKPOINTS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
        if not os.path.isfile(ckpt_path):
            print(f'!! {tag} ({ckpt_path}) not found — skipping\n')
            results_table.append((tag, None, None, 0, 0.0))
            continue

        print(f'═══ {tag} ({ckpt_path}) ═══')
        t0 = time.perf_counter()
        model = load_model(ckpt_path, device)
        per_frame = evaluate_on_rhd(model, device, args.data_root,
                                    max_samples=args.max_samples)
        stats = compute_summary_stats(per_frame,
                                       tau_min=args.tau_min,
                                       tau_max=args.tau_max)
        elapsed = time.perf_counter() - t0

        if stats is None:
            print(f'    no results — skipping\n')
            results_table.append((tag, None, None, 0, elapsed))
            continue

        auc      = stats['auc']
        mpjpe    = stats['mpjpe']
        n_frames = stats['n_frames']
        results_table.append((tag, auc, mpjpe, n_frames, elapsed))
        print(f'    AUC  PCK({args.tau_min}-{args.tau_max}px) = {auc:6.2f} %')
        print(f'    MPJPE (bbox-norm)        = {mpjpe:.4f}')
        print(f'    Frames                   = {n_frames:,}')
        print(f'    Elapsed                  = {elapsed:.1f} s\n')

    # ── Final table ───────────────────────────────────────────────────────────
    print('=' * 70)
    print(f'  RHD test set  ·  PCK τ ∈ [{args.tau_min}, {args.tau_max}] px')
    print('=' * 70)
    print(f'  {"Checkpoint":<10} {"AUC (%)":>10} {"MPJPE":>10} '
          f'{"n_frames":>10} {"Time (s)":>10}')
    print(f'  {"-"*10:<10} {"-"*10:>10} {"-"*10:>10} {"-"*10:>10} {"-"*10:>10}')

    scored = [r for r in results_table if r[1] is not None]
    for tag, auc, mpjpe, n, elapsed in results_table:
        if auc is None:
            print(f'  {tag:<10} {"-":>10} {"-":>10} {"-":>10} {elapsed:>10.1f}')
            continue
        print(f'  {tag:<10} {auc:>9.2f}% {mpjpe:>10.4f} {n:>10,} {elapsed:>10.1f}')

    print('=' * 70)
    if scored:
        best = max(scored, key=lambda r: r[1])
        print(f'  Best by AUC: {best[0]}  ({best[1]:.2f} %)')
    else:
        print('  No checkpoints produced results.')
    print()


if __name__ == '__main__':
    main()
