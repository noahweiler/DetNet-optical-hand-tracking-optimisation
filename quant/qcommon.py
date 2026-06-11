"""
Shared helpers for the DetNet INT8 quantization study.

Every quant/ script imports this module first. It:
  - silences the Windows OMP duplicate-lib error (cv2 + numpy + torch),
  - puts the repo root on sys.path so `model.*` / `evaluate_detnet` import
    correctly even though these scripts live in quant/,
  - centralises the canonical paths (FP32 checkpoint, data root),
  - reuses the *tested* eval pipeline from evaluate_detnet.py so the quant
    numbers are produced by exactly the same code as the rest of the thesis.

Evaluation is FreiHAND-only by deliberate decision: RHD's eval split was the
fine-tuning validation set, so scoring it would be train/val leakage. RHD is
used only for calibration (calibration != evaluation).
"""
import os

# Must be set before cv2/numpy/torch are imported (OMP Error #15 on Windows).
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time

import numpy as np
import torch

# --- repo root on path (scripts live in <repo>/quant/) ----------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --- canonical paths --------------------------------------------------------
DATA_ROOT = "C:/Imperial/FYP Data"
FP32_CKPT = os.path.join(REPO_ROOT, "new_check_point", "ckp_detnet_71.pth")
QUANT_DIR = os.path.join(REPO_ROOT, "quant")

# Reuse the project's tested load + eval code.
from evaluate_detnet import (
    load_model, run_freihand, compute_summary_stats,
    training_crop_and_transform, project_3d_to_2d, uv_to_pixels,
    per_sample_metrics,
)


def load_fp32(device="cpu"):
    """Original 2D-only DetNet with the ep71 FP32 weights (eval mode)."""
    return load_model(FP32_CKPT, torch.device(device))


def _scalarize(stats):
    """compute_summary_stats(...) -> flat scalar dict for JSON / printing."""
    return {
        "n_frames": int(stats["n_frames"]),
        "mpjpe": float(stats["mpjpe"]),
        "auc_pck": float(stats["auc"]),
        "pck20": float(stats["pck_curve"][0]),
        "pck50": float(stats["pck_curve"][-1]),
    }


def eval_model(model, device="cpu", max_samples=999_999):
    """FreiHAND eval + train accuracy. Returns per-split + combined scalars."""
    model.eval()
    dev = torch.device(device)
    out, all_results = {}, []
    for split in ("eval", "train"):
        results = run_freihand(DATA_ROOT, model, dev, max_samples, split=split)
        out[f"freihand_{split}"] = _scalarize(compute_summary_stats(results))
        all_results += results
    out["freihand_all"] = _scalarize(compute_summary_stats(all_results))
    return out


def measure_latency(model, n=500, warmup=50, device="cpu"):
    """Synthetic-input CPU latency, ms/frame. Kept for backward compatibility
    with quant/method2.py and quant/method3.py (the standalone dev-time
    measurements that write quant/*_results.json — these do not feed the CSV).

    For the production INT8 numbers reported in eval_results_cpu/ and the CSV,
    use ``run_freihand_with_latency`` instead, which mirrors the FP32 protocol
    exactly (per-real-frame timing during the accuracy pass).

    Batch size 1, n=500 forwards on a single torch.randn dummy input, first
    50 discarded as warm-up. Returns (mean_ms, std_ms)."""
    model.eval()
    dummy = torch.randn(1, 3, 128, 128, device=torch.device(device))
    ts = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            model(dummy)
            t1 = time.perf_counter()
            ts.append((t1 - t0) * 1000.0)
    arr = np.asarray(ts[warmup:])
    return float(arr.mean()), float(arr.std())


def run_freihand_with_latency(model, data_root, device, max_samples,
                              split='eval'):
    """Single-pass FreiHAND accuracy + per-frame whole-forward latency.

    Mirrors ``evaluate_detnet.run_freihand`` byte-for-byte on the data path
    (frame iteration order, GT projection, 2× bbox-padded crop, bbox-norm
    metric) and adds a single ``time.perf_counter`` bracket around the
    model's forward call per frame. Returns per_sample_metrics dicts for
    accuracy AND a parallel list of per-frame model-forward times in
    milliseconds.

    This is the protocol-matched counterpart to the FP32 timing path: the
    FP32 path times the same forward call inline during its accuracy pass
    via evaluate_detnet's LatencyAccumulator (which sums four per-stage
    sub-timings; their total is what populates ``inference_ms``). The
    measurement here uses a single whole-forward bracket — semantically
    identical (sum-of-four-stage overhead is sub-microsecond, well below the
    per-frame jitter floor) and necessary for INT8 because PyTorch's
    quantized modules have FloatFunctional / QuantStub ops between the
    backbone and the head that the per-stage path would skip.

    Caller is responsible for combining timings across splits and applying
    the 10-frame warm-up trim that the FP32 LatencyAccumulator applies.

    Returns: (results, times_ms) where ``results`` is a list of
    ``per_sample_metrics`` dicts and ``times_ms`` is a list[float] of the
    same length, with per-frame whole-forward times in ms.
    """
    import json
    import glob
    import cv2

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw): return it

    root = os.path.join(data_root, 'FreiHAND')
    if split == 'train':
        img_dir  = os.path.join(root, 'training', 'rgb')
        xyz_file = os.path.join(root, 'training_xyz.json')
        K_file   = os.path.join(root, 'training_K.json')
        desc     = 'FreiHAND train (unique poses)'
    else:
        img_dir  = os.path.join(root, 'evaluation', 'rgb')
        xyz_file = os.path.join(root, 'evaluation_xyz.json')
        K_file   = os.path.join(root, 'evaluation_K.json')
        desc     = 'FreiHAND eval'

    with open(xyz_file) as f: xyz_list = json.load(f)
    with open(K_file)   as f: K_list   = json.load(f)

    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                       glob.glob(os.path.join(img_dir, '*.png')))
    n = min(max_samples, len(xyz_list), len(img_files))

    n_unique   = len(xyz_list)
    img_offset = n_unique if split == 'train' else 0

    model.eval()
    results  = []
    times_ms = []

    with torch.no_grad():
        for i in tqdm(range(n), desc=desc):
            img_index   = img_offset + i
            label_index = img_index % n_unique
            img = cv2.imread(img_files[img_index])
            if img is None:
                continue

            kp2d_gt = project_3d_to_2d(xyz_list[label_index],
                                       K_list[label_index])
            tensor, center, scale = training_crop_and_transform(
                img, kp2d_gt, device)

            # Single-bracket whole-forward timing — protocol-matched to the FP32
            # LatencyAccumulator total (sum of 4 per-stage brackets).
            t0 = time.perf_counter()
            result = model(tensor)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

            uv = result['uv'].squeeze(0).cpu().numpy().astype(np.float32)
            kp2d_pred = uv_to_pixels(uv, center, scale)

            # Same bbox-norm convention as evaluate_detnet.run_freihand (Simon
            # et al. 2017; scale here is 2× bbox side, so the GT side is
            # scale/2).
            norm = float(scale / 2.0)
            results.append(per_sample_metrics(kp2d_pred, kp2d_gt, norm))

    return results, times_ms


def state_dict_size_mb(model):
    """On-disk size of the model's state_dict, in MB (decimal megabytes)."""
    tmp = os.path.join(QUANT_DIR, "_tmp_size.pth")
    torch.save(model.state_dict(), tmp)
    mb = os.path.getsize(tmp) / (1000 * 1000)
    os.remove(tmp)
    return mb
