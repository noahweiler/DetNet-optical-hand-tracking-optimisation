"""
Evaluate DetNet 2D hand pose estimation on FreiHAND.

RHD is intentionally NOT an evaluation target: its eval split was used as the
validation set during fine-tuning (finetune_pruned.py --datasets_test rhd →
*_rhdbest.pth), so scoring it here would be train/validation leakage. FreiHAND
was never seen in training, so the whole FreiHAND set (eval + train) is the
only legitimate held-out benchmark.

Metrics (normalised by image size):
  MPJPE  — Mean Per Joint Position Error          (eq. 7-8 in report)
  PCK@τ  — Percentage of Correct Keypoints        (eq. 9-10)

Evaluation defaults to the whole FreiHAND set (eval + train); no --dataset
flag is needed. It runs on CPU by default (the deployment/measurement target —
accuracy is device-independent and the reported latency is the meaningful CPU
number); pass --device cuda for a GPU run.

Usage:
  # Unpruned (original) DetNet
  python evaluate_detnet.py --checkpoint new_check_point/ckp_detnet_83.pth

  # Pruned + fine-tuned: pass the fine-tuned state_dict AND the matching _noft architecture
  python evaluate_detnet.py \\
      --checkpoint path/to/finetuned_state_dict.pth \\
      --arch pruned_architectures/detnet_l1_25pct_ep71_noft.pth
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Windows + cv2/numpy/torch: silence OMP Error #15

import argparse
import glob
import json
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model.detnet import detnet
from model.detnet.detnet import get_pose_tile_torch as _get_pos_tile
from utils import func, handutils
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

HEATMAP_SIZE = 32   # DetNet output heatmap resolution (32×32)
INPUT_SIZE   = 128  # DetNet input image size (128×128)


# ── Performance profiling helpers ─────────────────────────────────────────────

def _sync():
    """CUDA barrier — ensures GPU work finishes before reading the clock."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def predict_timed(model, tensor):   # Runs the DetNet forward pass with timing checkpoints between each stage, returns (uv, conf, timings)
    """
    Run DetNet stage-by-stage and measure wall-clock time for each stage.
    This manually replicates detnet.forward() so timing checkpoints can be
    inserted between each sub-module call.

    Stages timed (2D-only path):
      feature_extractor : ResNet50 backbone + positional encoding concat
      heatmap           : hmap_0 — produces 21 2D heatmaps
      uv_extraction     : argmax over heatmaps → (u, v) per joint

    Returns:
      uv       : (21, 2)  heatmap-space joint coordinates, for uv_to_pixels
      conf     : float    mean peak heatmap confidence across all joints
      timings  : dict     stage name → elapsed milliseconds
    """
    with torch.no_grad():
        # Stage 1: ResNet50 feature extraction + positional grid encoding
        _sync(); t0 = time.perf_counter()
        features = model.resnet50(tensor) # (B, 256, 32, 32)
        pos_tile = _get_pos_tile(features.shape[0]).to(tensor.device) # (B, 2, 32, 32) - ADDS posistional encoding to the feature map
        x = torch.cat([features, pos_tile], dim=1)   # (B, 258, 32, 32)
        _sync(); t1 = time.perf_counter()

        # Stage 2: Heatmap head — predicts a 32×32 confidence map per joint
        heat_map = model.hmap_0(x)   # (B, 21, 32, 32)
        _sync(); t2 = time.perf_counter()

        # Stage 3: Extract joint locations — argmax over heatmap
        uv_t, _ = model.map_to_uv(heat_map)          # (B, 21, 2)
        _sync(); t3 = time.perf_counter()

        # Convert tensors to numpy (drop batch dimension)
        uv   = uv_t.squeeze(0).cpu().numpy().astype(np.float32)
        # Confidence = mean of per-joint peak heatmap values
        conf = heat_map.squeeze(0).view(21, -1).max(dim=1).values.mean().item()

    ms = lambda a, b: (b - a) * 1000.0   # convert seconds to milliseconds
    return uv, conf, {
        'feature_extractor': ms(t0, t1),
        'heatmap':           ms(t1, t2),
        'uv_extraction':     ms(t2, t3),
    }


class LatencyAccumulator:
    """
    Collects per-frame stage timings (ms) over an entire evaluation run, then
    reports mean ± std on demand. Per-frame totals are tracked separately so
    the total-latency std is the std of (frame-sum), not the sum of stage stds
    (which would assume independence the stages don't have).

    The first `_WARMUP` frames are excluded from mean/std to skip CUDA kernel
    JIT compilation and other one-time setup costs that would inflate variance.
    """
    _STAGES = ['feature_extractor', 'heatmap', 'uv_extraction', 'uv_to_pixels']
    _WARMUP = 10   # frames discarded before computing stats — covers CUDA JIT warm-up

    def __init__(self):
        # Each stage maps to a list of per-frame timings (ms)
        self._data   = {s: [] for s in self._STAGES}
        self._totals = []   # per-frame total (sum of all stage times) — for true total std

    def update(self, timings):
        """Append one frame's timings dict to the running lists."""
        frame_total = 0.0
        for k, v in timings.items():
            if k in self._data:
                self._data[k].append(v)
                frame_total += v
        if frame_total > 0:
            self._totals.append(frame_total)

    def _trim(self, lst):
        """Drop warm-up samples; if there aren't enough samples, keep all."""
        return lst[self._WARMUP:] if len(lst) > self._WARMUP else lst

    def means(self):
        """Per-stage mean ms (post-warmup). Zero if no data yet."""
        return {k: (float(np.mean(self._trim(v))) if self._trim(v) else 0.0)
                for k, v in self._data.items()}

    def stds(self):
        """Per-stage standard deviation in ms (post-warmup). Zero with <2 samples."""
        return {k: (float(np.std(self._trim(v))) if len(self._trim(v)) > 1 else 0.0)
                for k, v in self._data.items()}

    def total_mean_std(self):
        """Mean ± std of the per-frame TOTAL latency (sum of stages)."""
        totals = self._trim(self._totals)
        if not totals:
            return 0.0, 0.0
        if len(totals) < 2:
            return float(totals[0]), 0.0
        return float(np.mean(totals)), float(np.std(totals))

    def count(self):
        """Total frames recorded (including warm-up)."""
        return len(self._totals)

    def count_after_warmup(self):
        """Frames actually used for mean/std (post-warm-up)."""
        return max(0, len(self._totals) - self._WARMUP)


def compute_flops(model, device):       # Computes FLOPs for one forward pass using thop, returns (gflops, note_str)
    """
    Count FLOPs for one forward pass using the thop library.
    thop counts MACs (multiply-accumulate ops); multiply by 2 to get FLOPs,
    which is the conventional reporting unit in the literature.
    Returns (gflops, note_str); gflops is None if thop is unavailable.
    """
    try:
        from thop import profile as thop_profile
        dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
        with torch.no_grad():
            macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return macs * 2 / 1e9, 'thop'   # MACs × 2 = FLOPs      This gives GFLOPS
    except ImportError:
        return None, 'thop not installed — run: pip install thop'
    except Exception as e:
        return None, str(e)


def count_params(model):
    """Total parameter count (trainable + non-trainable). Useful headline number
    for comparing pruning ratios — reported in millions in the perf table."""
    return sum(p.numel() for p in model.parameters())


def get_memory_stats(model, device):    # Computes model size and peak memory usage during inference
    """
    Measure model parameter size and peak activation memory during inference.
    GPU: uses torch.cuda.max_memory_allocated (exact).
    CPU: uses tracemalloc (approximate; Python-allocated objects only).
    Returns dict with keys model_gb, peak_gb, source.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes   = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buf_bytes
    model_gb    = total_bytes / 1e9
    total_mb    = total_bytes / (1000 * 1000)   # MB (decimal megabytes)

    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(dummy)
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        source  = 'GPU'
    else:
        import tracemalloc
        tracemalloc.start()
        with torch.no_grad():
            _ = model(dummy)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_gb = peak_bytes / 1e9
        source  = 'CPU RAM'

    return {'model_gb': model_gb, 'total_mb': total_mb,
            'peak_gb': peak_gb, 'source': source}     # Measures VRAM for GPU


def print_perf_report(perf, flops_info, mem_info, device,
                      param_count=None,
                      checkpoint_path=None, arch_path=None):
    """Print a consolidated 'Model Characterisation' block: model identity,
    parameter count, compute cost, memory footprint, and inference latency.
    Designed so a single run of evaluate_detnet.py gives you everything needed
    to compare a pruned model against the baseline."""
    means                = perf.means()
    stds                 = perf.stds()
    total_mean, total_std = perf.total_mean_std()
    labels   = {
        'feature_extractor': 'Feature extractor',
        'heatmap':           'Heatmap head',
        'uv_extraction':     'UV extraction',
        'uv_to_pixels':      'UV → pixel coords',
    }
    W = 60
    print(f'\n{"═" * W}')
    print(f'  Model Characterisation')
    print(f'{"─" * W}')
    print(f'  {"Device":<26}: {device}')
    if checkpoint_path is not None:
        print(f'  {"Checkpoint":<26}: {checkpoint_path}')
    if arch_path is not None:
        print(f'  {"Architecture (_noft)":<26}: {arch_path}')

    # ── Model ─────────────────────────────────────────────────────────────
    print(f'\n  Model')
    print(f'  {"─" * 44}')
    if param_count is not None:
        print(f'  {"Parameters":<26}: {param_count/1e6:>9.3f} M  ({param_count:,})')
    print(f'  {"Model params (RAM)":<26}: {mem_info["model_gb"]:>9.4f} GB')
    print(f'  {"Total size":<26}: {mem_info["total_mb"]:>9.2f} MB  (params + buffers)')

    # ── Compute ───────────────────────────────────────────────────────────
    print(f'\n  Compute (per forward pass)')
    print(f'  {"─" * 44}')
    gflops, note = flops_info
    if gflops is not None:
        print(f'  {"GFLOPs":<26}: {gflops:>9.3f}')
    else:
        print(f'  {"GFLOPs":<26}: N/A  ({note})')

    # ── Inference latency ────────────────────────────────────────────────
    n_eff = perf.count_after_warmup()
    print(f'\n  Inference latency (mean ± std ms/frame, n={n_eff} after {perf._WARMUP}-frame warm-up)')
    print(f'  {"─" * 44}')
    for stage, label in labels.items():
        if stage in means:
            print(f'  {label:<26}: {means[stage]:>7.3f} ± {stds[stage]:<6.3f} ms')
    print(f'  {"─" * 44}')
    print(f'  {"Total (per frame)":<26}: {total_mean:>7.3f} ± {total_std:<6.3f} ms')
    if total_mean > 0:
        print(f'  {"Throughput":<26}: {1000.0 / total_mean:>9.2f} fps')

    # ── Peak runtime memory ──────────────────────────────────────────────
    print(f'\n  Peak inference memory')
    print(f'  {"─" * 44}')
    print(f'  {"Peak":<26}: {mem_info["peak_gb"]:>9.4f} GB  ({mem_info["source"]})')
    print(f'{"═" * W}\n')


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device, arch_path=None):   # Loads weights for either an unpruned (original) or a pruned + fine-tuned DetNet, returns the model ready for inference
    """Load a DetNet state_dict. Two supported cases:

    1. Unpruned (original) DetNet — bare state_dict (e.g. ckp_detnet_83.pth).
       Loaded into a fresh detnet(); unmatched keys (the original checkpoint's
       dmap/lmap weights) are silently skipped.

    2. Pruned + fine-tuned — bare state_dict with pruned-shape weights.
       Pass arch_path pointing at the matching _noft.pth so the correct
       pruned architecture is rebuilt before loading the state_dict.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: '{checkpoint_path}'")

    if arch_path is not None:
        # Case 2: rebuild pruned architecture from the _noft pickle
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
        # Case 1: unpruned (original) DetNet
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


# ── Image preprocessing ───────────────────────────────────────────────────────

def training_crop_and_transform(img_bgr, kp2d_gt, device):  # Replicates the training crop pipeline on a single image. Bbox creation --> Affine transformation --> 128x128 which will be input to DetNet
    """
    Crop and normalise an image the same way the training pipeline does, so
    the model receives input in the distribution it was trained on.

    Steps:
      1. center = mean of the GT keypoint bounding box
      2. scale  = max(bbox_w, bbox_h) × 2.0  (matches training scale_factor)
      3. Compute affine transform that maps the crop to 128×128
      4. Apply affine crop via PIL
      5. Normalise: divide by 255 then subtract 0.5 (shifts [0,1] → [−0.5, 0.5])

    Returns:
      tensor : (1, 3, 128, 128) float tensor ready for DetNet
      center : (2,)  crop centre in original image pixels
      scale  : float crop side length in original image pixels
    """
    center = handutils.get_annot_center(kp2d_gt)
    scale  = handutils.get_ori_crop_scale(mask=None, mask_flag=False,
                                          side=None, kp2d=kp2d_gt)
    affinetrans, _ = handutils.get_affine_transform_test(
        center, scale, [INPUT_SIZE, INPUT_SIZE], rot=0
    )

    pil_img  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil_crop = handutils.transform_img(pil_img, affinetrans, [INPUT_SIZE, INPUT_SIZE])
    pil_crop = pil_crop.crop((0, 0, INPUT_SIZE, INPUT_SIZE))

    tensor = func.to_tensor(pil_crop).float().to(device)
    tensor = func.normalize(tensor, [0.5, 0.5, 0.5], [1, 1, 1])
    return tensor.unsqueeze(0), center, scale


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model, tensor):         # Runs a forward pass through DetNet, returns (uv, conf) where uv is (21, 2) heatmap-space coords and conf is mean heatmap confidence
    """Simple (untimed) forward pass. Returns uv (21,2) and mean heatmap confidence."""
    with torch.no_grad():
        result = model(tensor)
    uv   = result['uv'].squeeze(0).cpu().numpy().astype(np.float32)
    conf = result['h_map'].squeeze(0).view(21, -1).max(dim=1).values.mean().item()
    return uv, conf


def infer_2d(model, tensor, center, scale, perf=None):
    """
    Run DetNet and convert heatmap UV coords to original image pixel coords.

    If perf is provided (a LatencyAccumulator), both the model stages and the
    UV→pixel conversion step are timed and recorded.

    Returns:
      uv   : (21, 2) heatmap-space coords — needed by debug_sample
      kp2d : (21, 2) pixel coords in the original (uncropped) image
    """
    if perf is not None:
        uv, _, stage_times = predict_timed(model, tensor)
        _sync(); t0 = time.perf_counter()
        kp2d = uv_to_pixels(uv, center, scale)
        _sync(); t1 = time.perf_counter()
        stage_times['uv_to_pixels'] = (t1 - t0) * 1000.0
        perf.update(stage_times)
    else:
        uv, _ = predict(model, tensor)
        kp2d  = uv_to_pixels(uv, center, scale)
    return uv, kp2d


def uv_to_pixels(uv, center, scale):
    """
    Invert the training affine crop to map heatmap UV back to original image pixels.

    The heatmap is 32×32; each cell covers (scale/32) original pixels.
    The crop origin in the original image is (center − scale/2).

    UV convention: uv[:, 0] = row (y direction), uv[:, 1] = col (x direction).
    """
    pts = np.zeros((21, 2), dtype=np.float32)
    pts[:, 0] = uv[:, 1] * scale / HEATMAP_SIZE + (center[0] - scale / 2)  # x = col
    pts[:, 1] = uv[:, 0] * scale / HEATMAP_SIZE + (center[1] - scale / 2)  # y = row
    return pts


def debug_sample(sample_idx, uv, center, scale, kp2d_pred, kp2d_gt, img_h, img_w):  # Diagnostic printer used for debugging. Print predicted heatmap cell, GT heatmap cell, predicted pixel and GT pixel.
    """Print round-trip and model-vs-GT diagnostic info for one frame."""
    print(f'\n── Debug sample {sample_idx} ──────────────────────────')
    print(f'  center={center}  scale={scale:.1f}  img={img_w}×{img_h}')
    print(f'  GT affine round-trip (should be ~0):')
    for j in [0, 4, 8, 12, 16, 20]:   # wrist, thumb-tip, index-tip, …
        gx, gy = kp2d_gt[j]
        cx128 = (gx - (center[0] - scale/2)) * INPUT_SIZE / scale
        cy128 = (gy - (center[1] - scale/2)) * INPUT_SIZE / scale
        vh = np.clip(int(cx128 * HEATMAP_SIZE / INPUT_SIZE), 0, HEATMAP_SIZE - 1)
        uh = np.clip(int(cy128 * HEATMAP_SIZE / INPUT_SIZE), 0, HEATMAP_SIZE - 1)
        x_rt   = vh * scale / HEATMAP_SIZE + (center[0] - scale / 2)
        y_rt   = uh * scale / HEATMAP_SIZE + (center[1] - scale / 2)
        rt_err = np.sqrt((x_rt - gx)**2 + (y_rt - gy)**2)
        print(f'    j{j:2d}: GT({gx:.0f},{gy:.0f}) → hm({uh},{vh}) '
              f'→ rt({x_rt:.0f},{y_rt:.0f})  err={rt_err:.1f}px')
    print(f'  Model UV output (row,col) vs expected GT heatmap positions:')
    for j in [0, 4, 8, 12, 16, 20]:
        gx, gy = kp2d_gt[j]
        cx128  = (gx - (center[0] - scale/2)) * INPUT_SIZE / scale
        cy128  = (gy - (center[1] - scale/2)) * INPUT_SIZE / scale
        vh_gt  = np.clip(int(cx128 * HEATMAP_SIZE / INPUT_SIZE), 0, HEATMAP_SIZE - 1)
        uh_gt  = np.clip(int(cy128 * HEATMAP_SIZE / INPUT_SIZE), 0, HEATMAP_SIZE - 1)
        u_pred, v_pred = int(uv[j, 0]), int(uv[j, 1])
        px, py = kp2d_pred[j]
        print(f'    j{j:2d}: model_uv=({u_pred},{v_pred})  gt_uv=({uh_gt},{vh_gt})  '
              f'pred_px=({px:.0f},{py:.0f})  gt_px=({gx:.0f},{gy:.0f})')
    print()


# ── Metrics ───────────────────────────────────────────────────────────────────

def per_sample_metrics(kp2d_pred, kp2d_gt, norm):
    """
    Compute per-frame metrics.
    kp2d_pred, kp2d_gt : (n, 2) pixel coordinates.
    norm    : per-frame normalisation factor in pixels — GT-keypoint bounding-box
              side, max(bbox_w, bbox_h), following Simon et al. 2017 (CPM-Hands).
              Makes MPJPE scale-invariant: the error stays comparable whether
              the hand is close to the camera or far away.
    Returns raw pixel errors, the per-frame norm (saved for post-hoc renorm),
    and the bbox-normalised MPJPE.
    """
    errors_px   = np.linalg.norm(kp2d_pred - kp2d_gt, axis=1)   # (n,) pixels
    errors_norm = errors_px / norm
    return {
        'errors_px': errors_px,
        'norm_px':   float(norm),
        'mpjpe':     float(errors_norm.mean()),
    }


def compute_summary_stats(results, tau_min=0, tau_max=30):
    """Aggregate per-frame results into the headline stats used by summarise()
    (printing) and save_run() (persistence) — kept in one function so the
    numbers shown on stdout and saved to disk are guaranteed identical."""
    if not results:
        return None

    mpjpe = float(np.mean([r['mpjpe'] for r in results]))

    # PCK(τ) curve: for each threshold τ, fraction of joints within τ pixels
    all_errors_list = [r['errors_px'] for r in results]
    total_joints    = sum(len(e) for e in all_errors_list)
    taus            = np.arange(tau_min, tau_max + 1, dtype=float)
    pck_curve       = []
    for tau in taus:
        hits = sum(int((e < tau).sum()) for e in all_errors_list)
        pck_curve.append(min(hits / total_joints * 100.0, 100.0))

    auc = float(np.trapezoid(pck_curve, taus)) / (tau_max - tau_min)

    return {
        'n_frames':    len(results),
        'mpjpe':       mpjpe,
        'taus':        taus,
        'pck_curve':   pck_curve,
        'auc':         auc,
        'all_errors':  np.concatenate(all_errors_list),  # (N*21,) raw pixel errors
        'all_norms':   np.array([r['norm_px'] for r in results], dtype=np.float32),  # (N,) per-frame GT-bbox side in pixels
    }


def summarise(results, tau_min=0, tau_max=30):
    """
    Aggregate per-frame results and print MPJPE and PCK-AUC.
    PCK(τ) is swept from tau_min to tau_max pixels and the AUC is
    normalised to [0, 100] by dividing the integral by (tau_max − tau_min).
    """
    stats = compute_summary_stats(results, tau_min=tau_min, tau_max=tau_max)
    if stats is None:
        print('No results to summarise.')
        return

    print(f'\n{"─" * 50}')
    print(f'  Frames evaluated       : {stats["n_frames"]}')
    print(f'  MPJPE                  : {stats["mpjpe"]:.4f}  (normalised units)')
    print(f'  PCK @ {tau_min}px              : {stats["pck_curve"][0]:.2f} %')
    print(f'  PCK @ {tau_max}px              : {stats["pck_curve"][-1]:.2f} %')
    print(f'  AUC PCK({tau_min}-{tau_max}px)       : {stats["auc"]:.2f} %')
    print(f'{"─" * 50}\n')


def save_run(out_dir, run_name, *, results_by_split, args, device,
             n_params, flops_info, mem_info, perf):
    """Persist headline metrics + per-frame errors so plots can be regenerated
    without re-running evaluation. Writes <out_dir>/<run_name>/:
      summary.json — model identity, accuracy stats, latency, GFLOPs, memory.
      errors.npz   — for each split <name>:
                       <name>      : (N*21,) float32 raw pixel errors
                       norm_<name> : (N,)    float32 per-frame GT-bbox side (px)
                     Lets PCK be re-swept at any tau range and lets MPJPE be
                     re-normalised post-hoc (image-size or bbox).
    Overwrites any previous run with the same name."""
    import json
    from datetime import datetime

    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    splits        = {}
    errors_arrays = {}
    for name, results in results_by_split.items():
        stats = compute_summary_stats(results, tau_min=args.tau_min, tau_max=args.tau_max)
        if stats is None:
            continue
        splits[name] = {
            'n_frames':  stats['n_frames'],
            'mpjpe':     stats['mpjpe'],
            'pck_taus':  stats['taus'].tolist(),
            'pck_curve': stats['pck_curve'],
            'auc_pck':   stats['auc'],
        }
        errors_arrays[name]           = stats['all_errors'].astype(np.float32)
        errors_arrays[f'norm_{name}'] = stats['all_norms'].astype(np.float32)

    gflops, gflops_note   = flops_info
    means                 = perf.means()
    stds                  = perf.stds()
    total_mean, total_std = perf.total_mean_std()

    summary = {
        'timestamp':    datetime.now().isoformat(timespec='seconds'),
        'checkpoint':   args.checkpoint,
        'arch':         args.arch,
        'dataset':      args.dataset,
        'split':        args.split,
        'device':       str(device),
        'max_samples':  args.max_samples,
        'tau_min':      args.tau_min,
        'tau_max':      args.tau_max,
        'model': {
            'param_count':   int(n_params),
            'param_count_M': n_params / 1e6,
            'ram_gb':        mem_info['model_gb'],
            'total_mb':      mem_info['total_mb'],
        },
        'compute': {
            'gflops_forward': gflops,
            'gflops_note':    gflops_note,
        },
        'latency': {
            'frames_profiled':     perf.count(),
            'frames_after_warmup': perf.count_after_warmup(),
            'warmup_frames':       perf._WARMUP,
            'stages_ms':           means,        # per-stage mean ms
            'stages_ms_std':       stds,         # per-stage std ms
            'total_ms_per_frame':  total_mean,
            'total_ms_std':        total_std,
            'fps':                 (1000.0 / total_mean) if total_mean > 0 else None,
        },
        'memory': {
            'peak_inference_gb': mem_info['peak_gb'],
            'peak_source':       mem_info['source'],
        },
        'splits': splits,
    }

    json_path = os.path.join(run_dir, 'summary.json')
    npz_path  = os.path.join(run_dir, 'errors.npz')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    if errors_arrays:
        np.savez_compressed(npz_path, **errors_arrays)

    print(f'Saved → {run_dir}')


# ── FreiHAND ──────────────────────────────────────────────────────────────────

def project_3d_to_2d(xyz, K):
    """Perspective-project (21, 3) camera-space joints using intrinsics K (3, 3)."""
    xyz = np.array(xyz, dtype=np.float32)
    K   = np.array(K,   dtype=np.float32)
    uvw = K @ xyz.T        # (3, 21)
    uv  = (uvw[:2] / uvw[2]).T   # (21, 2) [x, y]
    return uv


def run_freihand(data_root, model, device, max_samples,
                 n_debug=0, split='eval', perf=None):
    """
    Evaluate on FreiHAND.
    split='eval'  : ~3,960 images (evaluation_xyz/K.json)
    split='train' : 32,560 unique poses. The training folder holds 4× images
                    (set 0 = green screen, sets 1-3 = composited backgrounds);
                    we evaluate the FIRST composited-background set, image
                    indices [32560, 65120), mapped to GT via index % 32560.
    GT is 3D camera-space xyz; projected to 2D with the provided K matrix.
    """
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
    n       = min(max_samples, len(xyz_list), len(img_files))
    results = []

    # FreiHAND's training/ folder holds 130,240 images = 4 repetitions of the
    # same 32,560 unique hand captures. Set 0 = indices [0, 32560) is the plain
    # GREEN-SCREEN capture; sets 1-3 are those identical captures composited onto
    # real backgrounds. The GT (training_xyz/K.json) only covers the first 32,560
    # samples. We evaluate on the FIRST composited-background set, image indices
    # [32560, 65120), to avoid the green screen, and recover each image's GT via
    # label_index = image_index % 32560 (== image_index - 32560 here). The eval/
    # folder has no such repetition, so it keeps offset 0.
    n_unique   = len(xyz_list)                          # 32,560 for the train split
    img_offset = n_unique if split == 'train' else 0    # start of first composited set

    for i in tqdm(range(n), desc=desc):
        img_index   = img_offset + i           # [32560, 65120) for the train split
        label_index = img_index % n_unique     # GT only covers the first 32,560 samples
        img = cv2.imread(img_files[img_index])
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        kp2d_gt = project_3d_to_2d(xyz_list[label_index], K_list[label_index])   # (21, 2) pixels
        tensor, center, scale = training_crop_and_transform(img, kp2d_gt, device)
        uv, kp2d_pred         = infer_2d(model, tensor, center, scale, perf)

        if i < n_debug:
            debug_sample(i, uv, center, scale, kp2d_pred, kp2d_gt, img_h, img_w)

        # Simon et al. 2017 (CPM-Hands): normalise per-joint pixel error by the
        # GT-keypoint bounding-box side. `scale` here is max(bbox_w, bbox_h) *
        # 2.0 (training-crop convention, see training_crop_and_transform docstring),
        # so the true bbox side is scale / 2.0. Per-frame norm makes MPJPE robust
        # to hand-in-frame size variation.
        norm = float(scale / 2.0)
        results.append(per_sample_metrics(kp2d_pred, kp2d_gt, norm))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate DetNet 2D pose')
    parser.add_argument('--dataset',
                        choices=['freihand', 'both'],
                        default='both',
                        help='Default "both" = FreiHAND eval + train (the full '
                             'unseen set) — the standard evaluation. Use '
                             '"freihand" + --split for a single split. RHD was '
                             'removed: its eval split was the fine-tuning '
                             'validation set, not a held-out test set.')
    parser.add_argument('--data_root', type=str,
                        default='C:/Imperial/FYP Data',
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--checkpoint', type=str,
                        default='new_check_point/ckp_detnet_83.pth',
                        help='Path to a DetNet state_dict (.pth). For the '
                             'unpruned original model use it alone; for a '
                             'pruned + fine-tuned model also pass --arch.')
    parser.add_argument('--arch', type=str, default=None,
                        help='Path to the matching pruned architecture file '
                             '(an _noft.pth from pruned_architectures/). '
                             'Required when --checkpoint is a pruned + '
                             'fine-tuned state_dict; omit for the unpruned '
                             'original model.')
    parser.add_argument('--max_samples', type=int, default=999_999,
                        help='Max images to evaluate per sub-run (default: unlimited)')
    parser.add_argument('--tau_min', type=int, default=0,
                        help='Lower PCK threshold in pixels (default: 0)')
    parser.add_argument('--tau_max', type=int, default=30,
                        help='Upper PCK threshold in pixels (default: 30)')
    parser.add_argument('--n_debug', type=int, default=0,
                        help='Print debug info for the first N samples (default: 0)')
    parser.add_argument('--split', choices=['eval', 'train'], default='eval',
                        help='FreiHAND split for --dataset freihand '
                             '(eval ~3,960 | train ~32,560). '
                             'Ignored when --dataset both (runs eval + train).')
    parser.add_argument('--out_dir', type=str, default='eval_results',
                        help='Folder where per-run summary.json + errors.npz '
                             'are saved (default: eval_results). Each run goes '
                             'into <out_dir>/<checkpoint_stem>_<dataset>/.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='Default cpu — the deployment/measurement target '
                             '(accuracy is device-independent; reported latency '
                             'is the meaningful CPU number). Pass cuda to use a '
                             'GPU if available.')
    args = parser.parse_args()

    want_cuda = (args.device == 'cuda') and torch.cuda.is_available()
    if args.device == 'cuda' and not want_cuda:
        print('CUDA requested but unavailable — using CPU.')
    device = torch.device('cuda:0' if want_cuda else 'cpu')
    print(f'Device : {device}')
    model = load_model(args.checkpoint, device, arch_path=args.arch)

    # Compute model-level stats once before the evaluation loop
    n_params   = count_params(model)
    flops_info = compute_flops(model, device)
    mem_info   = get_memory_stats(model, device)
    perf       = LatencyAccumulator()

    results_by_split = {}
    if args.dataset == 'freihand':
        results = run_freihand(
            args.data_root, model, device, args.max_samples,
            n_debug=args.n_debug, split=args.split, perf=perf,
        )
        summarise(results, tau_min=args.tau_min, tau_max=args.tau_max)
        results_by_split[f'freihand_{args.split}'] = results
    else:  # both = FreiHAND eval + train (the full unseen set; RHD removed)
        fh_eval = run_freihand(
            args.data_root, model, device, args.max_samples,
            n_debug=args.n_debug, split='eval', perf=perf,
        )
        fh_train = run_freihand(
            args.data_root, model, device, args.max_samples,
            n_debug=args.n_debug, split='train', perf=perf,
        )
        print('\n══ FreiHAND (eval + train) ══')
        summarise(fh_eval + fh_train, tau_min=args.tau_min, tau_max=args.tau_max)

        results_by_split['freihand_eval']  = fh_eval
        results_by_split['freihand_train'] = fh_train

    print_perf_report(perf, flops_info, mem_info, device,
                      param_count=n_params,
                      checkpoint_path=args.checkpoint,
                      arch_path=args.arch)

    # Persist everything for offline plotting / cross-model comparison
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    run_name  = f'{ckpt_stem}_{args.dataset}'
    save_run(args.out_dir, run_name,
             results_by_split=results_by_split,
             args=args, device=device,
             n_params=n_params,
             flops_info=flops_info, mem_info=mem_info, perf=perf)


if __name__ == '__main__':
    main()
