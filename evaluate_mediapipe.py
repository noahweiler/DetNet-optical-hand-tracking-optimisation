"""
evaluate_mediapipe.py — Evaluate Google MediaPipe Hands on the SAME datasets,
GT, and metrics as evaluate_detnet.py, writing the SAME eval_results/ schema.

This is the deployment-baseline counterpart to evaluate_detnet.py. It is
deliberately self-contained (no torch / model.detnet import) — MediaPipe is a
closed tflite graph, so the DetNet stack is irrelevant here. The metric maths
(project_3d_to_2d, per_sample_metrics, compute_summary_stats) are byte-for-byte
the same as evaluate_detnet.py so MediaPipe vs DetNet numbers are directly
comparable.

Differences from DetNet, by design:
  - No crop / affine. MediaPipe runs on the full RGB image and returns 21
    normalised landmarks; we scale them to pixels and match the detected
    hand to the GT by closest centroid.
  - Detection can FAIL. Accuracy (MPJPE/PCK/AUC, errors.npz) is computed over
    DETECTED frames only, so it is directly comparable to DetNet — DetNet is
    handed a GT-centred crop and always emits 21 points, so it has no concept
    of a "miss". Detection misses are NOT folded into the accuracy numbers
    (an image-diagonal penalty would conflate detection recall with
    localisation error); instead a per-split `detection_rate`
    (= n_detected / n_attempted) is recorded to summary.json only — it is
    NOT printed, so stdout (summarise + Model Characterisation block) is
    visually identical to evaluate_detnet.py.
  - Latency is the MediaPipe `process()` call only (mirrors DetNet's
    model-only timing that excluded data loading), CPU perf_counter timed,
    mean ± std with a warm-up discard.

Joint order: MediaPipe Hands (wrist, thumb 1-4, index 5-8, middle 9-12,
ring 13-16, pinky 17-20, MCP→tip) matches the SNAP order FreiHAND GT is in,
so no re-index is needed.

RHD is intentionally NOT an evaluation target: its eval split was the
fine-tuning validation set (finetune_pruned.py --datasets_test rhd), so
scoring it would be train/validation leakage. FreiHAND (eval + train) is the
only legitimate held-out benchmark — mirrors evaluate_detnet.py.

Persistence policy: ONLY the full sweep is written to eval_results/ —
`--dataset both` with no `--max_samples` cap → eval_results/mediapipe_hands_both/.
Single-dataset or capped runs still print their numbers but are NOT saved
(they are partial/smoke tests and would pollute the results table).

Usage:
  python evaluate_mediapipe.py                                     # both (default), saved
  python evaluate_mediapipe.py --dataset freihand --split eval     # printed only
  python evaluate_mediapipe.py --max_samples 200                   # smoke; printed only
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')   # tee/redirect-safe on Windows
except Exception:
    pass

import argparse
import glob
import json
import time
from datetime import datetime

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe not installed — run:  pip install mediapipe")
    raise


# ── Constants (shared convention with evaluate_detnet.py) ────────────────────

_LATENCY_WARMUP = 10   # frames discarded before latency stats (match evaluate_detnet)

# MediaPipe ≥0.10 dropped the legacy mp.solutions API; the Tasks API needs a
# downloadable .task bundle (not shipped in the wheel).
_MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
              'hand_landmarker/float16/1/hand_landmarker.task')


def ensure_model(path):
    """Download the hand_landmarker .task bundle if it isn't already present."""
    if os.path.isfile(path):
        return path
    import urllib.request
    print(f'Downloading MediaPipe hand_landmarker model -> {path}')
    urllib.request.urlretrieve(_MODEL_URL, path)
    print(f'  ({os.path.getsize(path) / (1000 * 1000):.2f} MB)')
    return path


# ── Metric maths — identical to evaluate_detnet.py ──────────────────────────

def project_3d_to_2d(xyz, K):
    """Perspective-project (21, 3) camera-space joints with intrinsics K (3, 3)."""
    xyz = np.array(xyz, dtype=np.float32)
    K   = np.array(K,   dtype=np.float32)
    uvw = K @ xyz.T
    return (uvw[:2] / uvw[2]).T          # (21, 2) [x, y]


def per_sample_metrics(kp2d_pred, kp2d_gt, norm):
    """Per-frame pixel errors + bbox-normalised MPJPE (same as evaluate_detnet.py).
    `norm` is the per-frame GT-keypoint bbox side (Simon et al. 2017)."""
    errors_px   = np.linalg.norm(kp2d_pred - kp2d_gt, axis=1)
    errors_norm = errors_px / norm
    return {'errors_px': errors_px, 'norm_px': float(norm),
            'mpjpe': float(errors_norm.mean())}


def compute_summary_stats(results, tau_min=0, tau_max=30):
    """Aggregate stats — identical integration scheme to evaluate_detnet.py so
    MediaPipe AUC/PCK is directly comparable to DetNet AUC/PCK."""
    if not results:
        return None
    mpjpe = float(np.mean([r['mpjpe'] for r in results]))
    all_errors_list = [r['errors_px'] for r in results]
    total_joints    = sum(len(e) for e in all_errors_list)
    taus            = np.arange(tau_min, tau_max + 1, dtype=float)
    pck_curve       = []
    for tau in taus:
        hits = sum(int((e < tau).sum()) for e in all_errors_list)
        pck_curve.append(min(hits / total_joints * 100.0, 100.0))
    auc = float(np.trapz(pck_curve, taus)) / (tau_max - tau_min)
    return {
        'n_frames':   len(results),
        'mpjpe':      mpjpe,
        'taus':       taus,
        'pck_curve':  pck_curve,
        'auc':        auc,
        'all_errors': np.concatenate(all_errors_list),
        'all_norms':  np.array([r['norm_px'] for r in results], dtype=np.float32),
    }


def summarise(stats, tau_min=0, tau_max=30):
    """Print MPJPE / PCK / AUC in the exact same format as evaluate_detnet.py's
    summarise(). Stats are over detected frames only (the recall figure is
    saved to summary.json, not printed, so stdout mirrors DetNet)."""
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


def print_model_characterisation(model_path, device_str, lat):
    """Consolidated 'Model Characterisation' block — same layout as
    evaluate_detnet.py print_perf_report() so MediaPipe and DetNet stdout are
    visually identical. Fields that can't be extracted from a closed tflite
    graph (param count, GFLOPs, peak memory) are reported as N/A."""
    lat_mean, lat_std = lat.mean_std()
    n_eff    = lat.count_after_warmup()
    total_mb = (os.path.getsize(model_path) / (1000 * 1000)
                if os.path.isfile(model_path) else None)
    W = 60
    print(f'\n{"═" * W}')
    print(f'  Model Characterisation')
    print(f'{"─" * W}')
    print(f'  {"Device":<26}: {device_str}')
    print(f'  {"Checkpoint":<26}: {model_path}')

    print(f'\n  Model')
    print(f'  {"─" * 44}')
    print(f'  {"Parameters":<26}: N/A  (closed tflite graph)')
    print(f'  {"Model params (RAM)":<26}: N/A')
    if total_mb is not None:
        print(f'  {"Total size":<26}: {total_mb:>9.2f} MB  (tflite bundle)')
    else:
        print(f'  {"Total size":<26}: N/A')

    print(f'\n  Compute (per forward pass)')
    print(f'  {"─" * 44}')
    print(f'  {"GFLOPs":<26}: N/A  (MediaPipe tflite)')

    print(f'\n  Inference latency (mean ± std ms/frame, n={n_eff} after {_LATENCY_WARMUP}-frame warm-up)')
    print(f'  {"─" * 44}')
    print(f'  {"MediaPipe detect":<26}: {lat_mean:>7.3f} ± {lat_std:<6.3f} ms')
    print(f'  {"─" * 44}')
    print(f'  {"Total (per frame)":<26}: {lat_mean:>7.3f} ± {lat_std:<6.3f} ms')
    if lat_mean > 0:
        print(f'  {"Throughput":<26}: {1000.0 / lat_mean:>9.2f} fps')

    print(f'\n  Peak inference memory')
    print(f'  {"─" * 44}')
    print(f'  {"Peak":<26}: N/A  (tflite, not instrumented)')
    print(f'{"═" * W}\n')


# ── MediaPipe predictor ─────────────────────────────────────────────────────

class MediaPipeHands:
    """Thin wrapper over the MediaPipe Tasks HandLandmarker: full BGR image
    -> (21, 2) pixel landmarks, matched to the GT hand by centroid.
    Returns (None, latency_ms) on a detection miss."""

    def __init__(self, model_path, max_num_hands=2, min_detection_confidence=0.5):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            running_mode=vision.RunningMode.IMAGE,      # independent images
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)

    def predict(self, img_bgr, gt_centroid):
        h, w = img_bgr.shape[:2]
        rgb  = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t0 = time.perf_counter()
        res = self._lm.detect(mp_img)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not res.hand_landmarks:
            return None, latency_ms

        # Tasks API: res.hand_landmarks is a list (one per hand) of 21
        # NormalizedLandmark. Scale to pixels; pick the hand whose centroid is
        # closest to the GT centroid (handles 2-hand RHD frames correctly).
        best_kp, best_d = None, np.inf
        for hand in res.hand_landmarks:
            kp = np.array([[lm.x * w, lm.y * h] for lm in hand],
                          dtype=np.float32)               # (21, 2)
            d = np.linalg.norm(kp.mean(axis=0) - gt_centroid)
            if d < best_d:
                best_kp, best_d = kp, d
        return best_kp, latency_ms

    def close(self):
        self._lm.close()


# ── Latency accumulator (mean ± std, warm-up discard) ───────────────────────

class Latency:
    def __init__(self):
        self._t = []

    def add(self, ms):
        self._t.append(ms)

    def _trim(self):
        return self._t[_LATENCY_WARMUP:] if len(self._t) > _LATENCY_WARMUP else self._t

    def mean_std(self):
        v = self._trim()
        if not v:
            return 0.0, 0.0
        if len(v) < 2:
            return float(v[0]), 0.0
        return float(np.mean(v)), float(np.std(v))

    def count(self):
        return len(self._t)

    def count_after_warmup(self):
        return max(0, len(self._t) - _LATENCY_WARMUP)


# ── Dataset runners (same GT extraction as evaluate_detnet.py) ──────────────

def _score_frame(pred, kp2d_gt, img_h, img_w):
    """Per-frame metrics for a DETECTED hand, or None on a detection miss.

    A miss is excluded from accuracy entirely (NOT penalised) so MPJPE/PCK/AUC
    are computed over the same kind of input DetNet sees: 21 predicted points
    vs 21 GT points, bbox-normalised per Simon et al. 2017. The miss is instead
    counted toward the recorded detection_rate — MediaPipe's distinct real-world
    failure mode, kept separate from localisation error.
    """
    if pred is None:
        return None
    # Same per-frame norm as evaluate_detnet: max side of the GT-keypoint bbox.
    # evaluate_detnet derives this from `scale = max(bbox_w, bbox_h) * 2.0`, here
    # we compute it directly from kp2d_gt — identical formula.
    delta = kp2d_gt.max(0) - kp2d_gt.min(0)
    norm  = float(max(delta[0], delta[1]))
    return per_sample_metrics(pred, kp2d_gt, norm)


def run_freihand(data_root, mp_model, max_samples, lat, split='eval'):
    root = os.path.join(data_root, 'FreiHAND')
    if split == 'train':
        img_dir  = os.path.join(root, 'training', 'rgb')
        xyz_file = os.path.join(root, 'training_xyz.json')
        K_file   = os.path.join(root, 'training_K.json')
        desc     = 'FreiHAND train'
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

    # FreiHAND's training/ folder holds 130,240 images = 4 repetitions of the
    # same 32,560 unique hand captures. Set 0 = indices [0, 32560) is the plain
    # GREEN-SCREEN capture; sets 1-3 are those identical captures composited onto
    # real backgrounds. The GT (training_xyz/K.json) only covers the first 32,560
    # samples. We evaluate on the FIRST composited-background set, image indices
    # [32560, 65120), to avoid the green screen, and recover each image's GT via
    # label_index = image_index % 32560 (== image_index - 32560 here). The eval/
    # folder has no such repetition, so it keeps offset 0. (Matches evaluate_detnet.)
    n_unique   = len(xyz_list)                          # 32,560 for the train split
    img_offset = n_unique if split == 'train' else 0    # start of first composited set

    results, n_attempted = [], 0
    for i in range(n):
        if i % 500 == 0:
            print(f'  {desc}: {i}/{n}', flush=True)
        img_index   = img_offset + i           # [32560, 65120) for the train split
        label_index = img_index % n_unique     # GT only covers the first 32,560 samples
        img = cv2.imread(img_files[img_index])
        if img is None:
            continue
        h, w = img.shape[:2]
        n_attempted += 1
        kp2d_gt = project_3d_to_2d(xyz_list[label_index], K_list[label_index]).astype(np.float32)
        pred, ms = mp_model.predict(img, kp2d_gt.mean(axis=0))
        lat.add(ms)
        m = _score_frame(pred, kp2d_gt, h, w)   # None on a detection miss
        if m is not None:
            results.append(m)                   # detected-only accuracy
    return results, n_attempted


# ── Persistence (same schema as evaluate_detnet.py save_run) ────────────────

def save_run(out_dir, run_name, *, results_by_split, attempted_by_split,
             args, lat):
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    splits, errors_arrays = {}, {}
    for name, results in results_by_split.items():
        stats = compute_summary_stats(results, tau_min=args.tau_min,
                                      tau_max=args.tau_max)
        if stats is None:
            continue
        n_det = stats['n_frames']                       # detected == scored
        n_att = attempted_by_split.get(name, n_det)     # frames MediaPipe ran on
        splits[name] = {
            'n_frames':       n_det,                     # frames the stats cover (detected)
            'n_attempted':    n_att,
            'n_detected':     n_det,
            'detection_rate': n_det / n_att if n_att else 0.0,
            'mpjpe':          stats['mpjpe'],            # detected-only
            'pck_taus':       stats['taus'].tolist(),
            'pck_curve':      stats['pck_curve'],
            'auc_pck':        stats['auc'],
        }
        errors_arrays[name]           = stats['all_errors'].astype(np.float32)
        errors_arrays[f'norm_{name}'] = stats['all_norms'].astype(np.float32)

    lat_mean, lat_std = lat.mean_std()

    # Model-size field = the .task bundle on disk (MB, decimal megabytes).
    total_mb = (os.path.getsize(args.model_path) / (1000 * 1000)
                if os.path.isfile(args.model_path) else None)

    summary = {
        'timestamp':   datetime.now().isoformat(timespec='seconds'),
        'checkpoint':  'mediapipe_hands',          # parsed as method 'baseline'
        'arch':        None,
        'dataset':     args.dataset,
        'split':       args.split,
        'device':      'CPU (MediaPipe tflite)',
        'max_samples': args.max_samples,
        'tau_min':     args.tau_min,
        'tau_max':     args.tau_max,
        'model': {
            'param_count':   None,                 # closed tflite graph
            'param_count_M': None,
            'ram_gb':        None,
            'total_mb':      total_mb,             # tflite file size (MB)
            'note':          'MediaPipe Hands is a closed tflite graph; '
                             'param/FLOP counts are not extractable via thop.',
        },
        'compute': {
            'gflops_forward': None,
            'gflops_note':    'N/A for MediaPipe tflite',
        },
        'latency': {
            'frames_profiled':     lat.count(),
            'frames_after_warmup': lat.count_after_warmup(),
            'warmup_frames':       _LATENCY_WARMUP,
            'total_ms_per_frame':  lat_mean,       # keys mirror evaluate_detnet
            'total_ms_std':        lat_std,
            'fps':                 (1000.0 / lat_mean) if lat_mean > 0 else None,
            'timing_method':       'perf_counter (CPU, process() only)',
        },
        'memory': {
            'peak_inference_gb': None,
            'peak_source':       'N/A',
        },
        'splits': splits,
    }

    with open(os.path.join(run_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    if errors_arrays:
        np.savez_compressed(os.path.join(run_dir, 'errors.npz'), **errors_arrays)
    print(f'Saved -> {run_dir}')


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Evaluate MediaPipe Hands (DetNet-comparable metrics)')
    p.add_argument('--dataset', choices=['freihand', 'both'], default='both',
                   help='Default "both" = FreiHAND eval + train (the full unseen '
                        'set) — the standard evaluation. "freihand" + --split '
                        'evaluates a single split. RHD removed: its eval split '
                        'was the fine-tuning validation set, not held-out.')
    p.add_argument('--data_root', type=str, default='C:/Imperial/FYP Data')
    p.add_argument('--max_samples', type=int, default=999_999)
    p.add_argument('--tau_min', type=int, default=0)
    p.add_argument('--tau_max', type=int, default=30)
    p.add_argument('--split', choices=['eval', 'train'], default='eval',
                   help='Only used with --dataset freihand. The default '
                        '(--dataset both) always evaluates eval + train.')
    p.add_argument('--out_dir', type=str, default='eval_results')
    p.add_argument('--allow_partial_save', action='store_true',
                   help='Save even a capped (--max_samples) run. Off by default so '
                        'incomplete numbers do not pollute eval_results/; used for '
                        'smoke tests via quant/run_all_eval.py --max_samples N.')
    p.add_argument('--model_path', type=str, default='hand_landmarker.task',
                   help='MediaPipe .task bundle; auto-downloaded if absent.')
    p.add_argument('--max_num_hands', type=int, default=2)
    p.add_argument('--min_detection_confidence', type=float, default=0.5)
    args = p.parse_args()

    ensure_model(args.model_path)
    print(f'Model  : MediaPipe Tasks HandLandmarker ({args.model_path}, '
          f'max_hands={args.max_num_hands}, conf={args.min_detection_confidence})')
    mp_model = MediaPipeHands(args.model_path, args.max_num_hands,
                              args.min_detection_confidence)
    lat = Latency()

    results_by_split, attempted_by_split = {}, {}
    if args.dataset == 'freihand':
        res, att = run_freihand(args.data_root, mp_model, args.max_samples, lat,
                                split=args.split)
        key = f'freihand_{args.split}'
        results_by_split[key], attempted_by_split[key] = res, att
        summarise(compute_summary_stats(res, args.tau_min, args.tau_max),
                  tau_min=args.tau_min, tau_max=args.tau_max)
    else:  # both = FreiHAND eval + train (the full unseen set; RHD removed)
        fe, fea = run_freihand(args.data_root, mp_model, args.max_samples, lat, 'eval')
        ft, fta = run_freihand(args.data_root, mp_model, args.max_samples, lat, 'train')
        results_by_split = {'freihand_eval': fe, 'freihand_train': ft}
        attempted_by_split = {'freihand_eval': fea, 'freihand_train': fta}

        print('\n══ FreiHAND (eval + train) ══')
        summarise(compute_summary_stats(fe + ft, args.tau_min, args.tau_max),
                  tau_min=args.tau_min, tau_max=args.tau_max)

    # Model Characterisation block — same layout as evaluate_detnet.py.
    # (Detection recall is saved to summary.json, not printed, so stdout
    #  mirrors DetNet exactly.)
    print_model_characterisation(args.model_path, 'CPU (MediaPipe tflite)', lat)

    mp_model.close()

    # Persist ONLY the full sweep (--dataset both, uncapped). Single-dataset
    # or --max_samples-capped runs are partial/smoke tests; saving them would
    # pollute eval_results/ with incomplete MediaPipe numbers that
    # build_results_csv.py would then have to second-guess.
    is_full_run = (args.dataset == 'both' and args.max_samples >= 999_999)
    if is_full_run or args.allow_partial_save:
        save_run(args.out_dir, 'mediapipe_hands_both',
                 results_by_split=results_by_split,
                 attempted_by_split=attempted_by_split,
                 args=args, lat=lat)
    else:
        print('\n[partial run — NOT saved to disk]')
        print('  evaluate_mediapipe.py only persists the full sweep: '
              '--dataset both with no --max_samples cap.')
        print('  This run was a single-dataset or capped (smoke) run; its '
              'numbers are printed above but intentionally not written to '
              f'{args.out_dir}/.')


if __name__ == '__main__':
    main()
