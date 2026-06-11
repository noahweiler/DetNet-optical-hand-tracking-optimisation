"""
3D evaluation of DetNet on Dexter+Object, EgoDexter, and RHD.

Matches the evaluation protocol used in:
  Mueller et al. / Baek et al. / similar works that report AUC of PCK on
  DO, ED, and RHD with thresholds 20–50 mm.

Protocol:
  DO / ED : centroid-align predicted fingertip positions to GT fingertip centroid.
  RHD     : root-align predicted wrist (joint 0) to GT wrist position.

  Because DetNet outputs bone-normalised xyz (not absolute mm), isotropic
  Procrustes scaling is applied before alignment to resolve the scale ambiguity.
  The paper's method includes MANO shape estimation and outputs absolute mm,
  so it uses translation-only alignment. Our numbers are therefore directly
  comparable at the pose-shape level once scale is normalised.

Metrics (matching the paper's Table 1):
  MPJPE   — mean 3D Euclidean error in mm (post-alignment)
  PCK(τ)  — fraction of joints within τ mm, τ ∈ [tau_min, tau_max]
  AUC     — area under PCK(τ) curve, normalised to [0, 1]  ← paper format
  DFR     — fraction of frames where mean error > dfr_tau_mm

Usage:
  python evaluate_detnet3D.py --dataset dexter
  python evaluate_detnet3D.py --dataset egodexter
  python evaluate_detnet3D.py --dataset rhd --split eval
"""
import argparse
import glob
import os
import pickle

import cv2
import numpy as np
import torch
from tqdm import tqdm

from evaluate_detnet import (
    load_model,
    training_crop_and_transform,
    _FINGERTIP_SNAP,
    _DEXTER_SEQS,
    _EGODEXTER_SEQS,
    _RHD_TO_SNAP,
    predict_timed,
    LatencyAccumulator,
    compute_flops,
    get_memory_stats,
    print_perf_report,
)

# ── Alignment ─────────────────────────────────────────────────────────────────

def scale_then_centroid_align(pred, gt):
    """
    Resolve scale ambiguity then translate centroid.
    pred, gt : (n, 3).  Returns pred_aligned in same units as gt.
    Step 1 — isotropic Procrustes scale: s = ||gt_centred|| / ||pred_centred||
    Step 2 — translate so centroid of scaled pred matches centroid of gt.
    """
    pred_c = pred - pred.mean(axis=0)
    gt_c   = gt   - gt.mean(axis=0)
    s = np.sqrt((gt_c ** 2).sum()) / (np.sqrt((pred_c ** 2).sum()) + 1e-8)
    return s * pred_c + gt.mean(axis=0)


def scale_then_root_align(pred, gt, root_idx=0):
    """
    Resolve scale ambiguity then align by a single root joint.
    pred, gt : (n, 3).  Returns pred_aligned in same units as gt.
    Step 1 — scale using all joints centred at root.
    Step 2 — shift so pred root matches gt root.
    """
    pred_c = pred - pred[root_idx]
    gt_c   = gt   - gt[root_idx]
    s = np.sqrt((gt_c ** 2).sum()) / (np.sqrt((pred_c ** 2).sum()) + 1e-8)
    return s * pred_c + gt[root_idx]


# ── Metrics ───────────────────────────────────────────────────────────────────

def per_sample_metrics_3d(errors_mm, dfr_tau_mm):
    """errors_mm : (n,) per-joint 3D errors in mm."""
    return {
        'errors_mm': errors_mm,
        'mpjpe_mm':  float(errors_mm.mean()),
        'dfr_fail':  int(errors_mm.mean() > dfr_tau_mm),
    }


def summarise_3d(results, dfr_tau_mm=50.0, tau_min=20, tau_max=50):
    """
    Print MPJPE, PCK@tau_min, PCK@tau_max, AUC [0,1], and DFR.
    AUC is reported as a decimal in [0, 1] to match Table 1 in the paper.
    """
    n = len(results)
    if n == 0:
        print('No results to summarise.')
        return

    mpjpe = np.mean([r['mpjpe_mm'] for r in results])
    dfr   = sum(r['dfr_fail'] for r in results) / n * 100.0

    all_errors   = [r['errors_mm'] for r in results]
    total_joints = sum(len(e) for e in all_errors)
    taus         = np.arange(tau_min, tau_max + 1, dtype=float)

    pck_curve_frac = []
    for tau in taus:
        hits = sum(int((e < tau).sum()) for e in all_errors)
        pck_curve_frac.append(min(hits / total_joints, 1.0))

    # AUC normalised to [0, 1]:  integral / (tau_max - tau_min)
    auc = float(np.trapz(pck_curve_frac, taus)) / (tau_max - tau_min)

    print(f'\n{"─" * 50}')
    print(f'  Frames evaluated       : {n}')
    print(f'  MPJPE                  : {mpjpe:.2f} mm')
    print(f'  PCK @ {tau_min}mm              : {pck_curve_frac[0]*100:.2f} %')
    print(f'  PCK @ {tau_max}mm              : {pck_curve_frac[-1]*100:.2f} %')
    print(f'  AUC PCK({tau_min}-{tau_max}mm)       : {auc:.3f}  (paper format [0,1])')
    print(f'  DFR (τ_f={dfr_tau_mm:.0f}mm)       : {dfr:.2f} %')
    print(f'{"─" * 50}\n')


# ── Dexter+Object 3D ──────────────────────────────────────────────────────────

def run_dexter_3d(data_root, model, device, max_samples, dfr_tau_mm, perf=None):
    """
    3D evaluation on Dexter+Object (6 sequences).
    GT: first 5 entries of <SEQ>3D.txt = fingertips in mm.
    Invalid: z == 32001.
    Alignment: Procrustes scale + centroid translation on visible fingertips.
    """
    data_dir = os.path.join(data_root, 'dexter+object', 'data')
    results  = []
    total    = 0

    for seq in _DEXTER_SEQS:
        seq_dir      = os.path.join(data_dir, seq)
        anno_dir     = os.path.join(seq_dir, 'annotations')
        img_dir      = os.path.join(seq_dir, 'color')
        anno_2d_file = os.path.join(anno_dir, f'{seq}2D.txt')
        anno_3d_file = os.path.join(anno_dir, f'{seq}3D.txt')

        if not os.path.isfile(anno_3d_file):
            print(f'Warning: not found: {anno_3d_file}')
            continue

        with open(anno_2d_file) as f:
            lines_2d = f.readlines()
        with open(anno_3d_file) as f:
            lines_3d = f.readlines()

        img_files = sorted(glob.glob(os.path.join(img_dir, '*.bmp')))
        n_frames  = min(len(lines_2d), len(lines_3d), len(img_files))

        for i in tqdm(range(n_frames), desc=f'Dexter+Object 3D [{seq}]'):
            if total >= max_samples:
                break

            # Parse 3D GT: first 5 semicolon-separated entries are fingertips (mm)
            pairs_3d  = [p.strip() for p in lines_3d[i].split(';')][:5]
            gt_3d_all = np.array(
                [[float(v) for v in p.split(',')] for p in pairs_3d],
                dtype=np.float32
            )  # (5, 3) mm

            valid_mask = gt_3d_all[:, 2] != 32001   # z == 32001 → occluded
            if valid_mask.sum() == 0:
                continue
            gt_valid = gt_3d_all[valid_mask]         # (n_valid, 3) mm

            # 2D GT for crop center/scale
            pairs_2d   = [p.strip() for p in lines_2d[i].split(';')][:5]
            kp2d_all   = np.array(
                [[float(v) for v in p.split(',')] for p in pairs_2d],
                dtype=np.float32
            )  # (5, 2) px
            kp2d_valid = kp2d_all[valid_mask]

            img = cv2.imread(img_files[i])
            if img is None:
                continue

            tensor, _, _ = training_crop_and_transform(img, kp2d_valid, device)
            if perf is not None:
                _, xyz_pred, _, stage_times = predict_timed(model, tensor)
                perf.update(stage_times)
            else:
                with torch.no_grad():
                    xyz_pred = model(tensor)['xyz'].squeeze(0).cpu().numpy()  # (21, 3)

            # Predicted fingertips for visible GT joints
            snap_valid   = np.array(_FINGERTIP_SNAP)[valid_mask]
            pred_tips    = xyz_pred[snap_valid]                # (n_valid, 3)

            # Procrustes scale + centroid align
            pred_aligned = scale_then_centroid_align(pred_tips, gt_valid)
            errors_mm    = np.linalg.norm(pred_aligned - gt_valid, axis=1)
            results.append(per_sample_metrics_3d(errors_mm, dfr_tau_mm))
            total += 1

        if total >= max_samples:
            break

    return results


# ── EgoDexter 3D ──────────────────────────────────────────────────────────────

def run_egodexter_3d(data_root, model, device, max_samples, dfr_tau_mm, perf=None):
    """
    3D evaluation on EgoDexter (4 sequences).
    GT: annotation.txt_3D.txt — (x,y,z) per fingertip in mm.
    Invalid: (0, 0, 0) entries (occluded or unannotated).
    Alignment: Procrustes scale + centroid translation on visible fingertips.
    """
    data_dir = os.path.join(data_root, 'EgoDexter', 'data')
    results  = []
    total    = 0

    for seq in _EGODEXTER_SEQS:
        seq_dir   = os.path.join(data_dir, seq)
        anno_2d   = os.path.join(seq_dir, 'annotation.txt')
        anno_3d   = os.path.join(seq_dir, 'annotation.txt_3D.txt')
        img_dir   = os.path.join(seq_dir, 'color')

        if not os.path.isfile(anno_3d):
            print(f'Warning: not found: {anno_3d}')
            continue

        with open(anno_2d) as f:
            lines_2d = f.readlines()
        with open(anno_3d) as f:
            lines_3d = f.readlines()

        img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        n_frames  = min(len(lines_2d), len(lines_3d), len(img_files))

        for i in tqdm(range(n_frames), desc=f'EgoDexter 3D [{seq}]'):
            if total >= max_samples:
                break

            line_3d = lines_3d[i].strip()
            if not line_3d:
                continue

            # Parse 3D: up to 5 fingertips; (0,0,0) = occluded/unannotated
            pairs_3d    = [p.strip() for p in line_3d.split(';') if p.strip()]
            gt_3d_list  = []
            valid_list  = []
            for p in pairs_3d[:5]:
                vals = [float(v) for v in p.split(',')]
                # -0 parses as 0.0 in Python, so abs check handles "0, -0, 0"
                is_valid = not np.allclose(vals, 0.0)
                valid_list.append(is_valid)
                gt_3d_list.append(vals)
            # Pad to 5 if fewer entries in line
            while len(valid_list) < 5:
                valid_list.append(False)
                gt_3d_list.append([0.0, 0.0, 0.0])

            valid_mask = np.array(valid_list, dtype=bool)
            gt_3d_all  = np.array(gt_3d_list, dtype=np.float32)   # (5, 3) mm

            if valid_mask.sum() == 0:
                continue
            gt_valid = gt_3d_all[valid_mask]   # (n_valid, 3) mm

            # 2D GT for crop — use same validity mask to stay consistent
            pairs_2d        = [p.strip() for p in lines_2d[i].split(';') if p.strip()]
            kp2d_valid_list = []
            for j in range(5):
                if valid_mask[j] and j < len(pairs_2d):
                    vals2d = [float(v) for v in pairs_2d[j].split(',')]
                    if vals2d[0] >= 0:
                        kp2d_valid_list.append(vals2d)
            if not kp2d_valid_list:
                continue
            kp2d_valid = np.array(kp2d_valid_list, dtype=np.float32)

            img = cv2.imread(img_files[i])
            if img is None:
                continue

            tensor, _, _ = training_crop_and_transform(img, kp2d_valid, device)
            if perf is not None:
                _, xyz_pred, _, stage_times = predict_timed(model, tensor)
                perf.update(stage_times)
            else:
                with torch.no_grad():
                    xyz_pred = model(tensor)['xyz'].squeeze(0).cpu().numpy()   # (21, 3)

            snap_valid   = np.array(_FINGERTIP_SNAP)[valid_mask]
            pred_tips    = xyz_pred[snap_valid]                 # (n_valid, 3)
            pred_aligned = scale_then_centroid_align(pred_tips, gt_valid)
            errors_mm    = np.linalg.norm(pred_aligned - gt_valid, axis=1)
            results.append(per_sample_metrics_3d(errors_mm, dfr_tau_mm))
            total += 1

        if total >= max_samples:
            break

    return results


# ── RHD 3D ────────────────────────────────────────────────────────────────────

def run_rhd_3d(data_root, model, device, max_samples, dfr_tau_mm, split='eval', perf=None):
    """
    3D evaluation on RHD (all 21 joints).
    GT: xyz in meters from pickle → converted to mm.
    Alignment: Procrustes scale + root (wrist, joint 0) translation.
    """
    if split == 'train':
        data_dir  = os.path.join(data_root, 'RHD', 'RHD_published_v2', 'training')
        anno_file = os.path.join(data_dir, 'anno_training.pickle')
        desc      = 'RHD train 3D'
    else:
        data_dir  = os.path.join(data_root, 'RHD', 'RHD_published_v2', 'evaluation')
        anno_file = os.path.join(data_dir, 'anno_evaluation.pickle')
        desc      = 'RHD eval 3D'

    img_dir = os.path.join(data_dir, 'color')

    with open(anno_file, 'rb') as f:
        anno = pickle.load(f, encoding='latin1')

    img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    n         = min(max_samples, len(img_files), len(anno))
    results   = []

    for i in tqdm(range(n), desc=desc):
        img = cv2.imread(img_files[i])
        if img is None:
            continue
        img_w = img.shape[1]

        uv_vis    = anno[i]['uv_vis']            # (42, 3): [u, v, visibility]
        xyz_anno  = anno[i]['xyz']               # (42, 3): meters
        vis_left  = uv_vis[:21, 2].sum()
        vis_right = uv_vis[21:, 2].sum()

        if vis_right >= vis_left:
            kp2d_gt  = uv_vis[21:, :2].copy()
            gt_xyz   = xyz_anno[21:].copy() * 1000.0   # → mm
            img_eval = img
        else:
            kp2d_gt       = uv_vis[:21, :2].copy()
            kp2d_gt[:, 0] = img_w - kp2d_gt[:, 0]    # mirror x
            gt_xyz        = xyz_anno[:21].copy() * 1000.0
            gt_xyz[:, 0]  = -gt_xyz[:, 0]             # mirror x in camera space
            img_eval      = cv2.flip(img, 1)

        # Reorder RHD (tip→MCP) → SNAP (MCP→tip)
        kp2d_gt = kp2d_gt[_RHD_TO_SNAP]
        gt_xyz  = gt_xyz[_RHD_TO_SNAP]            # (21, 3) mm

        tensor, _, _ = training_crop_and_transform(img_eval, kp2d_gt, device)
        if perf is not None:
            _, xyz_pred, _, stage_times = predict_timed(model, tensor)
            perf.update(stage_times)
        else:
            with torch.no_grad():
                xyz_pred = model(tensor)['xyz'].squeeze(0).cpu().numpy()   # (21, 3)

        # Procrustes scale + root alignment
        pred_aligned = scale_then_root_align(xyz_pred, gt_xyz, root_idx=0)
        errors_mm    = np.linalg.norm(pred_aligned - gt_xyz, axis=1)   # (21,) mm
        results.append(per_sample_metrics_3d(errors_mm, dfr_tau_mm))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='3D evaluation of DetNet')
    parser.add_argument('--dataset', choices=['dexter', 'egodexter', 'rhd'],
                        required=True)
    parser.add_argument('--data_root', type=str, default='C:/Imperial/FYP Data',
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--checkpoint', type=str,
                        default='new_check_point/ckp_detnet_83.pth')
    parser.add_argument('--model_type', choices=['detnet', 'light'], default='detnet',
                        help='Model architecture: detnet (original) or light (compressed)')
    parser.add_argument('--max_samples', type=int, default=999_999)
    parser.add_argument('--tau_min', type=int, default=20,
                        help='Lower PCK threshold in mm (default: 20)')
    parser.add_argument('--tau_max', type=int, default=50,
                        help='Upper PCK threshold in mm (default: 50)')
    parser.add_argument('--dfr_tau_mm', type=float, default=50.0,
                        help='DFR failure threshold in mm (default: 50)')
    parser.add_argument('--split', choices=['eval', 'train'], default='eval',
                        help='RHD split (ignored for dexter/egodexter)')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    model = load_model(args.checkpoint, device, model_type=args.model_type)

    flops_info = compute_flops(model, device)
    mem_info   = get_memory_stats(model, device)
    perf       = LatencyAccumulator(mode='3d')

    if args.dataset == 'dexter':
        results = run_dexter_3d(
            args.data_root, model, device,
            args.max_samples, args.dfr_tau_mm, perf=perf,
        )
    elif args.dataset == 'egodexter':
        results = run_egodexter_3d(
            args.data_root, model, device,
            args.max_samples, args.dfr_tau_mm, perf=perf,
        )
    else:
        results = run_rhd_3d(
            args.data_root, model, device,
            args.max_samples, args.dfr_tau_mm,
            split=args.split, perf=perf,
        )

    summarise_3d(results,
                 dfr_tau_mm=args.dfr_tau_mm,
                 tau_min=args.tau_min,
                 tau_max=args.tau_max)
    print_perf_report(perf, flops_info, mem_info, device)


if __name__ == '__main__':
    main()
