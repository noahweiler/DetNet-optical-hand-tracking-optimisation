"""
Evaluate DetNet on one or more datasets, plot the PCK(τ) curve for each
dataset on a single figure, and print a summary metrics table (2D / pixel-space).

Usage:
  python plot_metrics_2d.py --datasets freihand rhd gan \\
      --data_root "path/to/FYP data" --output pck_2d.png
"""
import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_detnet import (
    load_model,
    run_freihand,
    run_ganerated,
    run_rhd,
    run_dexter,
    run_egodexter,
)

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_LABELS = {
    'freihand':  'FreiHAND',
    'gan':       'GANeratedHands',
    'rhd':       'RHD',
    'dexter':    'Dexter+Object',
    'egodexter': 'EgoDexter',
}

# One distinct colour per dataset (colour-blind-friendly palette)
DATASET_COLOURS = {
    'freihand':  '#e41a1c',   # red
    'gan':       '#377eb8',   # blue
    'rhd':       '#4daf4a',   # green
    'dexter':    '#984ea3',   # purple
    'egodexter': '#ff7f00',   # orange
}


# ── Metric computation — 2D ───────────────────────────────────────────────────

def compute_pck_curve(results, tau_min, tau_max):
    """
    Compute PCK(τ) from 2D results (errors in pixels).
    Returns (taus, pck_curve %, auc %).
    """
    all_errors   = [r['errors_px'] for r in results]
    total_joints = sum(len(e) for e in all_errors)
    taus         = np.arange(tau_min, tau_max + 1, dtype=float)
    pck_curve    = []
    for tau in taus:
        hits = sum(int((e < tau).sum()) for e in all_errors)
        pck_curve.append(min(hits / total_joints * 100.0, 100.0))
    auc = float(np.trapz(pck_curve, taus)) / (tau_max - tau_min)
    return taus, np.array(pck_curve), auc


def compute_summary(results, tau_min, tau_max):
    """Compute all 2D metrics for one dataset and return as a dict."""
    n     = len(results)
    mpjpe = float(np.mean([r['mpjpe']  for r in results]))
    dfr   = sum(r['dfr_fail']           for r in results) / n * 100.0
    taus, pck_curve, auc = compute_pck_curve(results, tau_min, tau_max)
    return {
        'n':         n,
        'mpjpe':     mpjpe,
        'pck_min':   float(pck_curve[0]),
        'pck_max':   float(pck_curve[-1]),
        'auc':       auc,
        'dfr':       dfr,
        'taus':      taus,
        'pck_curve': pck_curve,
    }


# ── Dataset helpers ───────────────────────────────────────────────────────────

def get_image_size(name, data_root, split='eval', gan_subset='all'):
    """
    Read one sample image from the dataset and return its (H, W) in pixels.
    Returns None if no image can be found at the expected path.
    """
    r         = data_root
    gan_sub   = 'noObject' if gan_subset == 'all' else gan_subset
    split_dir = 'evaluation' if split == 'eval' else 'training'
    patterns  = {
        'freihand':  os.path.join(r, 'FreiHAND', split_dir, 'rgb', '*.jpg'),
        'gan':       os.path.join(r, 'GANeratedHands_Release', 'data',
                                  gan_sub, '*', '*_color_composed.png'),
        'rhd':       os.path.join(r, 'RHD', split_dir, 'color', '*.png'),
        'dexter':    os.path.join(r, 'dexter+object', 'data', 'Grasp1', 'color', '*.bmp'),
        'egodexter': os.path.join(r, 'EgoDexter', 'data', 'Desk', 'color', '*.png'),
    }
    pattern = patterns.get(name)
    if pattern is None:
        return None

    files = sorted(glob.glob(pattern))
    # FreiHAND eval images may be .jpg or .png
    if not files and name == 'freihand':
        files = sorted(glob.glob(pattern.replace('*.jpg', '*.png')))
    if not files:
        return None

    img = cv2.imread(files[0])
    if img is None:
        return None
    return img.shape[0], img.shape[1]   # (H, W)


# ── Dataset runners ────────────────────────────────────────────────────────────

def run_dataset(name, args, model, device):
    """Dispatch to the correct 2D run_* function."""
    common = dict(data_root=args.data_root, model=model, device=device,
                  max_samples=args.max_samples, dfr_tau=args.dfr_tau)
    if   name == 'freihand':  return run_freihand( **common, split=args.split)
    elif name == 'gan':       return run_ganerated(**common, subset=args.gan_subset)
    elif name == 'rhd':       return run_rhd(      **common, split=args.split)
    elif name == 'dexter':    return run_dexter(   **common)
    elif name == 'egodexter': return run_egodexter(**common)
    else: raise ValueError(f'Unknown dataset: {name}')


# ── Output ────────────────────────────────────────────────────────────────────

def print_table(summaries, tau_min, tau_max):
    """Print a formatted 2D metrics table."""
    unit = 'px'

    # Column widths: Dataset, Image size, Frames, MPJPE, PCK@min, PCK@max, AUC, DFR
    cw = [20, 12, 8, 12, 12, 12, 10, 9]
    sep = '─' * (sum(cw) + len(cw) - 1)

    header = (f"{'Dataset':<{cw[0]}} {'Image size':>{cw[1]}} {'Frames':>{cw[2]}} "
              f"{'MPJPE':>{cw[3]}} "
              f"{f'PCK@{tau_min}{unit}':>{cw[4]}} "
              f"{f'PCK@{tau_max}{unit}':>{cw[5]}} "
              f"{'AUC (%)':>{cw[6]}} "
              f"{'DFR (%)':>{cw[7]}}")

    print(f'\n{sep}')
    print(header)
    print(sep)
    for name, m in summaries.items():
        label    = DATASET_LABELS.get(name, name)
        hw       = m.get('img_hw')
        size_str = f'{hw[1]}×{hw[0]}' if hw else 'N/A'   # W×H
        print(f"{label:<{cw[0]}} {size_str:>{cw[1]}} {m['n']:>{cw[2]}} "
              f"{m['mpjpe']:>{cw[3]}.4f} "
              f"{m['pck_min']:>{cw[4]}.2f} "
              f"{m['pck_max']:>{cw[5]}.2f} "
              f"{m['auc']:>{cw[6]}.2f} "
              f"{m['dfr']:>{cw[7]}.2f}")
    print(f'{sep}\n')


def plot_pck_curves(summaries, tau_min, tau_max, output_path):
    """
    Plot one PCK(τ) curve per dataset (2D / pixels). AUC is shown in each
    legend entry. Saves the figure to output_path.
    """
    unit = 'pixels'

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, m in summaries.items():
        label = f"{DATASET_LABELS.get(name, name)}  (AUC = {m['auc']:.1f} %)"
        ax.plot(m['taus'], m['pck_curve'],
                color=DATASET_COLOURS.get(name, None), linewidth=2, label=label)

    ax.set_xlabel(f'Threshold τ ({unit})', fontsize=12)
    ax.set_ylabel('Accuracy (AUC,%)', fontsize=12)
    ax.set_title('PCK(τ) Curve — DetNet 2D Evaluation', fontsize=13)
    ax.set_xlim(tau_min, tau_max)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='lower right')
    fig.tight_layout()

    fig.savefig(output_path, dpi=800)
    print(f'Plot saved → {output_path}')
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot PCK(τ) curves and metrics table for DetNet 2D evaluation'
    )
    parser.add_argument('--datasets', nargs='+',
                        choices=['freihand', 'gan', 'rhd', 'dexter', 'egodexter'],
                        required=True,
                        help='One or more datasets to evaluate and plot')
    parser.add_argument('--data_root', type=str,
                        default='C:/Imperial/FYP Data',
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--checkpoint', type=str,
                        default='new_check_point/ckp_detnet_83.pth',
                        help='Path to DetNet checkpoint (.pth)')
    parser.add_argument('--max_samples', type=int, default=999_999,
                        help='Max frames per dataset (default: unlimited)')
    parser.add_argument('--tau_min', type=int, default=20,
                        help='Lower PCK threshold in pixels (default: 20)')
    parser.add_argument('--tau_max', type=int, default=50,
                        help='Upper PCK threshold in pixels (default: 50)')
    parser.add_argument('--dfr_tau', type=float, default=0.20,
                        help='DFR failure threshold as fraction of image size (default: 0.20)')
    parser.add_argument('--split', choices=['eval', 'train'], default='eval',
                        help='Dataset split for FreiHAND and RHD (default: eval)')
    parser.add_argument('--gan_subset',
                        choices=['all', 'noObject', 'withObject'], default='all',
                        help='GANeratedHands subset (default: all)')
    parser.add_argument('--output', type=str, default='pck_curves.png',
                        help='Output path for the saved plot (default: pck_curves.png)')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    model = load_model(args.checkpoint, device)

    summaries = {}
    for name in args.datasets:
        print(f'\n── Evaluating {DATASET_LABELS.get(name, name)} ──')

        results = run_dataset(name, args, model, device)
        if results:
            summaries[name] = compute_summary(
                results, args.tau_min, args.tau_max
            )
            summaries[name]['img_hw'] = get_image_size(
                name, args.data_root, split=args.split, gan_subset=args.gan_subset
            )
        else:
            print(f'  No results for {name} — skipping.')

    if not summaries:
        print('No results to plot.')
        return

    print_table(summaries, args.tau_min, args.tau_max)
    plot_pck_curves(summaries, args.tau_min, args.tau_max, args.output)


if __name__ == '__main__':
    main()
