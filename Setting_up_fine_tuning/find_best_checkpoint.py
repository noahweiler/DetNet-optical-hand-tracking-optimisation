"""
Find the best DetNet checkpoint by 2D PCK AUC, averaged across all five
evaluation datasets.

For each (checkpoint, dataset) pair, runs the existing 2D evaluation pipeline
from evaluate_detnet.py and computes AUC over τ ∈ [20, 50] pixels.
The checkpoint with the highest mean AUC across the five datasets is reported.

Run from project root with the conda env active and dataset folders present
under --data_root:
    python find_best_checkpoint.py
or, with a non-default data root:
    python find_best_checkpoint.py --data_root "D:/data"
"""
import argparse
import os
import time

import torch

from evaluate_detnet import (
    load_model,
    run_freihand,
    run_ganerated,
    run_rhd,
    run_dexter,
    run_egodexter,
)
from plot_metrics_2d import compute_summary

CHECKPOINT_DIR = 'new_check_point'
CHECKPOINTS = [
    ('ep68',  'ckp_detnet_68.pth'),
    ('ep71',  'ckp_detnet_71.pth'),
    ('ep83',  'ckp_detnet_83.pth'),
    ('ep101', 'ckp_detnet_101.pth'),
    ('ep106', 'ckp_detnet_106.pth'),
]
DATASETS = ['freihand', 'rhd', 'gan', 'dexter', 'egodexter']
DATASET_LABELS = {
    'freihand':  'FreiHAND',
    'rhd':       'RHD',
    'gan':       'GAN',
    'dexter':    'Dexter',
    'egodexter': 'EgoDexter',
}

# Per-dataset hard caps. GAN has ~1M images; 3000 is plenty for stable AUC.
DATASET_CAPS = {
    'gan': 3000,
}


def run_one(name, data_root, model, device, max_samples, dfr_tau,
            split, gan_subset):
    """Dispatch to the right 2D run_* function — same logic as plot_metrics_2d."""
    n = min(max_samples, DATASET_CAPS.get(name, max_samples))
    common = dict(data_root=data_root, model=model, device=device,
                  max_samples=n, dfr_tau=dfr_tau)
    if   name == 'freihand':  return run_freihand( **common, split=split)
    elif name == 'gan':       return run_ganerated(**common, subset=gan_subset)
    elif name == 'rhd':       return run_rhd(      **common, split=split)
    elif name == 'dexter':    return run_dexter(   **common)
    elif name == 'egodexter': return run_egodexter(**common)
    raise ValueError(f'Unknown dataset: {name}')


def main():
    parser = argparse.ArgumentParser(
        description='Find the best DetNet checkpoint by mean 2D AUC across all datasets'
    )
    parser.add_argument('--data_root', type=str,
                        default='C:/Imperial/FYP Data',
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--max_samples', type=int, default=999_999,
                        help='Max frames per dataset (default: unlimited)')
    parser.add_argument('--tau_min', type=int, default=20)
    parser.add_argument('--tau_max', type=int, default=50)
    parser.add_argument('--dfr_tau', type=float, default=0.20)
    parser.add_argument('--split', choices=['eval', 'train'], default='eval')
    parser.add_argument('--gan_subset',
                        choices=['all', 'noObject', 'withObject'], default='all')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # auc_table[ckpt_name][dataset_name] = AUC %
    auc_table = {name: {} for name, _ in CHECKPOINTS}

    for ckpt_name, ckpt_file in CHECKPOINTS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
        if not os.path.isfile(ckpt_path):
            print(f'!! {ckpt_name}: checkpoint not found at {ckpt_path} — skipping')
            continue

        print(f'═══ {ckpt_name}  ({ckpt_path}) ═══')
        t_ckpt = time.perf_counter()
        model = load_model(ckpt_path, device)

        for ds in DATASETS:
            t_ds = time.perf_counter()
            print(f'  → {DATASET_LABELS[ds]} ...', flush=True)
            try:
                results = run_one(
                    ds, args.data_root, model, device,
                    args.max_samples, args.dfr_tau,
                    args.split, args.gan_subset,
                )
            except FileNotFoundError as e:
                print(f'    skipped — dataset files missing: {e}')
                continue

            if not results:
                print(f'    no results — skipping')
                continue

            summary = compute_summary(results, args.tau_min, args.tau_max)
            auc_table[ckpt_name][ds] = summary['auc']
            elapsed = time.perf_counter() - t_ds
            print(f'    AUC = {summary["auc"]:6.2f} %   '
                  f'(n={summary["n"]}, {elapsed:.1f}s)')

        elapsed_ckpt = time.perf_counter() - t_ckpt
        print(f'   total: {elapsed_ckpt:.1f}s\n')

    # ── Final table ───────────────────────────────────────────────────────────
    print('=' * 80)
    print('Per-checkpoint AUC (%)  —  PCK τ ∈ [{}, {}] pixels'
          .format(args.tau_min, args.tau_max))
    print('=' * 80)
    header = f'{"":>8} ' + ''.join(f'{DATASET_LABELS[d]:>11}' for d in DATASETS) + f'{"Mean":>11}'
    print(header)
    print('-' * len(header))

    means = {}
    for ckpt_name, _ in CHECKPOINTS:
        row = auc_table[ckpt_name]
        scored = [row[d] for d in DATASETS if d in row]
        if not scored:
            print(f'{ckpt_name:>8} ' + ' '.join(f'{"-":>10}' for _ in DATASETS) + f'{"-":>11}')
            continue
        cells = [f'{row[d]:>10.2f}' if d in row else f'{"-":>10}' for d in DATASETS]
        mean = sum(scored) / len(scored)
        means[ckpt_name] = mean
        print(f'{ckpt_name:>8} ' + ' '.join(cells) + f'{mean:>10.2f}')

    print('=' * 80)

    if not means:
        print('No checkpoints produced any results — check --data_root and dataset folders.')
        return

    best = max(means, key=means.get)
    best_path = os.path.join(CHECKPOINT_DIR,
                             dict(CHECKPOINTS)[best])
    print(f'\nBest checkpoint by mean AUC: {best}  '
          f'(mean = {means[best]:.2f} %)')
    print(f'Path: {best_path}')


if __name__ == '__main__':
    main()
