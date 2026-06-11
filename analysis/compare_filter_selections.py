"""
Filter-level comparison: at each pruning ratio, which output filters does L1
keep vs which does Taylor keep? Per-layer counts don't reveal anything because
both methods use per-layer uniform ratio allocation (global_pruning=False in
prune_l1.py / prune_taylor.py) — only the *identity* of dropped filters varies.

Approach: replay both pruners in torch_pruning's interactive mode (yields
groups without applying them), capture the trigger-conv kept/dropped index
sets per layer, then compute per-layer Jaccard overlap, swap count, and
Spearman rank correlation of importance scores. Save full per-ratio per-layer
stats as JSON and a summary figure.

Nothing about filter selection is saved by the prune scripts, but the
ingredients are recoverable:
  - L1 importance is |W|_1 per output channel — baseline weights are enough.
  - Taylor importance is |W * grad| per channel — uses the cached calibration
    gradients in `taylor_calibration_grads.pt` (same file prune_taylor.py
    persists at the end of its first run).

Verification: after each capture, the resulting per-layer kept counts MUST
match the saved pruned architecture in pruned_architectures/. If they don't,
the reconstruction is buggy and the whole analysis is suspect — the script
aborts.

Usage:
  python analysis\\compare_filter_selections.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Windows OMP duplicate-lib

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')  # cp1252 console can't print ✓/✗
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import copy
import json

import cv2  # noqa: F401 — keep ABOVE torch (Windows DLL load order)
import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils.misc as misc
from model.detnet import detnet

CHECKPOINT = 'new_check_point/ckp_detnet_71.pth'
GRADS_FILE = 'taylor_calibration_grads.pt'
ARCH_DIR   = 'pruned_architectures'
OUT_JSON   = 'analysis/filter_selection_comparison.json'
OUT_FIG    = 'figures/fig_filter_selection_overlap'
RATIOS     = [0.10, 0.25, 0.40, 0.55, 0.70]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'DejaVu Serif'],
    'font.size':   9,
})


# ── Model + grad loading (verbatim contracts from prune_l1.py / prune_taylor.py) ──

def load_fresh_model(ckp_path):
    """Same as prune_l1.py:36-45 / prune_taylor.py:68-77."""
    m = detnet()
    state = misc.clean_state_dict(torch.load(ckp_path, map_location='cpu'))
    ms = m.state_dict()
    matched = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
    m.load_state_dict({**ms, **matched}, strict=False)
    return m


def load_taylor_grads(grads_file=GRADS_FILE):
    """Load cached calibration gradients (prune_taylor.py:139-142)."""
    if not os.path.isfile(grads_file):
        sys.exit(f"{grads_file} not found. Run prune_taylor.py once to "
                 f"generate it (or copy from a machine that has it).")
    ckpt = torch.load(grads_file, map_location='cpu', weights_only=False)
    return ckpt['grads'], int(ckpt.get('n_images', 0))


def assign_grads(model, saved_grads):
    """Same restore-grads-after-deepcopy idiom as prune_taylor.py:196-198.
    deepcopy drops .grad on Parameters; this puts it back."""
    n = 0
    for name, p in model.named_parameters():
        if name in saved_grads:
            p.grad = saved_grads[name].clone()
            n += 1
    return n


# ── Selection capture via interactive pruning ────────────────────────────────

def capture_selections(model, importance, ratio, example_input):
    """Run pruner.step(interactive=True) and record per-layer dropped index
    sets WITHOUT applying. Only output-channel pruning on Conv2d triggers is
    recorded — those are the actual filter-selection decisions. Downstream
    in-channel cascades are mechanical and skipped.

    Returns {layer_name: {'out_channels': N, 'kept': [...], 'dropped': [...],
                          'scores': [...] of len N}}.
    """
    name_of = {id(m): n for n, m in model.named_modules()}

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_input,
        importance=importance,
        pruning_ratio=ratio,
        global_pruning=False,
        ignored_layers=[model.hmap_0.prediction],
    )
    DG = pruner.DG

    selections = {}
    for group in pruner.step(interactive=True):
        trig_dep, trig_idxs = group[0]
        module = trig_dep.target.module
        handler = trig_dep.handler
        # Only filter-selection decisions (output-channel pruning on Conv2d).
        if not isinstance(module, nn.Conv2d):
            continue
        if not DG.is_out_channel_pruning_fn(handler):
            continue

        layer_name = name_of.get(id(module), repr(module))
        n_total = module.out_channels
        dropped = sorted({int(i) for i in trig_idxs})
        kept = sorted(set(range(n_total)) - set(dropped))

        # Per-channel importance scores from the same callable the pruner used.
        # Craft a full-index group via DG so imp() returns scores for all N
        # channels (group docstring: "for importance estimation we should craft
        # a group with full indices").
        full_group = DG.get_pruning_group(module, handler, list(range(n_total)))
        scores = importance(full_group).detach().cpu().numpy().astype(float).tolist()

        selections[layer_name] = {
            'out_channels': int(n_total),
            'kept': kept,
            'dropped': dropped,
            'scores': scores,
        }
    return selections


# ── Verification: reconstructed kept counts must match saved _noft.pth ───────

def verify_architecture_match(method, ratio_pct, reconstructed):
    """Compare reconstructed per-layer kept counts against the saved pruned
    architecture's Conv2d.out_channels. Returns (ok, mismatches:list[str])."""
    path = os.path.join(ARCH_DIR,
                        f'detnet_{method}_{ratio_pct:02d}pct_ep71_noft.pth')
    if not os.path.isfile(path):
        return False, [f'{path} not found']
    saved_model = torch.load(path, map_location='cpu', weights_only=False)
    saved_chans = {n: m.out_channels for n, m in saved_model.named_modules()
                   if isinstance(m, nn.Conv2d)}
    mismatches = []
    for name, info in reconstructed.items():
        expected = len(info['kept'])
        actual = saved_chans.get(name)
        if actual != expected:
            mismatches.append(
                f'{name}: reconstructed kept={expected}, saved={actual}')
    return len(mismatches) == 0, mismatches


# ── Comparison statistics ────────────────────────────────────────────────────

def compare(l1_sel, taylor_sel):
    """Per-layer Jaccard / swap count / Spearman + per-ratio aggregates."""
    per_layer = []
    common = sorted(set(l1_sel) & set(taylor_sel))
    for name in common:
        L, T = l1_sel[name], taylor_sel[name]
        kL, kT = set(L['kept']), set(T['kept'])
        union = kL | kT
        inter = kL & kT
        jaccard = len(inter) / len(union) if union else 1.0
        swaps = len(kL ^ kT) // 2
        sL, sT = np.array(L['scores']), np.array(T['scores'])
        rho = float(spearmanr(sL, sT).statistic) if len(sL) > 1 else 1.0
        per_layer.append({
            'name': name,
            'n_total': L['out_channels'],
            'n_kept': len(L['kept']),
            'jaccard': float(jaccard),
            'swaps': int(swaps),
            'spearman': rho,
        })
    jacc = [r['jaccard'] for r in per_layer]
    worst = min(per_layer, key=lambda r: r['jaccard']) if per_layer else None
    return {
        'per_layer': per_layer,
        'mean_jaccard': float(np.mean(jacc)) if jacc else 0.0,
        'min_jaccard':  float(np.min(jacc))  if jacc else 0.0,
        'max_jaccard':  float(np.max(jacc))  if jacc else 1.0,
        'total_swaps':  int(sum(r['swaps'] for r in per_layer)),
        'mean_spearman': float(np.mean([r['spearman'] for r in per_layer]))
                        if per_layer else 0.0,
        'worst_layer': worst,
    }


# ── Figure ───────────────────────────────────────────────────────────────────

def render_figure(per_ratio_stats, out_base):
    """X = pruning ratio %, Y = per-layer Jaccard. Faint per-layer trajectories,
    bold mean and dashed min lines across ratios."""
    ratios = sorted(per_ratio_stats.keys())
    x = [int(r * 100) for r in ratios]

    # Collect per-layer trajectories
    layer_names = [r['name']
                   for r in per_ratio_stats[ratios[0]]['comparison']['per_layer']]
    trajectories = {name: [] for name in layer_names}
    for r in ratios:
        by_name = {row['name']: row['jaccard']
                   for row in per_ratio_stats[r]['comparison']['per_layer']}
        for name in layer_names:
            trajectories[name].append(by_name.get(name, np.nan))

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)

    for name, ys in trajectories.items():
        ax.plot(x, ys, color='lightgray', linewidth=0.5, alpha=0.6, zorder=1)

    means = [per_ratio_stats[r]['comparison']['mean_jaccard'] for r in ratios]
    mins  = [per_ratio_stats[r]['comparison']['min_jaccard']  for r in ratios]

    ax.plot(x, means, color='#1f77b4', linewidth=2.2, marker='o',
            markersize=7, label='mean across layers', zorder=3)
    ax.plot(x, mins,  color='#d62728', linewidth=1.6, marker='s',
            markersize=5, linestyle='--', label='min (worst-overlap layer)',
            zorder=2)

    ax.set_xlabel('Pruning ratio (%)')
    ax.set_ylabel('Per-layer Jaccard overlap of kept filters\n(L1 vs Taylor)')
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='lower left', framealpha=0.95)
    ax.set_title('L1 vs Taylor: how much do they agree on which filters to keep?',
                 fontsize=10, pad=8)

    # Spearman annotation — constant across ratios (per-channel ranking is a layer
    # property, not a ratio property), so one number summarises it.
    mean_rho = float(np.mean([per_ratio_stats[r]['comparison']['mean_spearman']
                              for r in ratios]))
    ax.text(0.97, 0.97,
            f'Mean Spearman ρ ≈ {mean_rho:+.2f}\n'
            f'(per-channel rankings,\nconstant across ratios)',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#666666', linewidth=0.8))

    png, pdf = out_base + '.png', out_base + '.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {png}')
    print(f'Wrote {pdf}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    example = torch.randn(1, 3, 128, 128)

    print(f'Loading baseline: {CHECKPOINT}')
    base_model = load_fresh_model(CHECKPOINT)
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f'  {n_params:,} parameters')

    print(f'Loading Taylor calibration gradients: {GRADS_FILE}')
    taylor_grads, n_cal_images = load_taylor_grads()
    print(f'  Loaded grads for {len(taylor_grads)} parameters '
          f'(calibrated on {n_cal_images:,} images).')
    print()

    per_ratio_stats = {}
    for ratio in RATIOS:
        rp = int(ratio * 100)
        print(f"{'=' * 64}\nRatio {rp}%\n{'=' * 64}")

        # L1 — no grads needed
        m_l1 = copy.deepcopy(base_model)
        l1_imp = tp.importance.MagnitudeImportance(p=1)
        l1_sel = capture_selections(m_l1, l1_imp, ratio, example)
        print(f'  L1     : captured {len(l1_sel)} layer decisions')

        # Taylor — restore grads onto the deepcopy
        m_t = copy.deepcopy(base_model)
        n_assigned = assign_grads(m_t, taylor_grads)
        print(f'  Taylor : {n_assigned} grads assigned, ', end='')
        t_imp = tp.importance.TaylorImportance()
        t_sel = capture_selections(m_t, t_imp, ratio, example)
        print(f'captured {len(t_sel)} layer decisions')

        # Architecture-match check (HARD verification)
        l1_ok, l1_mm = verify_architecture_match('l1', rp, l1_sel)
        t_ok, t_mm   = verify_architecture_match('taylor', rp, t_sel)
        if not l1_ok:
            print(f'  ✗ L1 architecture-match FAILED:')
            for s in l1_mm[:5]:
                print(f'      {s}')
            sys.exit(1)
        if not t_ok:
            print(f'  ✗ Taylor architecture-match FAILED:')
            for s in t_mm[:5]:
                print(f'      {s}')
            sys.exit(1)
        print(f'  ✓ architecture-match OK for both methods')

        cmp = compare(l1_sel, t_sel)
        wl = cmp['worst_layer']
        print(f"  mean Jaccard = {cmp['mean_jaccard']:.3f}   "
              f"min = {cmp['min_jaccard']:.3f} ({wl['name']})   "
              f"total swaps = {cmp['total_swaps']}   "
              f"mean Spearman = {cmp['mean_spearman']:+.3f}")
        print()

        per_ratio_stats[ratio] = {
            'comparison': cmp,
            'l1_arch_match': l1_ok,
            'taylor_arch_match': t_ok,
        }

    # Save JSON
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    out = {f"{int(r * 100):02d}": per_ratio_stats[r] for r in RATIOS}
    out['_summary'] = {
        'ratios_pct': [int(r * 100) for r in RATIOS],
        'mean_jaccard':   {f"{int(r * 100):02d}":
                           per_ratio_stats[r]['comparison']['mean_jaccard']
                           for r in RATIOS},
        'mean_spearman':  {f"{int(r * 100):02d}":
                           per_ratio_stats[r]['comparison']['mean_spearman']
                           for r in RATIOS},
        'min_jaccard':    {f"{int(r * 100):02d}":
                           per_ratio_stats[r]['comparison']['min_jaccard']
                           for r in RATIOS},
        'worst_layer':    {f"{int(r * 100):02d}":
                           per_ratio_stats[r]['comparison']['worst_layer']['name']
                           for r in RATIOS},
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {OUT_JSON}')

    # Render figure
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    render_figure(per_ratio_stats, OUT_FIG)

    print('\nDone.')


if __name__ == '__main__':
    main()
