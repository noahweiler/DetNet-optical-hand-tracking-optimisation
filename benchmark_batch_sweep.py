"""
benchmark_batch_sweep.py — Inference latency / throughput vs BATCH SIZE,
for one model, on GPU and CPU. Makes the curves that explain the
launch-bound (GPU, bs=1) vs compute-bound (CPU, or GPU at large batch)
behaviour.

Reuses benchmark_inference.py's exact timing core (CUDA events on GPU,
perf_counter on CPU, warm-up discarded, fixed synthetic input — no data
pipeline), so every point is measured the same controlled way; only the
batch size and device change.

Produces, for the chosen model:
  figures/fig_batch_sweep_<stem>.pdf / .png   — 2 panels:
      (a) latency per frame (ms) vs batch size   — GPU drops then plateaus
          (launch overhead amortised); CPU ≈ flat (already compute-bound)
      (b) throughput (fps) vs batch size         — GPU rises then saturates
  benchmark_results/batch_sweep_<stem>.{json,csv}  — the raw numbers

Usage:
  python benchmark_batch_sweep.py                                  # baseline, GPU+CPU
  python benchmark_batch_sweep.py --checkpoint checkpoints_finetuned/l1_70pct/ft_l1_70pct_rhdbest.pth \\
      --arch pruned_architectures/detnet_l1_70pct_ep71_noft.pth
  python benchmark_batch_sweep.py --batches 1 2 4 8 16 32 64 128 256 --devices cuda
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import argparse
import csv
import json
from datetime import datetime

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Identical timing core + model loading as the single-point benchmark.
from benchmark_inference import benchmark_cuda, benchmark_cpu
from evaluate_detnet import load_model, count_params, compute_flops, INPUT_SIZE
from plot_pruning_results import COLORS

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'DejaVu Serif'],
    'font.size':   9,
})

DEV_STYLE = {
    'cuda': {'color': COLORS['blue'],   'marker': 'o', 'label': 'GPU (CUDA events)'},
    'cpu':  {'color': COLORS['orange'], 'marker': 's', 'label': 'CPU (perf_counter)'},
}


def sweep_device(checkpoint, arch, dev_type, batches, warmup, iters, input_size):
    """Benchmark `checkpoint` on one device across `batches`. Returns a list of
    per-batch dicts. Stops early on CUDA OOM (larger batches would also OOM)."""
    device = torch.device('cuda:0' if dev_type == 'cuda' else 'cpu')
    model  = load_model(checkpoint, device, arch_path=arch)
    model.eval()
    if dev_type == 'cuda':
        torch.backends.cudnn.benchmark = True   # stable kernels per shape

    rows = []
    for b in batches:
        torch.manual_seed(0)
        try:
            x = torch.randn(b, 3, input_size, input_size,
                            device=device, dtype=torch.float32)
            t = (benchmark_cuda if dev_type == 'cuda' else benchmark_cpu)(
                model, x, warmup, iters)
        except RuntimeError as e:                # OOM / alloc failure
            print(f'  [{dev_type}] batch {b}: skipped ({type(e).__name__}); '
                  f'stopping sweep here.')
            break
        ms_b   = float(t.mean())
        ms_bsd = float(t.std())
        rows.append({
            'device':           dev_type,
            'batch':            b,
            'ms_per_batch':     ms_b,
            'ms_per_batch_std': ms_bsd,
            'ms_per_frame':     ms_b / b,
            'ms_per_frame_std': ms_bsd / b,
            'fps':              b * 1000.0 / ms_b if ms_b > 0 else float('nan'),
        })
        print(f'  [{dev_type}] batch {b:>4}: {ms_b/b:7.3f} ms/frame  '
              f'{b*1000.0/ms_b:8.1f} fps')
        del x
        if dev_type == 'cuda':
            torch.cuda.empty_cache()
    return rows


def make_figure(per_dev, meta, out_base):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1),
                                   constrained_layout=True)
    for dev, rows in per_dev.items():
        if not rows:
            continue
        st = DEV_STYLE[dev]
        bs  = [r['batch'] for r in rows]
        lpf = [r['ms_per_frame'] for r in rows]
        fps = [r['fps'] for r in rows]
        ax1.plot(bs, lpf, st['marker'] + '-', color=st['color'],
                 label=st['label'], linewidth=1.4, markersize=5)
        ax2.plot(bs, fps, st['marker'] + '-', color=st['color'],
                 label=st['label'], linewidth=1.4, markersize=5)

    all_b = sorted({r['batch'] for rows in per_dev.values() for r in rows})
    for ax in (ax1, ax2):
        ax.set_xscale('log', base=2)
        ax.set_xticks(all_b)
        ax.set_xticklabels([str(b) for b in all_b])
        ax.set_xlabel('Batch size')
        ax.grid(True, which='both', linewidth=0.3, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    ax1.set_yscale('log')
    ax1.set_ylabel('Latency(ms)')
    ax2.set_ylabel('Throughput (frames/s)')

    fig.text(0.5, 1.02,
             f"{meta['stem']}  —  {meta['params_M']:.2f} M params, "
             f"{meta['gflops']:.1f} GFLOPs/frame",
             ha='center', va='bottom', fontsize=8)

    _add_axis_break_marks(fig)
    png, pdf = out_base + '.png', out_base + '.pdf'
    fig.savefig(png, dpi=800, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(pdf,            bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Wrote {png}')
    print(f'Wrote {pdf}')


_BREAK_STRAIGHT_TOP    = 0.15
_BREAK_STRAIGHT_BOT    = 0.05
_BREAK_NOTCH_SPAN      = 0.6
_BREAK_NOTCH_DEPTH     = 0.015
_BREAK_SLASH_V         = 0.025
_BREAK_SLASH_OFFSET_V  = 0.014


def _add_axis_break_marks(fig):
    """Extend the LEFT y-axis below its lowest visible tick to host an
    integrated '>' break notch (3 tick-spaces straight -> notch -> 2 tick-
    spaces straight -> x-axis corner). The spine is hidden and replaced by
    one polyline. Skips image / log-scale / hidden-spine / zero-origin axes,
    and axes with <2 visible ticks."""
    for ax in fig.axes:
        if ax.images:
            continue
        if ax.get_yscale() != 'linear':
            continue
        spine = ax.spines['left']
        if not spine.get_visible():
            continue
        y_lo, y_hi = ax.get_ylim()
        if y_lo <= 0:
            continue
        ticks_vis = [t for t in ax.get_yticks() if y_lo - 1e-9 <= t <= y_hi + 1e-9]
        if len(ticks_vis) < 2:
            continue
        tick_spacing = ticks_vis[1] - ticks_vis[0]
        lowest_tick  = ticks_vis[0]
        total_below  = (_BREAK_STRAIGHT_TOP + _BREAK_STRAIGHT_BOT
                        + _BREAK_NOTCH_SPAN)
        new_y_lo     = lowest_tick - total_below * tick_spacing
        if new_y_lo <= 0:
            continue
        ax.set_ylim(new_y_lo, y_hi)
        ax.set_yticks(ticks_vis)
        highest_tick = ticks_vis[-1]
        minor_ticks  = ax.get_yticks(minor=True)
        ax.set_yticks([t for t in minor_ticks
                       if lowest_tick - 1e-9 <= t <= highest_tick + 1e-9],
                      minor=True)
        new_range = y_hi - new_y_lo
        def _f(y_data, _lo=new_y_lo, _r=new_range):
            return (y_data - _lo) / _r
        y_notch_top_data = lowest_tick - _BREAK_STRAIGHT_TOP * tick_spacing
        y_notch_bot_data = y_notch_top_data - _BREAK_NOTCH_SPAN * tick_spacing
        y_top = _f(y_notch_top_data)
        y_bot = _f(y_notch_bot_data)
        slash_v     = _BREAK_SLASH_V
        slash_off_v = _BREAK_SLASH_OFFSET_V
        d           = _BREAK_NOTCH_DEPTH
        y_mid       = 0.5 * (y_top + y_bot)
        y_c_upper   = y_mid + 0.5 * slash_off_v
        y_c_lower   = y_mid - 0.5 * slash_off_v
        spine.set_visible(False)
        kw = dict(transform=ax.transAxes, clip_on=False,
                  color=spine.get_edgecolor(),
                  linewidth=spine.get_linewidth(),
                  solid_capstyle='round')
        ax.plot([0, 0], [1,         y_c_upper], **kw)
        ax.plot([0, 0], [y_c_lower, 0],         **kw)
        ax.plot([-d, +d], [y_c_upper - 0.5 * slash_v,
                           y_c_upper + 0.5 * slash_v], **kw)
        ax.plot([-d, +d], [y_c_lower - 0.5 * slash_v,
                           y_c_lower + 0.5 * slash_v], **kw)


def main():
    ap = argparse.ArgumentParser(description='Inference latency/throughput vs batch size (GPU & CPU)')
    ap.add_argument('--checkpoint', default='new_check_point/ckp_detnet_71.pth',
                    help='DetNet state_dict. Baseline: alone. Pruned: also pass --arch.')
    ap.add_argument('--arch', default=None,
                    help='Matching pruned _noft.pth (omit for baseline).')
    ap.add_argument('--batches', type=int, nargs='+',
                    default=[1, 2, 4, 8, 16, 32, 64, 128],
                    help='Batch sizes to sweep (default: 1..128 powers of 2).')
    ap.add_argument('--devices', nargs='+', choices=['cuda', 'cpu'],
                    default=['cuda', 'cpu'],
                    help='Devices to sweep (default: both).')
    ap.add_argument('--warmup', type=int, default=10,
                    help='GPU warm-up iters discarded (default: 10).')
    ap.add_argument('--iters', type=int, default=50,
                    help='GPU timed iters per batch (default: 50).')
    ap.add_argument('--cpu_max_batch', type=int, default=32,
                    help='Cap CPU sweep at this batch (CPU large-batch is slow; '
                         'default: 32).')
    ap.add_argument('--cpu_iters', type=int, default=15,
                    help='CPU timed iters per batch (default: 15).')
    ap.add_argument('--input_size', type=int, default=INPUT_SIZE)
    ap.add_argument('--out_dir', type=str, default='benchmark_results')
    ap.add_argument('--fig_dir', type=str, default='figures')
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    per_dev = {}
    for dev in args.devices:
        if dev == 'cuda' and not torch.cuda.is_available():
            print('cuda requested but unavailable — skipping GPU sweep.')
            continue
        if dev == 'cpu':
            batches = [b for b in args.batches if b <= args.cpu_max_batch]
            warm, it = max(2, args.warmup // 2), args.cpu_iters
        else:
            batches, warm, it = args.batches, args.warmup, args.iters
        print(f'\nSweeping {dev.upper()}  batches={batches}  '
              f'(warmup={warm}, iters={it})')
        per_dev[dev] = sweep_device(args.checkpoint, args.arch, dev,
                                    batches, warm, it, args.input_size)

    if not any(per_dev.values()):
        print('No results — nothing to plot.')
        return

    # Model stats for the figure caption (device-independent).
    dev0  = torch.device('cuda:0' if ('cuda' in per_dev and per_dev['cuda'])
                          else 'cpu')
    m     = load_model(args.checkpoint, dev0, arch_path=args.arch)
    gflops, _ = compute_flops(m, dev0)
    meta = {'stem': stem, 'params_M': count_params(m) / 1e6,
            'gflops': gflops if gflops is not None else float('nan')}

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    base = os.path.join(args.fig_dir, f'fig_batch_sweep_{stem}')
    make_figure(per_dev, meta, base)

    flat = [r for rows in per_dev.values() for r in rows]
    with open(os.path.join(args.out_dir, f'batch_sweep_{stem}.json'),
              'w', encoding='utf-8') as f:
        json.dump({'timestamp': datetime.now().isoformat(timespec='seconds'),
                   'checkpoint': args.checkpoint, 'arch': args.arch,
                   'model': meta, 'rows': flat}, f, indent=2)
    with open(os.path.join(args.out_dir, f'batch_sweep_{stem}.csv'),
              'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    print(f'Wrote {args.out_dir}/batch_sweep_{stem}.json / .csv')


if __name__ == '__main__':
    main()
