"""
benchmark_cpu_sweep.py — CPU latency vs pruning ratio, all models,
with statistically-correct ACROSS-RUN error bars.

CPU is compute-bound (no thousands-of-cores parallelism to hide the FLOP
cut), so unlike the launch-bound GPU eval-loop numbers, CPU latency tracks
pruning. This sweeps every model through the *controlled* benchmark
(benchmark_inference.py --device cpu: fixed synthetic input, no data
pipeline, perf_counter, warm-up discarded).

Why repeated sweeps
-------------------
benchmark_inference.py's reported std is the spread of the 500 iterations
*within one run*. That captures fast iteration jitter but NOT the slow
whole-run offset (CPU turbo/thermal state, OS background load, which cores
the thread-pool grabbed for that launch) — which shifts a run's whole mean
together and is the real uncertainty when comparing models. Averaging more
iterations inside a run does nothing to it; you must average over
*independent runs*.

So this does `--repeats R` full passes, **repeat-major** (a whole sweep
over all models, then the next whole sweep — repeats are separated in time
so they genuinely sample different thermal/load states), and reports each
model's mean ± std *across the R run-means*. That is the correct error bar
for "is there an L1-vs-Taylor difference?": same-ratio L1 and Taylor are
architecturally identical (same filter count/layer ⇒ identical compute), so
their across-run bars must overlap; if they don't, something is wrong.

Model list is auto-discovered from eval_results/*_both/summary.json (each
records its own `checkpoint` + `arch`) — baseline + every L1/Taylor ratio
(incl. 85% when present). MediaPipe is skipped (not a DetNet checkpoint).

Layout (idempotent — re-run any time; resumes; --force re-measures all):
  benchmark_results/run{r}/<stem>_cpu_bs1/summary.json   for r in 0..R-1

Outputs:
  figures/fig_cpu_latency_vs_ratio.{pdf,png}  (2 panels: CPU latency
      ms/frame, and speed-up × vs baseline; bars = ±1 std across R runs)
  benchmark_results/cpu_latency_vs_ratio.csv

Usage:
  python benchmark_cpu_sweep.py                       # 5 independent sweeps
  python benchmark_cpu_sweep.py --repeats 8
  python benchmark_cpu_sweep.py --force               # re-measure everything
  python benchmark_cpu_sweep.py --iters 300 --warmup 30
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
import glob
import json
import re
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_pruning_results import COLORS, METHOD_STYLE

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'DejaVu Serif'],
    'font.size':   9,
})


# ── Discover the models from eval_results ────────────────────────────────────

def parse_stem(stem):
    """benchmark_results stem -> (method, ratio_pct). Mirrors
    build_results_csv.parse_run_name."""
    s = stem.lower()
    m = re.search(r'_(\d+)pct', s)
    ratio = int(m.group(1)) if m else 0
    if 'taylor' in s:
        return 'taylor', ratio
    if 'l1' in s:
        return 'l1', ratio
    return 'baseline', 0


def discover(results_dir):
    """[(checkpoint, arch, stem), ...] for baseline + every L1/Taylor *_both."""
    specs = []
    for d in sorted(glob.glob(os.path.join(results_dir, '*_both'))):
        if 'mediapipe' in os.path.basename(d):
            continue
        sj = os.path.join(d, 'summary.json')
        if not os.path.isfile(sj):
            continue
        s = json.load(open(sj, encoding='utf-8'))
        ckpt = s.get('checkpoint')
        if not ckpt:
            continue
        stem = os.path.splitext(os.path.basename(ckpt))[0]
        specs.append((ckpt, s.get('arch'), stem))
    return specs


# ── Run the controlled CPU benchmark for one model, one repeat (idempotent) ──

def ensure_benchmark(ckpt, arch, stem, run_out_dir, warmup, iters, force):
    """Measure one model into run_out_dir/<stem>_cpu_bs1/. Cached per
    (repeat, model): skipped if its summary already exists unless --force."""
    bench_dir = os.path.join(run_out_dir, f'{stem}_cpu_bs1')
    sj        = os.path.join(bench_dir, 'summary.json')
    if os.path.isfile(sj) and not force:
        print(f'    cached  {stem}')
        return sj
    os.makedirs(run_out_dir, exist_ok=True)
    cmd = [sys.executable, 'benchmark_inference.py', '--device', 'cpu',
           '--checkpoint', ckpt, '--batch_size', '1',
           '--warmup', str(warmup), '--iters', str(iters),
           '--out_dir', run_out_dir]
    if arch:
        cmd += ['--arch', arch]
    print(f'    measuring {stem} ...', flush=True)
    # benchmark_inference.py reconfigures its stdout to UTF-8 (prints ±, ×).
    # Default Windows pipe decoding is cp1252, which crashes the capture
    # reader thread on those bytes — force UTF-8 with replacement.
    r = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                       errors='replace')
    if r.returncode != 0 or not os.path.isfile(sj):
        print(f'    FAILED {stem}:\n{(r.stdout or "")[-800:]}\n'
              f'{(r.stderr or "")[-800:]}')
        return None
    return sj


# ── Collate across repeats ───────────────────────────────────────────────────

def collate_repeats(base_out_dir, n_runs):
    """For each model, gather its per-run mean latency across the R run dirs
    and reduce to (across-run mean, across-run std). Falls back to the
    within-run std when only one run is present (so a single sweep still
    produces a usable figure, just with the weaker error bar)."""
    per = {}  # stem -> aggregate
    for r in range(n_runs):
        run_dir = os.path.join(base_out_dir, f'run{r}')
        for sj in sorted(glob.glob(os.path.join(run_dir, '*_cpu_bs1',
                                                'summary.json'))):
            s    = json.load(open(sj, encoding='utf-8'))
            stem = os.path.basename(os.path.dirname(sj))[:-len('_cpu_bs1')]
            L    = s['latency']
            d = per.setdefault(stem, {
                'stem': stem,
                'params_M': s['model']['param_count_M'],
                'gflops':   s['compute']['gflops_forward'],
                'run_means': [], 'within_stds': [],
            })
            d['run_means'].append(float(L['total_ms_per_frame']))
            d['within_stds'].append(float(L.get('total_ms_std', 0.0) or 0.0))

    rows = []
    for stem, d in per.items():
        means = np.asarray(d['run_means'], float)
        n = means.size
        if n == 0:
            continue
        method, ratio = parse_stem(stem)
        mean = float(means.mean())
        within = float(np.mean(d['within_stds'])) if d['within_stds'] else 0.0
        if n >= 2:
            std, err_kind = float(means.std(ddof=1)), 'across_runs'
        else:
            std, err_kind = within, 'within_run'   # single-sweep fallback
        rows.append({
            'stem': stem, 'method': method, 'ratio': ratio,
            'params_M': d['params_M'], 'gflops': d['gflops'],
            'n_runs': n,
            'ms_per_frame':       mean,
            'ms_per_frame_std':   std,          # plotted bar
            'err_kind':           err_kind,
            'ms_within_run_std':  within,       # reference only
            'fps': (1000.0 / mean) if mean else None,
        })
    return rows


# ── Plot ─────────────────────────────────────────────────────────────────────

def make_figure(rows, out_base, n_runs):
    base    = next((r for r in rows if r['method'] == 'baseline'), None)
    base_ms = base['ms_per_frame'] if base else None
    base_se = base['ms_per_frame_std'] if base else 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1),
                                   constrained_layout=True)
    ratios_seen = set()
    for method in ('l1', 'taylor'):
        pts = sorted((r for r in rows if r['method'] == method),
                     key=lambda r: r['ratio'])
        if not pts:
            continue
        st = METHOD_STYLE[method]
        xs = [p['ratio'] for p in pts]
        ys = [p['ms_per_frame'] for p in pts]
        es = [p['ms_per_frame_std'] for p in pts]
        ratios_seen.update(xs)
        ax1.errorbar(xs, ys, yerr=es, fmt=st['marker'] + '-',
                     color=st['color'], capsize=2, linewidth=1.4,
                     markersize=5, label=st['label'])
        if base_ms:
            sp  = [base_ms / p['ms_per_frame'] for p in pts]
            # error on a ratio: relative errors add in quadrature
            spe = [s * np.hypot(base_se / base_ms,
                                p['ms_per_frame_std'] / p['ms_per_frame'])
                   for s, p in zip(sp, pts)]
            ax2.errorbar(xs, sp, yerr=spe, fmt=st['marker'] + '-',
                         color=st['color'], capsize=2, linewidth=1.4,
                         markersize=5, label=st['label'])

    if base is not None:
        bs = METHOD_STYLE['baseline']
        ax1.errorbar([0], [base['ms_per_frame']], yerr=[base_se],
                     fmt=bs['marker'], color=bs['color'], capsize=2,
                     markersize=7, label=bs['label'])
        ax1.axhline(base['ms_per_frame'], linestyle='--',
                    color=bs['color'], linewidth=0.8, alpha=0.6)
        ax2.axhline(1.0, linestyle='--', color=bs['color'],
                    linewidth=0.8, alpha=0.6)
        ratios_seen.add(0)

    xticks = sorted(ratios_seen)
    for ax in (ax1, ax2):
        ax.set_xlabel('Filter removal ratio (%)')
        ax.set_xticks(xticks)
        ax.grid(True, linewidth=0.3, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    ax1.set_ylabel('Latency(ms)')
    ax2.set_ylabel('Speed-up vs baseline (×)')
    fig.suptitle(f'Error bars: ±1 std across {n_runs} independent sweeps',
                 fontsize=8, y=1.04)

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
    ap = argparse.ArgumentParser(
        description='CPU latency vs pruning ratio, repeated sweeps with '
                    'across-run error bars')
    ap.add_argument('--results_dir', default='eval_results')
    ap.add_argument('--out_dir', default='benchmark_results')
    ap.add_argument('--fig_dir', default='figures')
    ap.add_argument('--repeats', type=int, default=5,
                    help='number of independent full sweeps; the error bar '
                         'is the std across these run-means (default 5)')
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--force', action='store_true',
                    help='re-measure even where a cached summary exists')
    args = ap.parse_args()

    specs = discover(args.results_dir)
    if not specs:
        print(f'No *_both runs found in {args.results_dir}.')
        return
    R = max(1, args.repeats)
    print(f'Models discovered: {len(specs)}   |   repeats: {R}')

    # Repeat-major: a whole sweep, then the next — repeats are separated in
    # time so the across-run std actually captures the slow per-run offset.
    for r in range(R):
        print(f'\n=== sweep {r + 1}/{R} '
              f'(benchmark_results/run{r}/) ===', flush=True)
        run_out = os.path.join(args.out_dir, f'run{r}')
        for ckpt, arch, stem in specs:
            ensure_benchmark(ckpt, arch, stem, run_out,
                             args.warmup, args.iters, args.force)

    rows = collate_repeats(args.out_dir, R)
    if not rows:
        print('No CPU benchmark summaries collated.')
        return
    rows.sort(key=lambda r: ({'baseline': 0, 'l1': 1, 'taylor': 2}[r['method']],
                             r['ratio']))

    kinds = {r['err_kind'] for r in rows}
    if 'within_run' in kinds:
        print('\nWARNING: some/all points have only 1 run — error bar falls '
              'back to the weaker within-run std. Use --repeats >= 2 (and '
              'let every sweep finish) for the correct across-run bars.')

    os.makedirs(args.fig_dir, exist_ok=True)
    make_figure(rows, os.path.join(args.fig_dir,
                                   'fig_cpu_latency_vs_ratio'), R)

    csv_path = os.path.join(args.out_dir, 'cpu_latency_vs_ratio.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f'\n{"model":<26}{"ratio":>6}{"runs":>6}{"GFLOPs":>8}'
          f'{"CPU ms/frame (±across-run)":>28}{"fps":>8}')
    for r in rows:
        print(f'{r["stem"]:<26}{r["ratio"]:>6}{r["n_runs"]:>6}'
              f'{r["gflops"]:>8.1f}'
              f'{r["ms_per_frame"]:>18.2f} ± {r["ms_per_frame_std"]:<6.2f}'
              f'{(r["fps"] or 0):>8.1f}')
    print(f'\nWrote {csv_path}')


if __name__ == '__main__':
    main()
