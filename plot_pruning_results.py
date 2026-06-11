"""
plot_pruning_results.py — Publication-quality thesis figures from evaluate_detnet.py outputs.

Scans <results_dir>/<run_name>/summary.json (+ errors.npz), groups runs by
pruning method (l1 / taylor / baseline), and writes one PDF per (method × figure).

Style follows journal-publication conventions:
  - Serif font (Times New Roman), 10pt body, sized for a single-column figure.
  - Okabe-Ito colorblind-safe palette (Nature-standard).
  - No figure titles — captions belong in the LaTeX/Word doc.
  - Axis labels always carry units.
  - Vector PDF output at 300 DPI fallback.
  - Spines on all four sides, minor ticks on, frameless legend.

Currently produces (per method m ∈ {l1, taylor}):
  fig06_accuracy_vs_ratio_<m>.pdf       — FreiHAND + RHD AUC vs filter removal %
  fig07_inference_time_vs_ratio_<m>.pdf — GPU latency vs filter removal % + 33ms threshold
  fig08_model_size_vs_ratio_<m>.pdf     — Model size (MB) vs filter removal %
  fig09_gflops_vs_ratio_<m>.pdf         — GFLOPs vs filter removal %

The script is structured so adding new figures only needs another call to
`plot_single_metric_vs_ratio()` (with a new metric key) or a sibling function
modelled on `plot_accuracy_vs_ratio()`.

Usage:
  python plot_pruning_results.py
  python plot_pruning_results.py --methods l1 taylor --mediapipe_auc 85.0
  python plot_pruning_results.py --results_dir eval_results --out_dir figures --list_runs
"""
import glob
import json
import os
import re
import sys

try:
    # Skip messages / labels contain em-dashes; force UTF-8 so stdout doesn't
    # crash on Windows cp1252 when the run is piped or Tee-Object'd to a file.
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ── Publication-quality rcParams ────────────────────────────────────────────
# Applied globally — every figure inherits these settings.
plt.rcParams.update({
    # Font: serif family to match LaTeX/academic documents
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',

    # Font sizes — single-column figure (~3.5in) or double-column (~7in)
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,

    # Tick styling — inward, on all four spines, with minor ticks
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.major.size':    5,
    'ytick.major.size':    5,
    'xtick.minor.size':    2.5,
    'ytick.minor.size':    2.5,
    'xtick.major.width':   0.8,
    'ytick.major.width':   0.8,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.top':           True,
    'ytick.right':         True,

    # Lines
    'lines.linewidth':  1.5,
    'lines.markersize': 5,

    # Axes
    'axes.linewidth': 0.8,
    'axes.grid':      False,

    # Legend
    'legend.frameon':       False,
    'legend.handletextpad': 0.4,

    # Figure / saving
    'figure.dpi':        150,
    'savefig.dpi':       800,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
})


# ── Okabe-Ito colorblind-safe palette ───────────────────────────────────────
# Distinguishable under deuteranopia, protanopia and tritanopia; prints in greyscale.
COLORS = {
    # Unified per-method palette — applied identically in Fig 10 (pruning),
    # Fig 12 (Pareto), fig_quant_comparison, and fig14_qualitative_overlay so
    # the same model reads as the same colour everywhere. All entries are
    # Okabe-Ito (colour-blind safe) except baseline_grey.
    'blue':          '#0072B2',  # L1
    'orange':        '#E69F00',  # Taylor (amber)
    'red':           '#D55E00',  # QMM (INT8) — coral/red-orange, distinct from amber
    'green':         '#009E73',  # QMSE (INT8)
    'purple':        '#CC79A7',  # MediaPipe (pink) — distinct sixth slot
    'baseline_grey': '#555555',  # DetNet FP32 baseline — neutral, reads as reference
    # Auxiliary slots not part of the per-method scheme:
    'sky':           '#56B4E9',  # error bands / shading
    'black':         '#000000',  # threshold/annotation only (NOT a model colour)
}


# ── Shared layout constants — keep Fig 6/7/8/9 visually aligned when stacked ──
FIGSIZE      = (3.5, 2.8)               # single-column ≈ 89 mm wide
FIGSIZE_WIDE = (7.0, 2.8)               # double-column / 2-panel (Fig 10, 12)
XTICKS       = [0, 10, 25, 40, 55, 70]  # exact pruning ratios
XLIM         = (-3, 73)                 # small padding so endpoints aren't clipped

# Per-method styling — colour + marker. Marker shapes follow the Fig 12 spec
# (circle=L1, diamond=Taylor, square=quantized, star=combined) and are reused
# everywhere so a method looks identical across every figure.
METHOD_STYLE = {
    # Per-method style — colour is the source of truth shared across every
    # figure (Fig 10, Fig 12, fig_quant_comparison, fig14_qualitative_overlay).
    'l1':        {'color': COLORS['blue'],          'marker': 'o', 'label': 'L1 norm'},
    'taylor':    {'color': COLORS['orange'],        'marker': 'D', 'label': 'Taylor'},
    'qmm':       {'color': COLORS['red'],           'marker': 's', 'label': 'QMM (INT8)'},
    'qmse':      {'color': COLORS['green'],         'marker': '^', 'label': 'QMSE (INT8)'},
    'baseline':  {'color': COLORS['baseline_grey'], 'marker': 'P', 'label': 'DetNet (FP32)'},
    'mediapipe': {'color': COLORS['purple'],        'marker': 'X', 'label': 'MediaPipe'},
    # Legacy aliases kept for any callers still referencing them — point to
    # the canonical method slot above.
    'quantized': {'color': COLORS['red'],           'marker': 's', 'label': 'INT8 PTQ'},
    'combined':  {'color': COLORS['green'],         'marker': '*', 'label': 'Combined'},
}

# SNAP 21-joint hand skeleton connectivity (wrist=0; thumb 1-4, index 5-8,
# middle 9-12, ring 13-16, pinky 17-20). Used by the Fig 14 keypoint overlay.
SNAP_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]


# ── Discovery ───────────────────────────────────────────────────────────────

def parse_method_ratio(checkpoint_path):
    """Parse pruning method and ratio from a checkpoint filename.

    'ckp_detnet_71.pth'           -> ('baseline', 0)
    'ft_l1_25pct_rhdbest.pth'     -> ('l1', 25)
    'ft_taylor_40pct_rhdbest.pth' -> ('taylor', 40)
    'mediapipe_hands'             -> ('mediapipe', 0)
    """
    name = os.path.basename(checkpoint_path).lower()
    if 'mediapipe' in name:
        method = 'mediapipe'
    elif 'qmse' in name:
        method = 'qmse'
    elif 'qmm' in name:
        method = 'qmm'
    elif 'taylor' in name:
        method = 'taylor'
    elif 'l1' in name:
        method = 'l1'
    else:
        method = 'baseline'
    m = re.search(r'(\d+)pct', name)
    ratio = int(m.group(1)) if m else 0
    return method, ratio


def load_runs(results_dir):
    """Find every <results_dir>/*/summary.json and return one dict per run,
    annotated with parsed method/ratio and paths to the run's artefacts."""
    runs = []
    pattern = os.path.join(results_dir, '*', 'summary.json')
    for json_path in sorted(glob.glob(pattern)):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        run_dir = os.path.dirname(json_path)
        method, ratio = parse_method_ratio(data.get('checkpoint', ''))
        data['_method']      = method
        data['_ratio']       = ratio
        data['_run_dir']     = run_dir
        data['_errors_path'] = os.path.join(run_dir, 'errors.npz')
        runs.append(data)
    return runs


# ── Derived metrics ─────────────────────────────────────────────────────────

def _auc_from_errors(errors, tau_min, tau_max):
    """PCK-AUC from a flat array of per-joint pixel errors.
    Same integration scheme as evaluate_detnet.summarise() so the plotted numbers
    match what was printed to stdout."""
    taus = np.arange(tau_min, tau_max + 1, dtype=float)
    pck  = [min(float((errors < t).mean()) * 100.0, 100.0) for t in taus]
    return float(np.trapz(pck, taus)) / (tau_max - tau_min)


def _freihand_pooled_errors(run):
    """Concatenated per-joint pixel errors over FreiHAND eval + composited-train,
    or None if errors.npz is missing/empty. Shared by _freihand_combined_auc
    (Fig 6/10/11/12) and plot_pck_curves (Fig 5)."""
    if not os.path.isfile(run['_errors_path']):
        return None
    npz = np.load(run['_errors_path'])
    parts = [npz[k] for k in npz.files if k.startswith('freihand')]
    return np.concatenate(parts) if parts else None


def _freihand_combined_auc(run, tau_min, tau_max):
    """AUC over FreiHAND eval ∪ train, computed from raw errors so the union
    is exact (not a frame-weighted average of per-split AUCs)."""
    errs = _freihand_pooled_errors(run)
    if errs is None:
        return None
    return _auc_from_errors(errs, tau_min, tau_max)


def _rhd_auc(run):
    """RHD eval AUC from summary.json (no errors.npz load needed)."""
    return run.get('splits', {}).get('rhd_eval', {}).get('auc_pck')


def _scalar(run, key):
    """Lookup helper for the per-run scalars plotted in Fig 7/8/9."""
    lookups = {
        'total_mb': lambda r: r['model']['total_mb'],
        'gflops':   lambda r: r['compute']['gflops_forward'],
        'fps':      lambda r: r['latency']['fps'],
        'latency':  lambda r: r['latency']['total_ms_per_frame'],
    }
    return lookups[key](run)


# ── Shared per-axes styling ─────────────────────────────────────────────────

def _style_axes(ax, ylabel):
    """Apply the styling shared across every figure: x-axis configuration,
    light grid, and identical limits so figures stack visually."""
    ax.set_xlabel('Filter removal ratio (%)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(XTICKS)
    ax.set_xlim(*XLIM)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.3)


# Geometry of the y-axis break (in units of "tick-space" — the distance
# between two consecutive y-ticks). Below the lowest visible tick, the
# extended spine is laid out as:
#     <STRAIGHT_TOP ticks of straight line> -> '>' notch -> <STRAIGHT_BOT ticks of straight line> -> x-axis
_BREAK_STRAIGHT_TOP    = 0.15
_BREAK_STRAIGHT_BOT    = 0.05
_BREAK_NOTCH_SPAN      = 0.6    # vertical span of the '//' break region (tick-spaces)
_BREAK_NOTCH_DEPTH     = 0.015  # half-horizontal extent of each slash (axes-frac)
# FIXED axes-fraction visual size for the '//' pair so it's the same across
# every plot (does not scale with the per-plot tick-spacing).
_BREAK_SLASH_V         = 0.025
_BREAK_SLASH_OFFSET_V  = 0.014


def _add_axis_break_marks(fig):
    """Extend the LEFT y-axis BELOW its lowest visible tick to host an
    integrated '>' break notch, on every axes whose bottom y-limit is
    above 0. Layout below the lowest tick (tick-spaces):

        | <-- lowest visible tick (e.g. 62)
        |  3 ticks of straight extension
        > <-- the break notch
        |  2 ticks of straight extension
        +-- x-axis corner

    The matplotlib left-spine is hidden and replaced by a single polyline
    that traces this geometry, inheriting the spine's colour + linewidth
    so the notch reads as part of the axis line, not a decoration.

    Skipped: image-display axes, log-scale axes, axes whose y-limit
    already includes 0, axes whose left spine is already hidden (twinx
    secondaries), and axes with fewer than 2 visible ticks (can't
    compute the tick-spacing unit).
    """
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
        if new_y_lo <= 0:                  # would push origin below 0 — skip
            continue

        # Extend the visible range; lock the tick locator so no labels appear
        # in the new extension (otherwise matplotlib auto-fills it).
        ax.set_ylim(new_y_lo, y_hi)
        ax.set_yticks(ticks_vis)
        # Suppress minor ticks in the break region — keep them ONLY between
        # the lowest and highest major ticks (i.e., inside the data range).
        highest_tick   = ticks_vis[-1]
        minor_ticks    = ax.get_yticks(minor=True)
        ax.set_yticks([t for t in minor_ticks
                       if lowest_tick - 1e-9 <= t <= highest_tick + 1e-9],
                      minor=True)

        # Convert key y values from data coords -> axes-fraction.
        new_range = y_hi - new_y_lo
        def _f(y_data, _lo=new_y_lo, _r=new_range):
            return (y_data - _lo) / _r
        y_notch_top_data = lowest_tick - _BREAK_STRAIGHT_TOP * tick_spacing
        y_notch_bot_data = y_notch_top_data - _BREAK_NOTCH_SPAN * tick_spacing
        y_top = _f(y_notch_top_data)
        y_bot = _f(y_notch_bot_data)

        # Geometry of the '//' pair, centred vertically in the notch region.
        # FIXED axes-fraction sizes so the symbol looks the same across plots
        # regardless of tick-spacing (otherwise wider-spaced y-axes get a
        # visibly larger '//' than tightly-spaced ones).
        slash_v       = _BREAK_SLASH_V               # vertical extent per slash
        slash_off_v   = _BREAK_SLASH_OFFSET_V        # vert. offset between the two
        d             = _BREAK_NOTCH_DEPTH           # half-horizontal extent
        y_mid         = 0.5 * (y_top + y_bot)
        y_c_upper     = y_mid + 0.5 * slash_off_v   # x=0 crossing of upper slash
        y_c_lower     = y_mid - 0.5 * slash_off_v   # x=0 crossing of lower slash

        # Replace the left spine with: two vertical segments that extend in
        # to MEET the slash midpoints (so the spine connects straight into
        # each slash), plus the two parallel '//' diagonals themselves.
        spine.set_visible(False)
        kw = dict(transform=ax.transAxes, clip_on=False,
                  color=spine.get_edgecolor(),
                  linewidth=spine.get_linewidth(),
                  solid_capstyle='round')
        ax.plot([0, 0], [1,         y_c_upper], **kw)   # top -> upper slash
        ax.plot([0, 0], [y_c_lower, 0],         **kw)   # lower slash -> x-axis
        ax.plot([-d, +d], [y_c_upper - 0.5 * slash_v,
                           y_c_upper + 0.5 * slash_v], **kw)
        ax.plot([-d, +d], [y_c_lower - 0.5 * slash_v,
                           y_c_lower + 0.5 * slash_v], **kw)


def _annotate_baseline(ax, y, label, color=COLORS['baseline_grey'], linestyle='--',
                       x=0, ha='left', dy=2):
    """Draw a horizontal reference line and label it INSIDE the axes (not in
    the legend) — reduces legend clutter as per publication style.

    x  : data-x to anchor the inline label. Defaults to 0 (Fig 6-9 fix XLIM so 0
         is visible); Fig 10 auto-scales from ~10, so it passes a ratio instead.
    ha : 'left' (label to the right of x) or 'right' (label to the left of x).
    dy : vertical offset in points — positive = above the line, negative = below.
    """
    ax.axhline(y, linestyle=linestyle, color=color, linewidth=0.8, alpha=0.7)
    dx = 2 if ha == 'left' else -2
    va = 'bottom' if dy >= 0 else 'top'
    ax.annotate(label, xy=(x, y), xytext=(dx, dy),
                textcoords='offset points', fontsize=6.5,
                ha=ha, va=va, color=color, alpha=0.85)


# ── Plotting ────────────────────────────────────────────────────────────────

# ── Fig 5: PCK curves for baseline + every L1 / Taylor model ────────────────

def plot_pck_curves(runs, out_path, tau_min=0, tau_max=30,
                    ratios=(10, 25, 40, 55, 70)):
    """Fig 5 — PCK(τ) curves for the baseline and all 10 pruned models
    (L1 × 5 ratios + Taylor × 5 ratios) over τ ∈ [tau_min, tau_max] pixels.

    Single panel matching the suite's house style (Fig 10): full box, y-axis
    grid, inline legend. Method is encoded by colour family AND marker shape
    (greyscale-safe): L1 = blue→navy + circles, Taylor = orange→brown +
    diamonds. Within each family, a hand-picked hex ramp maps ratio → darkness
    (heavier pruning = darker). FP32 baseline = black reference line.

    Curves use the pooled FreiHAND eval + composited-train errors — the same
    union that feeds the Fig 6/10/11/12 AUC numbers.
    """
    taus = np.arange(tau_min, tau_max + 1, dtype=float)
    # Sequential 5-stop ramps: light → dark = lower → higher pruning ratio.
    L1_COLORS     = ['#A6CEE3', '#6BAED6', '#2171B5', '#08519C', '#08306B']
    TAYLOR_COLORS = ['#FDD0A2', '#FD8D3C', '#D94801', '#8C2D04', '#3E1306']
    families = {
        'l1':     {'colors': L1_COLORS,     'marker': 'o', 'label': 'L1 norm'},
        'taylor': {'colors': TAYLOR_COLORS, 'marker': 'D', 'label': 'Taylor'},
    }

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    handles, labels = [], []

    # Baseline drawn last (zorder=5) so it sits on top of the pruned curves
    # — it's the reference envelope.
    baseline = next((r for r in runs if r['_method'] == 'baseline'), None)
    if baseline is not None:
        errs = _freihand_pooled_errors(baseline)
        if errs is not None:
            pck = [float((errs < t).mean()) * 100.0 for t in taus]
            line, = ax.plot(taus, pck,
                            color=COLORS['black'], linestyle='-',
                            linewidth=1.4, zorder=5)
            handles.append(line)
            labels.append(METHOD_STYLE['baseline']['label'])

    def _add_method(method, spec):
        method_runs = {r['_ratio']: r for r in runs if r['_method'] == method}
        for i, ratio in enumerate(ratios):
            run = method_runs.get(ratio)
            if run is None:
                continue
            errs = _freihand_pooled_errors(run)
            if errs is None:
                continue
            pck = [float((errs < t).mean()) * 100.0 for t in taus]
            line, = ax.plot(taus, pck,
                            color=spec['colors'][i],
                            linestyle='-', linewidth=0.9,
                            marker=spec['marker'], markersize=3.5,
                            markevery=5)        # τ = 0, 5, 10, 15, 20, 25, 30
            handles.append(line)
            labels.append(f'{spec["label"]} {ratio}%')

    _add_method('l1', families['l1'])
    # Blank legend entry: visual separator between the L1 and Taylor groups.
    handles.append(plt.Line2D([], [], linestyle='', marker=''))
    labels.append('')
    _add_method('taylor', families['taylor'])

    ax.set_xlabel('PCK threshold τ (pixels)')
    ax.set_ylabel('Accuracy (AUC,%)')
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xlim(tau_min, tau_max + 2)     # extend axis past last tick for visual padding
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.3)
    ax.legend(handles, labels, loc='lower right', fontsize=6.5, ncol=2,
              handlelength=2.0, columnspacing=1.0)
    fig.tight_layout()
    _add_axis_break_marks(fig)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  -> {out_path}')


def plot_accuracy_vs_ratio(runs, out_path, method,
                           tau_min=20, tau_max=50, mediapipe_auc=None):
    """Fig 6 — Two-line accuracy plot for one pruning method:
       FreiHAND (blue, circles) + RHD (orange, squares).
    Different marker shapes ensure the lines are distinguishable in greyscale.
    Baselines are drawn as horizontal lines with inline text labels, not legend
    entries, to keep the legend uncluttered.
    """
    method_runs = sorted([r for r in runs if r['_method'] == method],
                         key=lambda r: r['_ratio'])
    ratios = [r['_ratio'] for r in method_runs]
    fh     = [_freihand_combined_auc(r, tau_min, tau_max) for r in method_runs]
    rhd    = [_rhd_auc(r)                                 for r in method_runs]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # yerr=None today → just line+markers. When multi-seed std arrays become
    # available, pass them here for error caps with the spec'd geometry.
    ax.errorbar(ratios, fh,  yerr=None, fmt='o-', color=COLORS['blue'],
                capsize=3, capthick=0.8, elinewidth=0.8, label='FreiHAND')
    if any(v is not None for v in rhd):   # CPU/FreiHAND-only runs have no RHD split
        ax.errorbar(ratios, rhd, yerr=None, fmt='s-', color=COLORS['orange'],
                    capsize=3, capthick=0.8, elinewidth=0.8, label='RHD')

    baseline = next((r for r in runs if r['_method'] == 'baseline'), None)
    if baseline is not None:
        fh_b  = _freihand_combined_auc(baseline, tau_min, tau_max)
        rhd_b = _rhd_auc(baseline)
        if fh_b is not None:
            _annotate_baseline(ax, fh_b,
                               f'DetNet baseline FreiHAND ({fh_b:.2f}%)',
                               linestyle='--')
        if rhd_b is not None:
            _annotate_baseline(ax, rhd_b,
                               f'DetNet baseline RHD ({rhd_b:.2f}%)',
                               linestyle=':')

    if mediapipe_auc is not None:
        _annotate_baseline(ax, mediapipe_auc,
                           f'MediaPipe ({mediapipe_auc:.2f}%)',
                           color=COLORS['green'], linestyle=':')

    _style_axes(ax, 'Accuracy (AUC,%)')
    ax.legend(loc='best')
    fig.tight_layout()
    _add_axis_break_marks(fig)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  -> {out_path}')


def plot_single_metric_vs_ratio(runs, out_path, *, method, metric, ylabel,
                                threshold=None, threshold_label=None,
                                baseline_unit_fmt='{value:.2f}',
                                color=None, force_ymin=None):
    """Generic single-line plot for Fig 7/8/9 — one pruning method, one metric.

    threshold        : optional horizontal red dashed reference (e.g. 33 ms / 30 FPS for Fig 7).
    threshold_label  : text annotation placed near the threshold (no legend entry).
    baseline_unit_fmt: format string for the inline DetNet-baseline annotation.
    color            : line/marker colour; defaults to COLORS['blue'] (Fig 7).
                       Fig 8/9 pass dark green / brown so the size and compute
                       plots are visually distinct from the latency plot.
    force_ymin       : if set, lower the y-axis so this value becomes the
                       bottom of the visible range (the auto-locator then
                       picks it as a tick). Used for the latency plot so y=10
                       appears as the lowest tick — the axis-break sits
                       safely below it with no data points at the same
                       horizontal level as the break symbol.
    """
    method_runs = sorted([r for r in runs if r['_method'] == method],
                         key=lambda r: r['_ratio'])
    ratios = [r['_ratio']        for r in method_runs]
    vals   = [_scalar(r, metric) for r in method_runs]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.errorbar(ratios, vals, yerr=None, fmt='o-', color=color or COLORS['blue'],
                capsize=3, capthick=0.8, elinewidth=0.8)

    baseline = next((r for r in runs if r['_method'] == 'baseline'), None)
    if baseline is not None:
        b_val = _scalar(baseline, metric)
        _annotate_baseline(
            ax, b_val,
            f'DetNet baseline ({baseline_unit_fmt.format(value=b_val)})',
            linestyle='--', x=XTICKS[-1], ha='right', dy=-4,
        )

    if threshold is not None:
        ax.axhline(threshold, color=COLORS['red'], linestyle='--', linewidth=1.0)
        if threshold_label:
            # Place text just above the threshold line, right-aligned to the
            # rightmost x-tick so it sits in usually-empty plot space.
            ax.annotate(threshold_label,
                        xy=(XTICKS[-1], threshold), xytext=(0, 4),
                        textcoords='offset points', fontsize=7,
                        ha='right', color=COLORS['red'])

    _style_axes(ax, ylabel)
    if force_ymin is not None:
        cur_lo, cur_hi = ax.get_ylim()
        if cur_lo > force_ymin:
            ax.set_ylim(force_ymin, cur_hi)
    fig.tight_layout()
    _add_axis_break_marks(fig)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  -> {out_path}')


# ── Fig 10: L1 vs Taylor comparison (2-panel) ───────────────────────────────

def plot_l1_vs_taylor(runs, out_path, tau_min=20, tau_max=50,
                       shared_ratios=None):
    """Fig 10 — Does a better importance metric preserve more accuracy?

    Left panel : FreiHAND AUC vs ratio, L1 (blue circles) vs Taylor (orange
                 diamonds).
    Right panel: inference time for both — expected near-identical, since the
                 same number of filters is removed regardless of selection rule.

    shared_ratios=None (default) auto-computes the intersection of ratios
    present for BOTH methods, so as more Taylor runs are added the figure
    widens automatically. Pass an explicit tuple to override.
    """
    l1_ratios  = sorted(r['_ratio'] for r in runs if r['_method'] == 'l1')
    tay_ratios = sorted(r['_ratio'] for r in runs if r['_method'] == 'taylor')
    if not l1_ratios:
        print('  [Fig 10] skipped — no L1 runs found.')
        return
    if shared_ratios is None:
        shared = sorted(set(l1_ratios) & set(tay_ratios))
        # If Taylor is missing/partial, fall back to L1's ratios so the figure
        # still conveys the L1 curve rather than collapsing to nothing.
        shared_ratios = shared if shared else l1_ratios
    if not tay_ratios:
        print('  [Fig 10] note — no Taylor runs; plotting L1 only.')
    elif set(l1_ratios) != set(tay_ratios):
        print(f'  [Fig 10] note — L1 and Taylor ratios differ; '
              f'plotting shared {shared_ratios}.')

    fig, (ax_a, ax_t) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for method in ('l1', 'taylor'):
        st = METHOD_STYLE[method]
        sel = sorted([r for r in runs
                      if r['_method'] == method and r['_ratio'] in shared_ratios],
                     key=lambda r: r['_ratio'])
        if not sel:
            continue
        ratios  = [r['_ratio'] for r in sel]
        aucs    = [_freihand_combined_auc(r, tau_min, tau_max) for r in sel]
        latency = [_scalar(r, 'latency') for r in sel]
        lat_std = [r['latency'].get('total_ms_std') for r in sel]
        ax_a.plot(ratios, aucs, marker=st['marker'], color=st['color'],
                  label=st['label'])
        # Latency carries a per-point std (batch-1 CPU timing) — show error caps
        # so overlapping bars make the L1≈Taylor speed equivalence explicit.
        ax_t.errorbar(ratios, latency, yerr=lat_std, fmt=st['marker'] + '-',
                      color=st['color'], label=st['label'],
                      capsize=3, capthick=0.8, elinewidth=0.8)

    # DetNet baseline (unpruned) as a horizontal reference on BOTH panels —
    # same style as Fig 6-9: dashed line + inline label, not a legend entry.
    # Label right-aligned at the rightmost ratio and dropped just below the line
    # so it sits in clear space away from the L1/Taylor curves.
    baseline = next((r for r in runs if r['_method'] == 'baseline'), None)
    if baseline is not None:
        x_right = shared_ratios[-1]
        fh_b = _freihand_combined_auc(baseline, tau_min, tau_max)
        if fh_b is not None:
            _annotate_baseline(ax_a, fh_b,
                               f'DetNet baseline ({fh_b:.2f}%)',
                               linestyle='--', x=x_right, ha='right', dy=-4)
        lat_b = _scalar(baseline, 'latency')
        _annotate_baseline(ax_t, lat_b,
                           f'DetNet baseline ({lat_b:.2f} ms)',
                           linestyle='--', x=x_right, ha='right', dy=-4)

    for ax in (ax_a, ax_t):
        ax.set_xlabel('Filter removal ratio (%)')
        ax.set_xticks(list(shared_ratios))
        ax.grid(True, axis='y', linewidth=0.3, alpha=0.3)
    ax_a.set_ylabel('Accuracy (AUC, %)')
    ax_t.set_ylabel('Latency(ms)')
    # Cap the accuracy panel at 70% so the dashed DetNet-baseline line sits
    # near the top of the visible range with breathing room above the data.
    _a_lo, _a_hi = ax_a.get_ylim()
    ax_a.set_ylim(_a_lo, 70)
    # Force y=10 as the lowest tick on the latency panel — break symbol sits
    # safely below it with no data overlap (same rationale as Fig 7).
    _cur_lo, _cur_hi = ax_t.get_ylim()
    if _cur_lo > 10:
        ax_t.set_ylim(10, _cur_hi)
    ax_a.legend(loc='best')
    # (a) / (b) subplot identifiers, drawn in matplotlib (NOT LaTeX subfigure
    # — wrapping this PDF in a subfigure would double-label it).
    ax_a.text(-0.13, 1.02, '(a)', transform=ax_a.transAxes,
              fontsize=12, fontweight='bold', va='bottom')
    ax_t.text(-0.13, 1.02, '(b)', transform=ax_t.transAxes,
              fontsize=12, fontweight='bold', va='bottom')

    fig.tight_layout()
    _add_axis_break_marks(fig)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  -> {out_path}')


# ── Fig 8+9 (combined): Model size + GFLOPs, L1 vs Taylor — 2-panel ─────────

SIZE_COLOR   = '#1B5E20'   # dark green — Fig 8 size panel / standalone
GFLOPS_COLOR = '#5D4037'   # brown      — Fig 8 GFLOPs panel / standalone Fig 9


def plot_l1_vs_taylor_compression(runs, out_path, shared_ratios=None):
    """Combined Fig 8 — Model size + GFLOPs in a 1×2 panel.

    L1 and Taylor produce IDENTICAL per-layer compression at each ratio
    (same filter count per layer ⇒ same parameter count and op count), so
    their size/GFLOPs curves overlap exactly. To avoid suggesting the
    methods differ on compute/storage, both methods are drawn in the SAME
    panel-specific colour (size=dark green, GFLOPs=brown) with no legend —
    overlapping markers read as a single line.

    Left  panel: Model size (MB) vs filter removal ratio — dark green.
    Right panel: GFLOPs vs filter removal ratio — brown.
    """
    l1_ratios  = sorted(r['_ratio'] for r in runs if r['_method'] == 'l1')
    tay_ratios = sorted(r['_ratio'] for r in runs if r['_method'] == 'taylor')
    if not l1_ratios:
        print('  [Fig 8/9] skipped — no L1 runs found.')
        return
    if shared_ratios is None:
        shared = sorted(set(l1_ratios) & set(tay_ratios))
        # Fall back to L1-only ratios if Taylor is partial, so the figure still
        # conveys the L1 curve rather than collapsing to nothing.
        shared_ratios = shared if shared else l1_ratios
    if not tay_ratios:
        print('  [Fig 8/9] note — no Taylor runs; plotting L1 only.')
    elif set(l1_ratios) != set(tay_ratios):
        print(f'  [Fig 8/9] note — L1 and Taylor ratios differ; '
              f'plotting shared {shared_ratios}.')

    fig, (ax_s, ax_g) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for method in ('l1', 'taylor'):
        st = METHOD_STYLE[method]
        sel = sorted([r for r in runs
                      if r['_method'] == method and r['_ratio'] in shared_ratios],
                     key=lambda r: r['_ratio'])
        if not sel:
            continue
        ratios = [r['_ratio']            for r in sel]
        sizes  = [_scalar(r, 'total_mb') for r in sel]
        flops  = [_scalar(r, 'gflops')   for r in sel]
        # Single colour per panel; no per-method label (no legend either).
        ax_s.plot(ratios, sizes, marker=st['marker'], color=SIZE_COLOR)
        ax_g.plot(ratios, flops, marker=st['marker'], color=GFLOPS_COLOR)

    baseline = next((r for r in runs if r['_method'] == 'baseline'), None)
    if baseline is not None:
        x_right = shared_ratios[-1]
        size_b = _scalar(baseline, 'total_mb')
        _annotate_baseline(ax_s, size_b,
                           f'DetNet baseline ({size_b:.2f} MB)',
                           linestyle='--', x=x_right, ha='right', dy=-4)
        flops_b = _scalar(baseline, 'gflops')
        if flops_b is not None:
            _annotate_baseline(ax_g, flops_b,
                               f'DetNet baseline ({flops_b:.3f} GFLOPs)',
                               linestyle='--', x=x_right, ha='right', dy=-4)

    for ax in (ax_s, ax_g):
        ax.set_xlabel('Filter removal ratio (%)')
        ax.set_xticks(list(shared_ratios))
        ax.grid(True, axis='y', linewidth=0.3, alpha=0.3)
    ax_s.set_ylabel('Model Size(MB)')
    ax_g.set_ylabel('GFLOPs')
    # No legend — L1 and Taylor are visually indistinguishable on these metrics
    # (overlapping curves), so a legend would only invite a non-existent comparison.
    # (a) / (b) subplot identifiers — drawn in matplotlib.
    ax_s.text(-0.13, 1.02, '(a)', transform=ax_s.transAxes,
              fontsize=12, fontweight='bold', va='bottom')
    ax_g.text(-0.13, 1.02, '(b)', transform=ax_g.transAxes,
              fontsize=12, fontweight='bold', va='bottom')

    fig.tight_layout()
    _add_axis_break_marks(fig)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  -> {out_path}')


# ── Fig 11: Quantization grouped bar chart ──────────────────────────────────

def plot_quantization_bars(fp32_run, int8_run, out_path, tau_min=20, tau_max=50):
    """Fig 11 — FP32 baseline vs INT8 PTQ across three metrics.

    Grouped bars: AUC FreiHAND (%), CPU inference time (ms), model size (MB).
    Each metric is min-max irrelevant — they're on different scales — so the
    three groups use independent normalised bar heights with the true value
    annotated on top of each bar.

    Data contract: two run dicts (loaded summary.json), one FP32 one INT8.
    INT8 quantization is future work; until a quantized run exists this is
    skipped with an actionable message.
    """
    if fp32_run is None or int8_run is None:
        print('  [Fig 11] skipped — needs both an FP32 and an INT8 run. '
              'Provide them via --fp32_run / --int8_run once PTQ is done.')
        return

    def metrics(run):
        return {
            'AUC FreiHAND (%)':   _freihand_combined_auc(run, tau_min, tau_max),
            'CPU latency (ms)':   _scalar(run, 'latency'),
            'Model size (MB)':    _scalar(run, 'total_mb'),
        }

    m_fp32, m_int8 = metrics(fp32_run), metrics(int8_run)
    labels = list(m_fp32.keys())
    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    # Each metric on its own normalised height (bar value / FP32 value) so the
    # three different units share one axis; real numbers annotated on the bars.
    fp32_h = [1.0] * len(labels)
    int8_h = [m_int8[k] / m_fp32[k] if m_fp32[k] else 0.0 for k in labels]

    b1 = ax.bar(x - w/2, fp32_h, w, color=METHOD_STYLE['baseline']['color'],
                label='FP32 baseline')
    b2 = ax.bar(x + w/2, int8_h, w, color=METHOD_STYLE['quantized']['color'],
                label='INT8 PTQ')

    for bars, mvals in ((b1, m_fp32), (b2, m_int8)):
        for rect, key in zip(bars, labels):
            ax.annotate(f'{mvals[key]:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Relative to FP32 (×)')
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    _add_axis_break_marks(fig)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  -> {out_path}')


# ── Fig 12: Pareto scatter (headline figure) ────────────────────────────────

# Slightly darker amber for Taylor LABEL TEXT — Okabe-Ito amber (#E69F00) used
# for the marker is too pale at 6pt to read cleanly against white.
_TAYLOR_LABEL_COLOR = '#B57700'
# MediaPipe disk footprint used to size its Pareto marker (item 8). 7.82 MB
# matches the canonical hand_landmarker.task size on disk.
_MEDIAPIPE_MB = 7.82


def plot_pareto(runs, out_path, tau_min=20, tau_max=50,
                mediapipe_auc=None, mediapipe_latency=None):
    """Fig 12 ★ — The whole compression landscape on one axis.

    x = GPU inference time (ms), y = FreiHAND AUC (%). One point per model:
    marker shape encodes method (METHOD_STYLE), colour encodes method, point
    AREA encodes model size (MB). Baseline DetNet and (optionally) MediaPipe
    are added as reference points.

    Per-point tags follow a layout rule designed so L1/Taylor labels at the
    SAME ratio (markers nearly overlap) never collide:
      - L1     -> label LEFT of marker, blue text
      - Taylor -> label RIGHT of marker, darker amber text
      - Top-AUC cluster (baseline / 10% / 25%) -> label BELOW the marker
        so it doesn't clip the top frame; everything else sits ABOVE.

    Two legends:
      - method   : top-left, marker shape/colour key
      - size ref : bottom-right, 5/25/45 MB reference markers so the
                   marker-area-encodes-MB mapping is quantitative.

    Quantized models are CPU-timed — flagged in the method legend so the
    x-axis mixing is explicit and honest.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Scale model-size (MB) → marker area. Reference ~ baseline size.
    # Kept deliberately small — points cluster tightly near the baseline so
    # large markers overlap and obscure the labels.
    sizes_mb = [_scalar(r, 'total_mb') for r in runs] or [40.0]
    smax = max(sizes_mb)
    def area(mb): return 16 + 120 * (mb / smax)   # pts^2

    seen_methods = set()
    for r in runs:
        m  = r['_method']
        st = METHOD_STYLE.get(m, METHOD_STYLE['l1'])
        y  = _freihand_combined_auc(r, tau_min, tau_max)
        xv = _scalar(r, 'latency')
        if y is None or xv is None:
            continue
        ax.scatter(xv, y, s=area(_scalar(r, 'total_mb')),
                   marker=st['marker'], facecolor=st['color'],
                   edgecolor='k', linewidth=0.5, alpha=0.85,
                   label=st['label'] if m not in seen_methods else None)
        seen_methods.add(m)

        # Per-point tag
        ratio = r['_ratio']
        if m in ('qmm', 'qmse'):
            tag = st['label'].split()[0]    # 'QMM' / 'QMSE'
        elif ratio:
            tag = f'{ratio}%'
        elif m == 'mediapipe':
            tag = 'MP'          # distinct marker/colour + legend say the rest
        else:
            tag = 'base'

        # Label placement: L1 to the LEFT, Taylor to the RIGHT so they don't
        # collide where the two methods' markers overlap. Top-cluster points
        # (AUC > 67.5% — baseline + 10% + 25%) get the label BELOW the marker
        # so it doesn't clip the top frame; everything else sits above. PTQ
        # models (QMM/QMSE) are EXEMPT from the cluster rule — QMSE at 67.51%
        # would otherwise get a below-marker label that collides with the
        # 55%-pruned L1/Taylor pair sitting just under it.
        in_top = (y is not None and y > 67.5) and m not in ('qmm', 'qmse')
        dy = -3 if in_top else 3
        va = 'top' if in_top else 'bottom'
        if m == 'l1':
            xytext, ha, text_color = (-4, dy), 'right', COLORS['blue']
        elif m == 'taylor':
            xytext, ha, text_color = (4, dy),  'left',  _TAYLOR_LABEL_COLOR
        else:
            xytext, ha, text_color = (4, dy),  'left',  'k'
        ax.annotate(tag, xy=(xv, y), xytext=xytext,
                    textcoords='offset points', fontsize=6,
                    ha=ha, va=va, color=text_color)

    if mediapipe_auc is not None and mediapipe_latency is not None:
        st = METHOD_STYLE['mediapipe']
        ax.scatter(mediapipe_latency, mediapipe_auc, s=area(_MEDIAPIPE_MB),
                   marker=st['marker'], facecolor=st['color'],
                   edgecolor='k', linewidth=0.5,
                   label=st['label'])
        ax.annotate('MP', xy=(mediapipe_latency, mediapipe_auc), xytext=(4, 3),
                    textcoords='offset points', fontsize=6,
                    ha='left', va='bottom')

    ax.set_xlabel('Latency(ms)')
    ax.set_ylabel('Accuracy (AUC, %)')
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.3)

    # Two legends side by side at the bottom-right: method (rightmost), then
    # marker-size reference immediately to its left at the same vertical
    # height. The first ax.legend() call is "remembered" by ax; ax.add_artist
    # keeps it visible while the second legend() call installs the size ref.
    from matplotlib.lines import Line2D
    # Method legend uses FIXED-size proxy markers (markersize=9) instead of
    # the auto-picked first-scatter size — otherwise baseline/L1/Taylor
    # (high-MB → ~12pt) overlap vertically while MediaPipe/QMM/QMSE (low-MB
    # → ~6pt) look anaemic. Size encoding is communicated separately by the
    # "Model size" legend below, so the method legend doesn't need to repeat it.
    methods_in_plot = set(seen_methods)
    if mediapipe_auc is not None and mediapipe_latency is not None:
        methods_in_plot.add('mediapipe')
    method_order = ['baseline', 'l1', 'taylor', 'mediapipe', 'qmm', 'qmse']
    # Per-method override — circles (L1) and diamonds (Taylor) visually fill
    # more space at the same nominal markersize than the other shapes
    # (plus, X, square, triangle), so they need a smaller nominal size to
    # read as the same visual weight.
    _legend_msize = {'l1': 7.5, 'taylor': 7.5}
    method_handles = [
        Line2D([0], [0], marker=METHOD_STYLE[m]['marker'], linestyle='',
               markersize=_legend_msize.get(m, 9),
               markerfacecolor=METHOD_STYLE[m]['color'],
               markeredgecolor='k', markeredgewidth=0.5,
               label=METHOD_STYLE[m]['label'])
        for m in method_order if m in methods_in_plot
    ]
    method_legend = ax.legend(handles=method_handles, loc='lower right',
                              title='Method', title_fontsize=7, fontsize=7,
                              framealpha=0.95)
    ax.add_artist(method_legend)
    # Marker-size reference legend (item 7): 3 reference markers at 5/25/45 MB
    # so "marker area ~ model size" becomes quantitative, not just relative.
    # bbox_to_anchor right edge at axes-frac x=0.78 sits the size legend just
    # to the LEFT of the method legend (which spans roughly the rightmost
    # ~22% of the axes width).
    ref_sizes = [5, 25, 45]
    ref_handles = [
        Line2D([0], [0], marker='o', linestyle='',
               markersize=np.sqrt(area(mb)),
               markerfacecolor='lightgray',
               markeredgecolor='k', markeredgewidth=0.4,
               label=f'{mb} MB')
        for mb in ref_sizes
    ]
    ax.legend(handles=ref_handles,
              bbox_to_anchor=(0.85, 0.0), loc='lower right',
              title='Marker area ~ Model size', title_fontsize=7, fontsize=6,
              framealpha=0.9, labelspacing=1.4, borderpad=0.8)

    fig.tight_layout()
    _add_axis_break_marks(fig)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  -> {out_path}')


# ── Fig 14: Qualitative keypoint-overlay grid ───────────────────────────────

def plot_qualitative_overlay(qual_npz, out_path):
    """Fig 14 — 4×3 grid: 4 FreiHAND images × {baseline, best L1, best Taylor}.

    Predicted joints drawn as coloured dots connected by SNAP_BONES; each
    sub-image titled with model name + AUC.

    Data contract — qual_npz is an .npz produced by a small dump step
    (future: an evaluate_detnet.py --dump_qualitative flag) containing:
      images : (4, H, W, 3) uint8   — the 4 chosen FreiHAND frames
      models : (3,) str             — e.g. ['DetNet', 'L1-25%', 'Taylor-25%']
      aucs   : (3,) float           — per-model AUC for the subtitle
      preds  : (4, 3, 21, 2) float  — predicted kp2d per image per model
    Pick ≥1 challenging frame (occlusion / unusual pose) when selecting images.
    """
    if qual_npz is None or not os.path.isfile(qual_npz):
        print('  [Fig 14] skipped — needs a qualitative-predictions .npz '
              '(images/models/aucs/preds). See docstring for the contract.')
        return

    d      = np.load(qual_npz, allow_pickle=True)
    images = d['images']; models = d['models']; aucs = d['aucs']; preds = d['preds']
    n_img, n_mod = images.shape[0], len(models)

    fig, axes = plt.subplots(n_img, n_mod,
                             figsize=(2.3 * n_mod, 2.3 * n_img))
    finger_colors = [COLORS['red'], COLORS['blue'], COLORS['green'],
                     COLORS['orange'], COLORS['purple']]
    for i in range(n_img):
        for j in range(n_mod):
            ax = axes[i, j] if n_img > 1 else axes[j]
            ax.imshow(images[i]); ax.set_xticks([]); ax.set_yticks([])
            kp = preds[i, j]
            for b, (a, c) in enumerate(SNAP_BONES):
                ax.plot([kp[a, 0], kp[c, 0]], [kp[a, 1], kp[c, 1]],
                        '-', color=finger_colors[b // 4], linewidth=1.0)
            ax.scatter(kp[:, 0], kp[:, 1], s=8, c='white',
                       edgecolor='k', linewidth=0.4, zorder=3)
            if i == 0:
                ax.set_title(f'{models[j]}\nAUC {aucs[j]:.1f}%', fontsize=8)

    fig.tight_layout()
    _add_axis_break_marks(fig)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  -> {out_path}')


# ── Fig 15: Heatmap precision-loss comparison ───────────────────────────────

def plot_heatmap_comparison(heatmap_npz, out_path):
    """Fig 15 — One FreiHAND image; max-across-channels heatmap from the
    baseline vs the most aggressively pruned model (70% L1). Shows spatial
    precision loss at extreme compression.

    Data contract — heatmap_npz is an .npz containing:
      image    : (H, W, 3) uint8                 — the chosen frame
      hmap_base: (32, 32) float                  — max over 21 channels, baseline
      hmap_pruned: (32, 32) float                — same, 70% L1 model
      label_pruned : str                         — e.g. 'L1 70%'
    Produced by a future evaluate_detnet.py --dump_heatmap step.
    """
    if heatmap_npz is None or not os.path.isfile(heatmap_npz):
        print('  [Fig 15] skipped — needs a heatmap .npz '
              '(image/hmap_base/hmap_pruned). See docstring for the contract.')
        return

    d = np.load(heatmap_npz, allow_pickle=True)
    img   = d['image']
    hb, hp = d['hmap_base'], d['hmap_pruned']
    lbl   = str(d['label_pruned']) if 'label_pruned' in d.files else 'pruned'

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    axes[0].imshow(img);                axes[0].set_title('Input', fontsize=9)
    axes[1].imshow(hb, cmap='inferno'); axes[1].set_title('Baseline heatmap', fontsize=9)
    axes[2].imshow(hp, cmap='inferno'); axes[2].set_title(f'{lbl} heatmap', fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    _add_axis_break_marks(fig)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  -> {out_path}')


# ── Configuration ───────────────────────────────────────────────────────────
# Edit these, then toggle the figureN() calls in the __main__ block below.

RESULTS_DIR       = 'eval_results_cpu'      # CPU / FreiHAND-eval runs (08_eval_all_cpu.py + 07 bridge)
OUT_DIR           = 'figures'               # where PDFs are saved
TAU_MIN           = 0
TAU_MAX           = 30
MEDIAPIPE_AUC     = None    # e.g. 85.0  — Fig 6 reference line + Fig 12 point
MEDIAPIPE_LATENCY = None    # e.g. 4.0   — ms, Fig 12 point
FP32_RUN          = None    # Fig 11 — path to an FP32 run dir or summary.json
INT8_RUN          = None    # Fig 11 — path to an INT8 PTQ run dir or summary.json
QUAL_DATA         = None    # Fig 14 — qualitative-predictions .npz
HEATMAP_DATA      = None    # Fig 15 — heatmap-comparison .npz


# ── Shared helpers ──────────────────────────────────────────────────────────

_RUNS = None

def _runs():
    """Load + cache all runs once (errors.npz reads aren't free, and several
    figures need the same runs)."""
    global _RUNS
    if _RUNS is None:
        _RUNS = load_runs(RESULTS_DIR)
        if not _RUNS:
            print(f'WARNING: no summary.json found under {RESULTS_DIR}/')
        else:
            print(f'Loaded {len(_RUNS)} run(s) from {RESULTS_DIR}/:')
            for r in _RUNS:
                print(f'  {r["_method"]:8s} {r["_ratio"]:3d}pct  <- {r["_run_dir"]}')
    return _RUNS


def _methods():
    """Pruning methods for the per-ratio figures (Fig 6-9, 13). Only L1 and Taylor
    sweep the filter-removal ratio; baseline / quant (QMM, QMSE) / mediapipe are
    single points and appear only in the Pareto scatter (Fig 12)."""
    return sorted({r['_method'] for r in _runs() if r['_method'] in ('l1', 'taylor')})


def _out(name):
    """Resolve an output path under OUT_DIR, creating the folder on demand."""
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def _load_single_run(path):
    """Load one summary.json (dir or file path) into a run dict for Fig 11."""
    if not path:
        return None
    jp = path if path.endswith('.json') else os.path.join(path, 'summary.json')
    if not os.path.isfile(jp):
        print(f'  run not found: {jp}')
        return None
    with open(jp, 'r', encoding='utf-8') as f:
        d = json.load(f)
    rd = os.path.dirname(jp)
    d['_run_dir']     = rd
    d['_errors_path'] = os.path.join(rd, 'errors.npz')
    m, rt = parse_method_ratio(d.get('checkpoint', ''))
    d['_method'], d['_ratio'] = m, rt
    return d


# ── One function per figure — toggle these in __main__ ──────────────────────

def figure5():
    """Fig 5 — PCK curves: baseline + L1 (×5 ratios) + Taylor (×5 ratios)."""
    print('[Fig 5] PCK curves (baseline + L1 + Taylor)')
    plot_pck_curves(_runs(), _out('fig05_pck_curves.pdf'),
                    tau_min=TAU_MIN, tau_max=TAU_MAX)


def figure6():
    """Fig 6 — Accuracy (AUC) vs filter removal ratio. One PDF per method."""
    print('[Fig 6] accuracy vs ratio')
    for m in _methods():
        plot_accuracy_vs_ratio(
            _runs(), _out(f'fig06_accuracy_vs_ratio_{m}.pdf'),
            method=m, tau_min=TAU_MIN, tau_max=TAU_MAX,
            mediapipe_auc=MEDIAPIPE_AUC)


def figure7():
    """Fig 7 — GPU inference time vs filter removal ratio. One PDF per method."""
    print('[Fig 7] inference time vs ratio')
    for m in _methods():
        plot_single_metric_vs_ratio(
            _runs(), _out(f'fig07_inference_time_vs_ratio_{m}.pdf'),
            method=m, metric='latency', ylabel='Latency(ms)',
            baseline_unit_fmt='{value:.2f} ms',
            force_ymin=10)


def figure8():
    """Fig 8 — Model size: 1×2 side-by-side (L1 ≡ Taylor, single colour per
    panel) AND a standalone single-line model-size-vs-ratio plot."""
    print('[Fig 8] model size + GFLOPs side-by-side (L1 ≡ Taylor)')
    plot_l1_vs_taylor_compression(
        _runs(), _out('fig08_size_gflops_l1_vs_taylor.pdf'))
    print('[Fig 8] standalone model size vs ratio (L1, dark green)')
    plot_single_metric_vs_ratio(
        _runs(), _out('fig08_model_size_vs_ratio.pdf'),
        method='l1', metric='total_mb', ylabel='Model Size(MB)',
        baseline_unit_fmt='{value:.2f} MB',
        color=SIZE_COLOR)


def figure9():
    """Fig 9 — GFLOPs vs ratio. L1 ≡ Taylor on op count (same per-layer
    pruning), so a single brown line stands in for both methods."""
    print('[Fig 9] GFLOPs vs ratio (L1, brown)')
    plot_single_metric_vs_ratio(
        _runs(), _out('fig09_gflops_vs_ratio.pdf'),
        method='l1', metric='gflops', ylabel='GFLOPs',
        baseline_unit_fmt='{value:.3f} GFLOPs',
        color=GFLOPS_COLOR)


def figure10():
    """Fig 10 — L1 vs Taylor, 2-panel accuracy + latency (auto shared ratios)."""
    print('[Fig 10] L1 vs Taylor')
    plot_l1_vs_taylor(_runs(), _out('fig10_l1_vs_taylor.pdf'),
                      tau_min=TAU_MIN, tau_max=TAU_MAX)


def figure11():
    """Fig 11 — FP32 vs INT8 quantization bars. Needs FP32_RUN / INT8_RUN."""
    print('[Fig 11] quantization bars')
    plot_quantization_bars(
        _load_single_run(FP32_RUN), _load_single_run(INT8_RUN),
        _out('fig11_quantization_bars.pdf'),
        tau_min=TAU_MIN, tau_max=TAU_MAX)


def figure12():
    """Fig 12 ★ — Pareto scatter, the headline compression-landscape figure."""
    print('[Fig 12] Pareto landscape')
    plot_pareto(_runs(), _out('fig12_pareto.pdf'),
                tau_min=TAU_MIN, tau_max=TAU_MAX,
                mediapipe_auc=MEDIAPIPE_AUC,
                mediapipe_latency=MEDIAPIPE_LATENCY)


def figure14():
    """Fig 14 — Qualitative keypoint-overlay grid. Needs QUAL_DATA .npz."""
    print('[Fig 14] qualitative overlay')
    plot_qualitative_overlay(QUAL_DATA, _out('fig14_qualitative_overlay.pdf'))


def figure15():
    """Fig 15 — Heatmap precision-loss comparison. Needs HEATMAP_DATA .npz."""
    print('[Fig 15] heatmap comparison')
    plot_heatmap_comparison(HEATMAP_DATA, _out('fig15_heatmap_comparison.pdf'))


if __name__ == '__main__':
    # Comment out any figure you don't want to (re)generate this run.
    figure5()
    figure6()
    figure7()
    figure8()
    figure9()
    figure10()
    figure11()
    figure12()
    figure14()
    figure15()
