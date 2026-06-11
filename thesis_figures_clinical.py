"""
thesis_figures_clinical.py — two publication-ready thesis figures for the
clinical-application chapter:

  (a) pareto_auc_size      : AUC vs. model size — Pareto frontier (Figure 1)
  (b) perjoint_mpjpe       : Per-joint normalised MPJPE for the 55% pruned +
                             PTQ models (Figure 2)

Outputs go to figures/{name}.{pdf,png} at 1000 DPI. Real data is read from
eval_results_MPJPE/*/summary.json (+ errors.npz) when available (bbox-norm
re-run), with eval_results_cpu/ as fallback for runs that weren't part of the
bbox-norm re-run. If any input is missing a warning is printed and placeholder
values are used so the script still runs.

Captions are printed to stdout for copy-paste into the thesis document.
"""
import json
import os

# Keep this BEFORE numpy/matplotlib so torch-built OpenMP doesn't double-load
# (Fig 2 doesn't actually need torch, but consistent setup is cheap insurance).
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import matplotlib.pyplot as plt


_BREAK_STRAIGHT_TOP    = 0.15
_BREAK_STRAIGHT_BOT    = 0.05
_BREAK_NOTCH_SPAN      = 0.6
_BREAK_NOTCH_DEPTH     = 0.015
_BREAK_SLASH_V         = 0.025
_BREAK_SLASH_OFFSET_V  = 0.014


def _add_axis_break_marks(fig):
    """Extend the LEFT y-axis below its lowest visible tick to host an
    integrated '>' break notch:
        |  3 tick-spaces of straight extension
        >  the notch
        |  2 tick-spaces of straight extension
        +-- x-axis corner
    The left spine is hidden and replaced by one polyline that traces the
    above, inheriting the spine's colour + linewidth so the notch reads as
    part of the axis line. Skips image / log-scale / hidden-spine / zero-
    origin axes, and axes with <2 visible ticks."""
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
        # Fixed visual size — keeps the '//' symbol consistent across plots.
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


# ── Style ────────────────────────────────────────────────────────────────────

COLORS = {
    'l1':        '#2E6FA3',   # blue
    'taylor':    '#E8A020',   # orange
    'qmm':       '#1A6B5A',   # teal
    'qmse':      '#7B2D8B',   # purple
    'baseline':  '#555555',   # dark grey
    'mediapipe': '#64B5F6',   # light blue (dashed reference line in Fig 4)
    'highlight': '#D62828',   # red callout
}
SPINE_COLOR  = '#AAAAAA'
GRID_COLOR   = '#DDDDDD'

# Joints — MediaPipe order (matches SNAP order used by DetNet / errors.npz).
# Thumb labels (PIP/DIP) follow the reference figure convention; anatomically
# the thumb has CMC/MCP/IP joints (no PIP/DIP) — adjust if needed.
JOINT_NAMES = [
    'Wrist',                                                          # 0
    'Thumb CMC',  'Thumb MCP',  'Thumb IP',   'Thumb Tip',             # 1-4
    'Index MCP',  'Index PIP',  'Index DIP',  'Index Tip',             # 5-8
    'Middle MCP', 'Middle PIP', 'Middle DIP', 'Middle Tip',            # 9-12
    'Ring MCP',   'Ring PIP',   'Ring DIP',   'Ring Tip',              # 13-16
    'Pinky MCP',  'Pinky PIP',  'Pinky DIP',  'Pinky Tip',             # 17-20
]
# (label, joint_idx_lo, joint_idx_hi, background_color)
FINGER_GROUPS = [
    ('Thumb',  1,  4,  '#E8E8F8'),
    ('Index',  5,  8,  '#E8F4E8'),
    ('Middle', 9,  12, '#FFF4E0'),
    ('Ring',   13, 16, '#F4E8F8'),
    ('Pinky',  17, 20, '#F0F0F0'),
]

# Try the bbox-norm dir first (eval_results_MPJPE/ holds the 6 re-run models
# with per-frame norm arrays in errors.npz), then fall back to eval_results_cpu/
# for the 8 prune ratios that weren't part of the bbox-norm re-run. Figure 1's
# AUC is invariant to normalisation so it doesn't care which dir it reads from.
EVAL_ROOTS = ['eval_results_MPJPE', 'eval_results_cpu']
RATIOS    = [10, 25, 40, 55, 70]
NORM_PX   = 224.0                  # FreiHAND image size; legacy fallback only


# ── Data loading ────────────────────────────────────────────────────────────

def _load_run(run_dir):
    """Read one <eval_root>/<run>/ -> dict, or None if absent.
    Searches EVAL_ROOTS in order; first match wins."""
    for root in EVAL_ROOTS:
        p = os.path.join(root, run_dir)
        sj = os.path.join(p, 'summary.json')
        if not os.path.isfile(sj):
            continue
        ej = os.path.join(p, 'errors.npz')
        with open(sj, 'r', encoding='utf-8') as f:
            s = json.load(f)
        return {
            'summary':     s,
            'size_mb':     s.get('model', {}).get('total_mb'),
            'errors_path': ej if os.path.isfile(ej) else None,
        }
    return None


def _pooled_errors(run):
    """Concatenate every freihand_* split's per-joint pixel errors -> (N,)."""
    if run is None or not run['errors_path']:
        return None
    z = np.load(run['errors_path'])
    parts = [z[k] for k in z.files if k.startswith('freihand')]
    return np.concatenate(parts) if parts else None


def _auc_from_errors(errs, tau_min=0, tau_max=30):
    """Same integration as evaluate_detnet (and plot_pruning_results)."""
    taus = np.arange(tau_min, tau_max + 1, dtype=float)
    pck = np.array([min(float((errs < t).mean()) * 100.0, 100.0) for t in taus])
    return float(np.trapz(pck, taus)) / (tau_max - tau_min)


def _per_joint_norm_mpjpe(run, n_joints=21):
    """Mean per-joint bbox-normalised error, Simon et al. 2017 convention.
    Each frame's per-joint pixel error is divided by that frame's own
    max(bbox_w, bbox_h) BEFORE averaging across frames, so the metric is
    scale-invariant (closer/farther hands contribute on equal footing).

    Falls back to image-size normalisation (legacy /NORM_PX) if errors.npz
    was produced by a pre-bbox-norm eval run (no norm_freihand_* arrays).
    Returns None if data missing or the layout is non-standard."""
    if run is None or not run['errors_path']:
        return None
    z = np.load(run['errors_path'])
    split_keys = sorted(k for k in z.files if k.startswith('freihand'))
    if not split_keys:
        return None
    err_parts = [z[k] for k in split_keys]
    errs      = np.concatenate(err_parts)
    if errs.size % n_joints != 0:
        return None
    errs_2d   = errs.reshape(-1, n_joints)              # (N_frames, 21)

    # Per-frame norm arrays are saved as norm_<split> alongside <split>.
    norm_keys = [f'norm_{k}' for k in split_keys]
    if all(nk in z.files for nk in norm_keys):
        norms = np.concatenate([z[nk] for nk in norm_keys])  # (N_frames,)
        if norms.size == errs_2d.shape[0]:
            return (errs_2d / norms[:, None]).mean(axis=0)

    # Legacy fallback: image-size normalisation.
    return errs_2d.mean(axis=0) / NORM_PX


def load_all():
    base   = _load_run('ckp_detnet_71_both')
    prune  = {m: {} for m in ('l1', 'taylor')}
    for m in prune:
        for r in RATIOS:
            run = _load_run(f'ft_{m}_{r:02d}pct_rhdbest_both')
            if run is not None:
                prune[m][r] = run
    qmm    = _load_run('qmm_both')
    qmse   = _load_run('qmse_both')
    mp     = _load_run('mediapipe_hands_both')

    missing = []
    if base is None:                                            missing.append('baseline')
    if qmm  is None:                                            missing.append('qmm')
    if qmse is None:                                            missing.append('qmse')
    if mp   is None:                                            missing.append('mediapipe')
    for m in prune:
        for r in RATIOS:
            if r not in prune[m]:
                missing.append(f'{m}@{r}%')
    if missing:
        print(f'  WARNING: missing runs ({", ".join(missing)}); '
              f'placeholders will be used where needed.')
    return base, prune, qmm, qmse, mp


# ── Shared axes styling ─────────────────────────────────────────────────────

def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(SPINE_COLOR)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors='#333333', labelsize=9)
    ax.grid(True, axis='both', linewidth=0.6, color=GRID_COLOR, zorder=0)


# ── Figure 1: AUC vs Model Size — Pareto frontier ──────────────────────────

# Placeholder values used if any data is missing.
PLACEHOLDER_L1_SIZES   = [35.6, 25.0, 15.8, 9.3, 4.1]
PLACEHOLDER_TY_SIZES   = PLACEHOLDER_L1_SIZES[:]
PLACEHOLDER_L1_AUC     = [69.4, 69.2, 68.6, 66.8, 62.3]
PLACEHOLDER_TY_AUC     = [69.5, 69.1, 68.3, 66.5, 63.5]
PLACEHOLDER_BASE_SIZE  = 44.0
PLACEHOLDER_BASE_AUC   = 68.5
PLACEHOLDER_MP_AUC     = 68.4
PLACEHOLDER_INT8_SIZE  = 11.4
PLACEHOLDER_QMM_AUC    = 62.2
PLACEHOLDER_QMSE_AUC   = 67.5


def figure1(base, prune, qmm, qmse, mp, out_dir):
    # Resolve data (real or placeholder per series).
    def _seq(m):
        if all(r in prune[m] for r in RATIOS):
            sizes = [prune[m][r]['size_mb'] for r in RATIOS]
            aucs  = [_auc_from_errors(_pooled_errors(prune[m][r])) for r in RATIOS]
            return sizes, aucs, True
        return (PLACEHOLDER_L1_SIZES if m == 'l1' else PLACEHOLDER_TY_SIZES,
                PLACEHOLDER_L1_AUC   if m == 'l1' else PLACEHOLDER_TY_AUC,
                False)

    L1_sizes,  L1_aucs,  l1_real = _seq('l1')
    TY_sizes,  TY_aucs,  ty_real = _seq('taylor')
    if not (l1_real and ty_real):
        print('  Figure 1: using PLACEHOLDER values for L1/Taylor.')

    base_size = base['size_mb']                if base else PLACEHOLDER_BASE_SIZE
    base_auc  = _auc_from_errors(_pooled_errors(base)) if base else PLACEHOLDER_BASE_AUC
    qmm_size  = qmm['size_mb']                 if qmm  else PLACEHOLDER_INT8_SIZE
    qmm_auc   = _auc_from_errors(_pooled_errors(qmm))  if qmm  else PLACEHOLDER_QMM_AUC
    qmse_size = qmse['size_mb']                if qmse else PLACEHOLDER_INT8_SIZE
    qmse_auc  = _auc_from_errors(_pooled_errors(qmse)) if qmse else PLACEHOLDER_QMSE_AUC
    mp_auc    = _auc_from_errors(_pooled_errors(mp))   if mp   else PLACEHOLDER_MP_AUC
    int8_size = (qmm_size + qmse_size) / 2.0

    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
    _style_axes(ax)

    y_lo = min(L1_aucs + TY_aucs + [qmm_auc, qmse_auc, mp_auc]) - 2
    y_hi = max(L1_aucs + TY_aucs + [base_auc]) + 1
    ax.set_ylim(y_lo, y_hi)

    # Shade region where models BEAT MediaPipe (above the MediaPipe AUC line).
    ax.axhspan(mp_auc, y_hi, color='#A8E6A1', alpha=0.12, zorder=1)

    # FP32 horizontal reference.
    ax.axhline(base_auc, linestyle='--', color=COLORS['baseline'],
               linewidth=0.8, zorder=2)
    ax.text(base_size, base_auc + 0.12,
            f'DetNet FP32 ({base_size:.0f} MB)',
            fontsize=8, color=COLORS['baseline'], ha='right', va='bottom')

    # MediaPipe horizontal reference.
    ax.axhline(mp_auc, linestyle='--', color=COLORS['mediapipe'],
               linewidth=0.8, zorder=2)
    ax.text(max(L1_sizes) + 1.5, mp_auc + 0.12,
            f'MediaPipe ({mp_auc:.1f}%)',
            fontsize=8, color=COLORS['mediapipe'], ha='left', va='bottom')

    # INT8 vertical reference.
    ax.axvline(int8_size, linestyle='--', color='#888888',
               linewidth=0.8, zorder=2)
    ax.text(int8_size + 0.4, y_lo + 0.6, f'INT8 (~{int8_size:.0f} MB)',
            fontsize=8, color='#666666', rotation=90, va='bottom')

    # L1 and Taylor lines + markers.
    ax.plot(L1_sizes, L1_aucs, '-', color=COLORS['l1'], linewidth=1.2, zorder=3)
    ax.scatter(L1_sizes, L1_aucs, marker='o', color=COLORS['l1'], s=60,
               edgecolors='white', linewidths=0.8, zorder=5, label='L1 norm')
    ax.plot(TY_sizes, TY_aucs, '-', color=COLORS['taylor'], linewidth=1.2, zorder=3)
    ax.scatter(TY_sizes, TY_aucs, marker='D', color=COLORS['taylor'], s=55,
               edgecolors='white', linewidths=0.8, zorder=5, label='Taylor')

    # PTQ single points.
    ax.scatter([qmm_size],  [qmm_auc],  marker='s', color=COLORS['qmm'],  s=95,
               edgecolors='white', linewidths=0.8, zorder=5,
               label='PTQ MinMax (M2)')
    ax.scatter([qmse_size], [qmse_auc], marker='^', color=COLORS['qmse'], s=110,
               edgecolors='white', linewidths=0.8, zorder=5,
               label='PTQ Hist-MSE (M3)')

    # Selected-model highlight: enlarged red ring around the two 55% points,
    # plus a single annotation pointing toward the cluster.
    sel_i = RATIOS.index(55)
    sel_xs = [L1_sizes[sel_i], TY_sizes[sel_i]]
    sel_ys = [L1_aucs[sel_i],  TY_aucs[sel_i]]
    ax.scatter(sel_xs, sel_ys, marker='o', s=220, facecolors='none',
               edgecolors=COLORS['highlight'], linewidths=1.6, zorder=6)
    cx, cy = float(np.mean(sel_xs)), float(np.mean(sel_ys))
    ax.annotate('Selected model',
                xy=(cx + 0.8, cy - 0.15),
                xytext=(cx + 8, cy - 2.5),
                fontsize=9, color=COLORS['highlight'], fontweight='bold',
                arrowprops={'arrowstyle': '->',
                            'color': COLORS['highlight'],
                            'lw': 1.0})

    # Two-line x-axis ticks: row 1 = size in MB, row 2 = pruning ratio label.
    # Align to L1/Taylor points (they share sizes -> single tick set).
    ax.set_xticks(L1_sizes)
    pr_labels = [f'{r}% PR' for r in RATIOS]
    ax.set_xticklabels([f'{s:.1f}\n{pr}' for s, pr in zip(L1_sizes, pr_labels)],
                       fontsize=9)
    # Keep size + PR ticks visible; ensure axis label sits below them.
    ax.tick_params(axis='x', which='major', pad=4)

    ax.set_xlabel('Model Size(MB)', fontsize=10, labelpad=14)
    ax.set_ylabel('Accuracy (AUC,%)', fontsize=10)
    ax.set_title('(a) AUC vs. model size — Pareto frontier',
                 fontsize=11, fontweight='bold', loc='left', pad=10)

    leg = ax.legend(loc='lower right', fontsize=8.5,
                    framealpha=0.95, edgecolor=SPINE_COLOR)
    leg.get_frame().set_linewidth(0.6)
    _add_axis_break_marks(fig)

    out = os.path.join(out_dir, 'pareto_auc_size')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=1000, bbox_inches='tight',
                    facecolor='white')
        print(f'  -> {out}.{ext}')
    plt.close(fig)

    print('\nCaption (Figure 1):\n'
          '  Accuracy–size Pareto frontier. Each point is a pruned or '
          'quantized model. The 55% pruned models are selected as the best '
          'trade-off between accuracy and size. PTQ quantization (M2/M3) '
          'further reduces the selected models to ~11 MB with minimal '
          'accuracy loss.\n')


# ── Figure 2: Per-joint normalised MPJPE ───────────────────────────────────


def figure2(base, prune, qmm, qmse, mp, out_dir):
    # Six lines: FP32 baseline, 55% pruned L1/Taylor, PTQ M2/M3, MediaPipe ref.
    # Each tuple: (label, run, color, marker, linewidth, linestyle).
    series = [
        ('DetNet FP32',     base,                      COLORS['baseline'],  's', 2.0, '-'),
        ('L1 55%',          prune['l1'].get(55),       COLORS['l1'],        'o', 1.8, '-'),
        ('Taylor 55%',      prune['taylor'].get(55),   COLORS['taylor'],    'D', 1.8, '-'),
        ('PTQ MinMax',      qmm,                       COLORS['qmm'],       '^', 1.8, '-'),
        ('PTQ Hist-MSE',    qmse,                      COLORS['qmse'],      'v', 1.8, '-'),
        ('MediaPipe Hands', mp,                        COLORS['mediapipe'], 'x', 1.4, '--'),
    ]

    fig, ax = plt.subplots(figsize=(16, 7), facecolor='white')
    _style_axes(ax)
    # Horizontal-only grid (matches reference figure style).
    ax.xaxis.grid(False)
    x = np.arange(21)

    # Finger group background shading (drawn first so it sits under everything).
    for label, lo, hi, color in FINGER_GROUPS:
        ax.axvspan(lo - 0.5, hi + 0.5, color=color, alpha=0.18, zorder=0)

    series_data = []
    for label, run, color, marker, lw, ls in series:
        per_j = _per_joint_norm_mpjpe(run)
        if per_j is None:
            raise RuntimeError(
                f'Figure 4: per-joint data missing for "{label}". '
                f'Check that the corresponding run exists in eval_results_MPJPE/ '
                f'(or eval_results_cpu/) with a valid errors.npz.')
        series_data.append((label, per_j, color, marker, lw, ls))

    # Fixed y-range padded around the observed bbox-normalised band. Under
    # Simon-2017 bbox-norm the per-joint MPJPE sits in roughly [0.08, 0.20] for
    # baseline / pruned / MediaPipe and stretches a bit higher for PTQ tips.
    # Tighten after seeing the actual figure.
    y_lo, y_hi = 0.075, 0.230

    # Cursor-control ROI columns at joints 4 (Thumb Tip) and 8 (Index Tip).
    # Filled pink axvspan only — borders made it read as a selection box
    # rather than a subtle annotation. Sits above finger-group shading
    # (zorder=0) and below the data lines (zorder=4).
    for jx in (4, 8):
        ax.axvspan(jx - 0.5, jx + 0.5, color='#FFCCCC', alpha=0.50, zorder=1)

    # Wrist | fingers separator.
    ax.axvline(0.5, linestyle=':', color='#999999', linewidth=0.8, zorder=2)

    # Plot lines. 'x' is an unfilled marker (two crossing lines); white
    # outline would erase it, so the marker edge keeps the line colour and
    # uses a slightly thicker stroke for legibility.
    for label, per_j, color, marker, lw, ls in series_data:
        is_x_marker = marker == 'x'
        ax.plot(x, per_j, marker=marker, color=color, label=label,
                linewidth=lw, linestyle=ls,
                markersize=7 if is_x_marker else 6,
                markeredgecolor=color if is_x_marker else 'white',
                markeredgewidth=1.2 if is_x_marker else 0.6,
                zorder=4)

    # Finger group labels at the top of the panel (after y_lo/y_hi are set).
    ax.set_ylim(y_lo, y_hi)
    for label, lo, hi, _ in FINGER_GROUPS:
        ax.text((lo + hi) / 2, y_hi - 0.0025, label,
                ha='center', va='top', fontsize=9.5, color='#888888',
                fontstyle='italic')

    # X-tick labels: full joint names rotated 45°, right-aligned so the end
    # of each label sits under its tick (standard pattern for angled ticks).
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', va='top',
                       fontsize=9)
    ax.set_xlim(-0.7, 20.7)

    ax.set_ylabel('Normalised MPJPE', fontsize=10)
    ax.set_title('Per joint normalised MPJPE',
                 fontsize=11, fontweight='bold', loc='left', pad=10)

    leg = ax.legend(loc='lower right', ncol=3, fontsize=8.5,
                    framealpha=0.95, edgecolor=SPINE_COLOR)
    leg.get_frame().set_linewidth(0.6)
    _add_axis_break_marks(fig)

    out = os.path.join(out_dir, 'figure4_perjointMPJPE')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=1000, bbox_inches='tight',
                    facecolor='white')
        print(f'  -> {out}.{ext}')
    plt.close(fig)

    print('Caption (Figure 2):\n'
          '  Per-joint normalised MPJPE for the FP32 baseline and four '
          'compressed models (55% pruned L1, 55% pruned Taylor, PTQ M2, '
          'PTQ M3). Joints 4 (Thumb Tip) and 8 (Index Tip) are highlighted '
          'as the cursor control landmarks used in the stroke rehabilitation '
          'application. Compression consistently degrades accuracy at distal '
          'fingertip joints relative to proximal MCP and wrist joints.\n')


# ── Figure 3: per-joint MPJPE collapsed into 5 joint groups (2x2) ───────────
# Additive companion to figure2() — same per-joint source data, redrawn as a
# 2x2 panel of grouped bar charts (Wrist / MCP / PIP / DIP / Tip), one model
# per panel. Tests whether the proximal->distal error trend is preserved
# across compression families and across architectures. The line-plot
# figure2() output (figure4_perjointMPJPE) is intentionally untouched — this
# is a SEPARATE PDF (figure4_perjointMPJPE_2x2) saved alongside it.


def figure3(base, prune, qmm, qmse, mp, out_dir):
    # Joint-type groups. The thumb is anatomically CMC/MCP/IP (3 segment
    # joints + tip), not MCP/PIP/DIP + tip; its 4 joints are mapped by
    # POSITION onto the 4-finger scheme used for the other digits:
    #     joint 1 (thumb CMC) -> MCP group   (proximal-most)
    #     joint 2 (thumb MCP) -> PIP group   (middle)
    #     joint 3 (thumb IP)  -> DIP group   (distal segment joint, pre-tip)
    #     joint 4 (thumb TIP) -> Tip group
    # Use this mapping verbatim in the caption.
    GROUPS = [
        ('Wrist', [0]),
        ('MCP',   [1, 5, 9,  13, 17]),
        ('PIP',   [2, 6, 10, 14, 18]),
        ('DIP',   [3, 7, 11, 15, 19]),
        ('Tip',   [4, 8, 12, 16, 20]),
    ]

    # Per-panel colours match the unified Okabe-Ito palette used in
    # fig12_pareto / fig_quant_comparison (plot_pruning_results.py COLORS):
    # same model -> same colour everywhere in the thesis.
    PANEL_BASELINE  = '#555555'   # DetNet FP32 — neutral dark grey
    PANEL_L1        = '#0072B2'   # L1         — Okabe-Ito blue
    PANEL_QMSE      = '#009E73'   # QMSE (M3)  — Okabe-Ito green
    PANEL_MEDIAPIPE = '#CC79A7'   # MediaPipe  — Okabe-Ito muted pink

    # MediaPipe caveat for the caption: its per-joint values are computed
    # over DETECTED frames only (30,094 of 36,520 — see
    # eval_results_cpu/mediapipe_hands_both/summary.json:splits[*].n_detected
    # vs n_attempted). The other three panels cover all 36,520 frames.
    # Panel (d) is therefore conditional on detection; surface this in the
    # caption when discussing it.

    PANELS = [
        ('(a) DetNet FP32',     base,                 PANEL_BASELINE),
        ('(b) L1 55%',          prune['l1'].get(55),  PANEL_L1),
        ('(c) PTQ Hist-MSE',    qmse,                 PANEL_QMSE),
        ('(d) MediaPipe Hands', mp,                   PANEL_MEDIAPIPE),
    ]

    # Publication rcParams scoped to this figure only (won't leak to
    # figure1/figure2 if main() is reordered). Same dict as the
    # plot_pruning_results.py global block — serif font, inward ticks
    # on all four sides, minor ticks visible. Adopting these is what
    # gives the figure the same visual style as fig_quant_comparison.
    PUB_RCPARAMS = {
        'font.family':         'serif',
        'font.serif':          ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset':    'cm',
        'font.size':           12,
        'axes.titlesize':      13,
        'axes.labelsize':      12,
        'xtick.labelsize':     10,
        'ytick.labelsize':     10,
        'legend.fontsize':     10,
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
        'axes.linewidth':      0.8,
        'axes.grid':           False,
    }

    with plt.rc_context(PUB_RCPARAMS):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor='white')

        # LOCKED y-limits — identical across all four panels. The whole
        # claim is "same SHAPE across panels", verifiable only on a common
        # scale. Same broken-axis lower bound (0.075) as figure2() so the
        # two figures read on the same vertical scale.
        y_lo, y_hi = 0.075, 0.190

        for ax, (label, run, color) in zip(axes.flatten(), PANELS):
            # Match fig_quant_comparison axis style: all four spines
            # visible (so rcParams' xtick.top + ytick.right actually
            # show ticks on top/right), light horizontal gridlines only.
            # Deliberately NOT calling _style_axes() — that helper hides
            # top/right spines for the line-plot figures in this script.
            ax.tick_params(colors='#333333')
            ax.grid(True, axis='y', linewidth=0.3, alpha=0.3, zorder=0)
            ax.xaxis.grid(False)
            # X-axis is categorical (Wrist/MCP/PIP/DIP/Tip); suppress
            # the rcParams-default minor x-ticks between the bars.
            ax.xaxis.set_minor_locator(plt.NullLocator())

            per_j = _per_joint_norm_mpjpe(run)
            if per_j is None:
                raise RuntimeError(
                    f'figure3 (2x2): per-joint data missing for "{label}". '
                    f'Check eval_results_cpu/<run>/errors.npz.')

            means, stds = [], []
            for _, idxs in GROUPS:
                vals = per_j[np.asarray(idxs)]
                means.append(float(vals.mean()))
                # Wrist is a single joint; std across 1 element is 0 -> no
                # error bar drawn for that bar.
                stds.append(float(vals.std(ddof=0)) if len(vals) > 1 else 0.0)

            x = np.arange(len(GROUPS))
            ax.bar(x, means, yerr=stds, color=color, alpha=0.88,
                   edgecolor=color, linewidth=0.8, zorder=3,
                   error_kw=dict(ecolor='0.2', capsize=3, lw=1.0, zorder=4))

            # Bar value annotations above the error-bar cap. 3dp because
            # the per-joint MPJPE range is narrow (~0.08-0.16); 2dp would
            # collapse adjacent bars to the same number.
            for xi, m_val, s_val in zip(x, means, stds):
                ax.annotate(f'{m_val:.3f}', (xi, m_val + s_val),
                            textcoords='offset points', xytext=(0, 5),
                            ha='center', fontsize=10)

            ax.set_xticks(x)
            ax.set_xticklabels([g[0] for g in GROUPS])
            ax.set_xlim(-0.6, len(GROUPS) - 0.4)
            ax.set_ylim(y_lo, y_hi)
            # Y-label on the left column only; locked y-axis means
            # right-column bars sit on exactly the same scale.
            if ax in (axes[0, 0], axes[1, 0]):
                ax.set_ylabel('Normalised MPJPE')

            # Panel label + model name placed above the axes — same
            # bold panel-identifier convention as fig_quant_comparison.
            ax.set_title(label, loc='left', fontsize=14, fontweight='bold',
                         pad=8, color='#222222')

        plt.tight_layout(pad=1.2)
        # Same notch helper as figure2() — applies the '>' break mark to
        # each panel's left spine after the ylim is finalised.
        _add_axis_break_marks(fig)

        out = os.path.join(out_dir, 'figure4_perjointMPJPE_2x2')
        for ext in ('pdf', 'png'):
            fig.savefig(f'{out}.{ext}', dpi=1000, bbox_inches='tight',
                        facecolor='white')
            print(f'  -> {out}.{ext}')
        plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join(os.getcwd(), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    base, prune, qmm, qmse, mp = load_all()

    figure1(base, prune, qmm, qmse, mp, out_dir)
    figure2(base, prune, qmm, qmse, mp, out_dir)
    figure3(base, prune, qmm, qmse, mp, out_dir)


if __name__ == '__main__':
    main()
