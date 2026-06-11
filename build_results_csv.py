"""
build_results_csv.py — Collate every saved evaluation run into ONE clean
results.csv (and a printed table). Rows = models, columns = metrics.

It does NOT re-run anything. It reads the per-run artefacts that
evaluate_detnet.py and evaluate_mediapipe.py already wrote in an identical
schema:

    eval_results/<run>_both/
        summary.json   model size, GFLOPs, latency, per-split headline stats
        errors.npz     per-split (N*21,) raw pixel-error arrays

Only `*_both` runs are collated (the full FreiHAND eval + composited-train
sweep). AUC is recomputed by pooling the raw errors.npz arrays with the *exact*
same formula evaluate_detnet.py uses, so the numbers match its printed
"FreiHAND (eval + train)" block bit-for-bit. MPJPE is the frame-count-weighted
mean of the split MPJPEs (exact, since MPJPE is a per-frame mean).

Columns (one row per model):
    model, method, ratio_pct,
    FreiHAND : AUC, MPJPE      (pooled eval + composited-train)
    size_MB, GFLOPs, inference_ms (mean ± std), FPS

RHD evaluation is no longer part of the pipeline (rhd_eval was the fine-tune
validation set — leakage). PCK@τ columns were dropped: AUC integrates PCK
over τ ∈ [tau_min, tau_max] anyway, so the single-point PCKs were redundant.

Usage:
    python build_results_csv.py
    python build_results_csv.py --results_dir eval_results --output results.csv
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')   # tee/redirect-safe on Windows (±, box chars)
except Exception:
    pass

import argparse
import csv
import json
import os
import re

import numpy as np

# Which saved splits make up each reported dataset group. RHD removed (was the
# fine-tune val set — leakage); 'Both' would now equal 'FreiHAND', so dropped.
GROUPS = {
    'FreiHAND': ['freihand_eval', 'freihand_train'],
}

CSV_COLUMNS = (
    ['model', 'method', 'ratio_pct']
    + [f'{g}_{m}' for g in GROUPS for m in ('AUC', 'MPJPE')]
    + ['size_MB', 'GFLOPs', 'inference_ms', 'FPS']
)

# Two-line headers for the rendered table (group on line 1, metric on line 2).
DISPLAY_LABELS = {
    'model':        'Model',
    'method':       'Method',
    'ratio_pct':    'Filters\nremoved (%)',
    'size_MB':      'Size\n(MB)',
    'GFLOPs':       'GFLOPs',
    'inference_ms': 'Latency\n(ms, mean±std)',
    'FPS':          'FPS',
}
for _g in GROUPS:
    DISPLAY_LABELS[f'{_g}_AUC']   = f'{_g}\nAUC (%)'
    DISPLAY_LABELS[f'{_g}_MPJPE'] = f'{_g}\nMPJPE'


# ── Identify a run from its directory name ────────────────────────────────────
def parse_run_name(name):
    """eval_results dir name -> (model_label, method, ratio_pct, sort_key).
    ratio_pct is '' for models without a pruning ratio."""
    base = name[:-5] if name.endswith('_both') else name
    if 'mediapipe' in base:
        return 'MediaPipe Hands', 'MediaPipe', '', (3, 0)
    m = re.match(r'ft_(l1|taylor)_(\d+)pct', base)
    if m:
        meth = 'L1' if m.group(1) == 'l1' else 'Taylor'
        r = int(m.group(2))
        return f'{meth} {r}%', meth, r, (1 if meth == 'L1' else 2, r)
    if base == 'qmm':
        return 'PTQ QMM', 'QMM', '', (4, 0)
    if base == 'qmse':
        return 'PTQ QMSE', 'QMSE', '', (4, 1)
    if 'detnet' in base:
        return 'DetNet (baseline)', 'Baseline', 0, (0, 0)
    return base, 'Unknown', '', (9, 0)


# ── AUC from pooled raw errors — same maths as evaluate_detnet.py ─────────────
def auc_from_errors(errs, tau_min, tau_max):
    """errs: 1-D array of per-joint pixel errors pooled over the group.
    Returns AUC-PCK (%) integrated over τ ∈ [tau_min, tau_max]."""
    taus = np.arange(tau_min, tau_max + 1, dtype=float)
    n = errs.size
    pck = np.array([min((errs < t).sum() / n * 100.0, 100.0) for t in taus])
    return float(np.trapz(pck, taus)) / (tau_max - tau_min)


def collate_run(run_dir):
    """Read one <run>_both folder -> a CSV row dict (or None if unreadable)."""
    name = os.path.basename(run_dir.rstrip('/\\'))
    sj = os.path.join(run_dir, 'summary.json')
    ej = os.path.join(run_dir, 'errors.npz')
    if not (os.path.isfile(sj) and os.path.isfile(ej)):
        return None

    with open(sj, encoding='utf-8') as f:
        summ = json.load(f)
    with np.load(ej) as z:
        errs = {k: z[k] for k in z.files}

    splits   = summ.get('splits', {})
    # Default τ range matches plot_pruning_results.TAU_MIN/TAU_MAX (0-30). The
    # quant bridge (07_quant_to_evalresults.py) doesn't write tau_min/tau_max
    # to summary.json, so without this default QMM/QMSE rows would integrate
    # at a different τ range than the rest.
    tau_min  = int(summ.get('tau_min', 0))
    tau_max  = int(summ.get('tau_max', 30))
    label, method, ratio, sort_key = parse_run_name(name)

    row = {'model': label, 'method': method, 'ratio_pct': ratio}

    for group, members in GROUPS.items():
        present = [s for s in members if s in errs and errs[s].size]
        if not present:
            for m in ('AUC', 'MPJPE'):
                row[f'{group}_{m}'] = ''
            continue
        pooled = np.concatenate([errs[s] for s in present])
        auc = auc_from_errors(pooled, tau_min, tau_max)
        # frame-count-weighted MPJPE (exact: MPJPE is a per-frame mean)
        num = sum(splits[s]['mpjpe'] * splits[s]['n_frames']
                  for s in present if s in splits)
        den = sum(splits[s]['n_frames'] for s in present if s in splits)
        row[f'{group}_AUC']   = round(auc, 2)
        row[f'{group}_MPJPE'] = round(num / den, 4) if den else ''

    model_mb = summ.get('model', {}).get('total_mb')
    gflops   = summ.get('compute', {}).get('gflops_forward')
    lat      = summ.get('latency', {})
    lm, ls   = lat.get('total_ms_per_frame'), lat.get('total_ms_std')
    fps      = lat.get('fps')

    row['size_MB']      = round(model_mb, 2) if model_mb is not None else ''
    row['GFLOPs']       = round(gflops, 3) if gflops is not None else ''
    row['inference_ms'] = (f'{lm:.3f} ± {ls:.3f}'
                           if lm is not None and ls is not None else '')
    row['FPS']          = round(fps, 2) if fps else ''
    row['_sort']        = sort_key
    return row


# ── Pretty console table ──────────────────────────────────────────────────────
def print_table(rows):
    cols = CSV_COLUMNS
    widths = {c: max(len(c), *(len(str(r.get(c, ''))) for r in rows)) for c in cols}
    line = '─' * (sum(widths.values()) + 3 * (len(cols) - 1))
    print('\n' + line)
    print('  '.join(f'{c:<{widths[c]}}' for c in cols))
    print(line)
    for r in rows:
        print('  '.join(f'{str(r.get(c, "")):<{widths[c]}}' for c in cols))
    print(line + '\n')


# ── Rendered table image (PNG + PDF), styled like the thesis figures ─────────
def export_table_image(rows, out_base):
    """Render the results table as results.png (300 dpi) + results.pdf (vector),
    serif/Times-New-Roman styled to match plot_pruning_results.py: shaded
    header, zebra rows, baseline & MediaPipe reference rows lightly tinted."""
    try:
        import matplotlib
        matplotlib.use('Agg')                       # headless / no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print('(matplotlib not installed — skipping table image; '
              'run: pip install matplotlib)')
        return

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif':  ['Times New Roman', 'DejaVu Serif'],
    })
    HEADER_BG  = '#0072B2'   # Okabe-Ito blue (matches thesis figures)
    BASELINE   = '#FDEBD0'   # light amber  — DetNet baseline reference row
    MEDIAPIPE  = '#E8F5F0'   # light green  — MediaPipe reference row
    ZEBRA      = '#F4F4F4'   # alternating data row
    EDGE       = '#CCCCCC'

    cols    = CSV_COLUMNS
    headers = [DISPLAY_LABELS.get(c, c) for c in cols]
    text    = [[str(r.get(c, '')) for c in cols] for r in rows]

    # Column weights: widest line (header lines or any cell) drives the width.
    def _w(c, h):
        cells = [len(str(r.get(c, ''))) for r in rows]
        hlines = [len(s) for s in h.split('\n')]
        return max(max(cells, default=1), max(hlines), 4)
    weights = [_w(c, h) for c, h in zip(cols, headers)]
    total   = sum(weights)
    col_w   = [w / total for w in weights]

    fig_w = min(26.0, max(13.0, 0.17 * total))
    fig_h = 1.4 + 0.42 * len(rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    tbl = ax.table(cellText=text, colLabels=headers, colWidths=col_w,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.55)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(EDGE)
        cell.set_linewidth(0.5)
        if r == 0:                                   # header row
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_height(cell.get_height() * 1.5)
        else:
            method = rows[r - 1].get('method', '')
            if method == 'Baseline':
                bg = BASELINE
            elif method == 'MediaPipe':
                bg = MEDIAPIPE
            else:
                bg = ZEBRA if (r % 2 == 0) else 'white'
            cell.set_facecolor(bg)
            # Left-align the two text identity columns for readability.
            if cols[c] in ('model', 'method'):
                cell.set_text_props(ha='left')
                cell.PAD = 0.04

    png, pdf = out_base + '.png', out_base + '.pdf'
    fig.savefig(png, dpi=800, bbox_inches='tight', pad_inches=0.06)
    fig.savefig(pdf,            bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print(f'Wrote table image -> {png}')
    print(f'Wrote table image -> {pdf}')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Collate eval_results/ into one results.csv')
    ap.add_argument('--results_dir', default='eval_results_cpu',
                    help='folder of <run>_both/ run directories (default: eval_results_cpu — the canonical CPU run dir that run_all_eval.py writes to)')
    ap.add_argument('--output', default='results.csv')
    ap.add_argument('--include_partial', action='store_true',
                    help='also include single-dataset runs (e.g. *_rhd) — '
                         'normally skipped as they are partial smoke runs')
    args = ap.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f'No such folder: {args.results_dir}')
        return

    rows = []
    for d in sorted(os.listdir(args.results_dir)):
        run_dir = os.path.join(args.results_dir, d)
        if not os.path.isdir(run_dir):
            continue
        # Never collate a partial MediaPipe run — only the full sweep
        # (mediapipe_hands_both) is valid. This holds even with
        # --include_partial, so smoke/single-dataset MediaPipe numbers can
        # never leak into results.csv.
        if 'mediapipe' in d and not d.endswith('_both'):
            print(f'(skipped {d}: partial MediaPipe run — not collated)')
            continue
        if not args.include_partial and not d.endswith('_both'):
            continue
        row = collate_run(run_dir)
        if row is None:
            print(f'(skipped {d}: missing summary.json / errors.npz)')
            continue
        rows.append(row)
        print(f'collated {d}')

    if not rows:
        print('\nNo runs collated — nothing written. '
              'Run evaluate_detnet.py / evaluate_mediapipe.py with --dataset both first.')
        return

    rows.sort(key=lambda r: r.pop('_sort'))

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    out_base = os.path.splitext(args.output)[0]
    export_table_image(rows, out_base)

    print_table(rows)
    print(f'Wrote {len(rows)} rows -> {args.output}')


if __name__ == '__main__':
    main()
