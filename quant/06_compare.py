"""
PHASE 6 - compare FP32 baseline vs INT8 Method 2 (MinMax) vs Method 3 (Histogram-MSE).

Reads eval_results_cpu/{ckp_detnet_71_both, qmm_both, qmse_both}/ (populated by
quant/run_all_eval.py) and produces:
  - a printed comparison table + quant/quantization_comparison.csv
  - figures/fig_quant_comparison.{png,pdf}  (AUC / latency across the 3 models)

Reuses publication style + tau range (TAU_MIN/TAU_MAX) from plot_pruning_results.py
so the figure matches the rest of the thesis. AUC is pooled across FreiHAND eval +
composited-train (recomputed from errors.npz) to match the other figures.

Usage:
  python quant\\06_compare.py
"""
import csv
import os

import qcommon as q          # sets sys.path (repo root) + KMP_DUPLICATE_LIB_OK
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_pruning_results as ppr   # publication rcParams + Okabe-Ito COLORS

RESULTS_DIR = os.path.join(q.REPO_ROOT, "eval_results_cpu")

# (table label, figure label (2-line), run dir under eval_results_cpu/, bar colour)
# QMM  = Quantized MinMax        (Method 2)
# QMSE = Quantized MSE/Histogram (Method 3)
# Per-bar colours come from the unified palette in plot_pruning_results.COLORS
# so the same model reads as the same colour in every figure (item 1):
#   baseline_grey = DetNet FP32 (reference, not a method)
#   red           = QMM (INT8) — coral, distinct from Taylor's amber
#   green         = QMSE (INT8)
MODELS = [
    ("FP32", "Baseline\n(FP32)", "ckp_detnet_71_both", ppr.COLORS["baseline_grey"]),
    ("QMM",  "QMM\n(INT8)",      "qmm_both",            ppr.COLORS["red"]),
    ("QMSE", "QMSE\n(INT8)",     "qmse_both",           ppr.COLORS["green"]),
]


def load(run_dir):
    """Load eval_results_cpu/<run_dir>/ as a ppr-style run dict (None if missing)."""
    path = os.path.join(RESULTS_DIR, run_dir)
    if not os.path.isdir(path):
        return None
    return ppr._load_single_run(path)


def main():
    fp32 = load(MODELS[0][2])
    base_lat = fp32["latency"]["total_ms_per_frame"] if fp32 else None

    rows, colors, fig_labels = [], [], []
    for label, fig_label, run_dir, color in MODELS:
        run = load(run_dir)
        if run is None:
            print(f"  (skip {label}: {RESULTS_DIR}/{run_dir}/ not found)")
            continue
        # AUC pooled across eval+train, recomputed from errors.npz at ppr.TAU_MIN-MAX
        combined_auc = ppr._freihand_combined_auc(run, ppr.TAU_MIN, ppr.TAU_MAX)
        eval_split = run["splits"]["freihand_eval"]
        lat = run["latency"]["total_ms_per_frame"]
        rows.append({
            "model":          label,
            "size_mb":        run["model"]["total_mb"],
            "latency_ms":     lat,
            "latency_std_ms": run["latency"]["total_ms_std"],
            "speedup":        (base_lat / lat) if base_lat else 1.0,
            "eval_mpjpe":     eval_split["mpjpe"],
            "combined_auc":   combined_auc,    # FreiHAND eval + composited-train pooled
        })
        colors.append(color)
        fig_labels.append(fig_label)

    if not rows:
        print(f"No result dirs found under {RESULTS_DIR}/ - "
              f"run quant/run_all_eval.py first.")
        return

    # ── table ───────────────────────────────────────────────────────────────
    hdr = ["Model", "Size MB", "Lat ms", "Std ms", "Speedup",
           "eval MPJPE", f"AUC%(τ{ppr.TAU_MIN}-{ppr.TAU_MAX})"]
    print("\n" + "  ".join(f"{h:>14s}" for h in hdr))
    print("  ".join("-" * 14 for _ in hdr))
    for r in rows:
        print("  ".join([
            f"{r['model']:>14s}", f"{r['size_mb']:>14.2f}", f"{r['latency_ms']:>14.2f}",
            f"{r['latency_std_ms']:>14.2f}",
            f"{r['speedup']:>13.2f}x",
            f"{r['eval_mpjpe']:>14.4f}", f"{r['combined_auc']:>14.2f}",
        ]))

    csv_path = os.path.join(q.QUANT_DIR, "quantization_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved table -> {csv_path}")

    # ── figure: 2 panels (AUC / latency) ─────────────────────────────────────
    # Model-size panel intentionally dropped — the size numbers are already in
    # the printed table + quantization_comparison.csv, and a 2-panel figure
    # gives the remaining two more room.
    labels = fig_labels
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.9))

    aucs = [r["combined_auc"] for r in rows]
    axes[0].bar(x, aucs, color=colors)
    axes[0].set_ylabel("Accuracy (AUC, %)")
    # Upper limit pinned at 69.5 (was auto ~max+3 -> 71): removes the dead
    # headroom above the tallest bar (~68.5) while leaving ~1 unit of breathing
    # space so the baseline bar isn't pressed against the frame. Lower limit
    # ~3% below the smallest bar so the QMM dip reads against baseline+QMSE;
    # the axis-break at the bottom (added by ppr._add_axis_break_marks below)
    # hides the gap between zero and that lower bound.
    axes[0].set_ylim(max(0, min(aucs) - 3), 69.5)
    for xi, v in zip(x, aucs):
        axes[0].annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                         xytext=(0, 2), ha="center", fontsize=7)

    lats = [r["latency_ms"] for r in rows]
    stds = [r["latency_std_ms"] for r in rows]
    axes[1].bar(x, lats, yerr=stds, color=colors, capsize=3,
                error_kw={"ecolor": "0.2", "elinewidth": 1.0})
    axes[1].set_ylabel("Latency(ms)")
    axes[1].set_ylim(0, max(l + s for l, s in zip(lats, stds)) * 1.20)
    for xi, r in zip(x, rows):
        axes[1].annotate(f"{r['latency_ms']:.2f}", (xi, r["latency_ms"] + r["latency_std_ms"]),
                         textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.3)

    # (a) / (b) subplot identifiers, drawn in matplotlib (NOT LaTeX subfigure
    # — wrapping this PDF in a subfigure would double-label it).
    axes[0].text(-0.18, 1.02, '(a)', transform=axes[0].transAxes,
                 fontsize=12, fontweight='bold', va='bottom')
    axes[1].text(-0.18, 1.02, '(b)', transform=axes[1].transAxes,
                 fontsize=12, fontweight='bold', va='bottom')

    fig.tight_layout()
    ppr._add_axis_break_marks(fig)
    out_dir = os.path.join(q.REPO_ROOT, "figures")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"fig_quant_comparison.{ext}")
        fig.savefig(p)
        print(f"saved figure -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
