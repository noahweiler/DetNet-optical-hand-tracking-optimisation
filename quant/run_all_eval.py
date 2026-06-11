"""
ONE command to regenerate every model in the thesis figures, all under identical
conditions: CPU latency + FreiHAND accuracy on 1 evaluation split (~3,960) + 1
composited-background training set (32,560 images, indices [32560, 65120), green
screen avoided). Results go to eval_results_cpu/ as *_both run dirs.

Phases:
  0. WIPE eval_results_cpu/  (everything below is regenerated, so stale *_freihand
     runs from earlier capped/eval-only sweeps don't double-plot every model).
  1. DetNet  : baseline + L1x5 + Taylor x5   (08_eval_all_cpu.py)
  2. Quant   : QMM + QMSE                      (07_quant_to_evalresults.py)
  3. MediaPipe: deployment reference           (evaluate_mediapipe.py, full run)

The full run is long (several hours - every model does ~36.5k forward passes).
Kick it off and leave it. Afterwards run plot_pruning_results.py; tau can be
chosen later (figures recompute PCK/AUC from the saved raw pixel errors).

Usage:
  python quant\\run_all_eval.py                  # full run (~6 h)
  python quant\\run_all_eval.py --max_samples 500  # smoke test (~10 min, caps each split)
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

import qcommon as q   # sets repo root on path + KMP_DUPLICATE_LIB_OK

OUT = os.path.join(q.REPO_ROOT, "eval_results_cpu")
PY = sys.executable


def _plotter_tau_default():
    """Read TAU_MIN / TAU_MAX from plot_pruning_results.py so the orchestrator's
    eval-time tau defaults always match the plotter's figure tau (single source
    of truth — edit one place). Falls back to 0/30 (the thesis-chosen range) if
    the file isn't readable.
    """
    try:
        with open(os.path.join(q.REPO_ROOT, "plot_pruning_results.py"),
                  encoding="utf-8") as f:
            text = f.read()
        tmin = int(re.search(r"^TAU_MIN\s*=\s*(\d+)", text, re.M).group(1))
        tmax = int(re.search(r"^TAU_MAX\s*=\s*(\d+)", text, re.M).group(1))
        return tmin, tmax
    except Exception:
        return 0, 30


def wipe_results():
    if os.path.isdir(OUT):
        existing = sorted(d for d in os.listdir(OUT)
                          if os.path.isdir(os.path.join(OUT, d)))
        print(f"[phase 0] wiping {OUT}/  ({len(existing)} existing run dir(s))")
        for d in existing:
            print(f"          - {d}")
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)


def run(cmd, label):
    print(f"\n{'#' * 70}\n# {label}\n{'#' * 70}")
    subprocess.run(cmd, cwd=q.REPO_ROOT, check=True)


def main():
    tau_def_min, tau_def_max = _plotter_tau_default()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max_samples", type=int, default=999_999,
                    help="Cap per split. Default = full (~36.5k/model). "
                         "Small value (e.g. 500) = quick smoke test.")
    ap.add_argument("--tau_min", type=int, default=tau_def_min,
                    help=f"PCK lower threshold in pixels. Default = "
                         f"plot_pruning_results.TAU_MIN ({tau_def_min}). Forwarded "
                         f"to evaluate_detnet, qquant bridge, and evaluate_mediapipe "
                         f"so printed AUC + summary.json scalars match the figures.")
    ap.add_argument("--tau_max", type=int, default=tau_def_max,
                    help=f"PCK upper threshold in pixels. Default = "
                         f"plot_pruning_results.TAU_MAX ({tau_def_max}).")
    args = ap.parse_args()
    ms = str(args.max_samples)
    tmin, tmax = str(args.tau_min), str(args.tau_max)
    scope = "full" if args.max_samples >= 999_999 else f"SMOKE TEST ({args.max_samples}/split)"
    print(f"PCK / AUC tau range: {args.tau_min}-{args.tau_max} px  "
          f"(default = plot_pruning_results.py TAU_MIN/TAU_MAX)")

    wipe_results()
    run([PY, os.path.join(q.QUANT_DIR, "08_eval_all_cpu.py"),
         "--max_samples", ms, "--tau_min", tmin, "--tau_max", tmax],
        f"phase 1/3  DetNet: baseline + L1 + Taylor  (FreiHAND eval + train, {scope})")
    run([PY, os.path.join(q.QUANT_DIR, "07_quant_to_evalresults.py"),
         "--max_samples", ms, "--tau_min", tmin, "--tau_max", tmax],
        f"phase 2/3  Quant: QMM + QMSE  (FreiHAND eval + train, {scope})")
    mp_cmd = [PY, "evaluate_mediapipe.py",
              "--dataset", "both", "--max_samples", ms,
              "--tau_min", tmin, "--tau_max", tmax,
              "--out_dir", "eval_results_cpu"]
    if args.max_samples < 999_999:        # let MediaPipe save a capped smoke run
        mp_cmd.append("--allow_partial_save")
    run(mp_cmd, f"phase 3/3  MediaPipe reference  (FreiHAND eval + train, {scope})")
    print(f"\nDONE ({scope}). All models -> {OUT}/  "
          f"Next: python plot_pruning_results.py")


if __name__ == "__main__":
    main()
