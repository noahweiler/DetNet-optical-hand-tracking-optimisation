"""
Re-evaluate baseline + L1 + Taylor on CPU, FreiHAND eval + train, into eval_results_cpu/.

So every model in the thesis figures uses the SAME measurement conditions
(CPU latency, FreiHAND accuracy). Each model is evaluated on the FULL FreiHAND
set: 1 evaluation split (~3,960) + 1 composited-background training set (32,560
images, indices [32560, 65120) — green screen avoided; see run_freihand). Writes
NEW *_both run dirs; the GPU eval_results/ is left untouched.

Usually launched via quant/run_all_eval.py (which also wipes the folder first and
runs the quant + MediaPipe phases). Can also be run standalone:
  python quant\\08_eval_all_cpu.py
"""
import argparse
import os
import subprocess
import sys

import qcommon as q   # sets repo root on path

OUT = "eval_results_cpu"
RATIOS = [10, 25, 40, 55, 70]
# --max_samples default = uncapped -> full eval (~3,960) + full composited train
# set (32,560). Pass a small value (e.g. 500) for a quick smoke test.

# (checkpoint, arch-or-None) relative to repo root
configs = [("new_check_point/ckp_detnet_71.pth", None)]   # baseline
for method in ("l1", "taylor"):
    for r in RATIOS:
        configs.append((
            f"checkpoints_finetuned/{method}_{r}pct/ft_{method}_{r}pct_rhdbest.pth",
            f"pruned_architectures/detnet_{method}_{r}pct_ep71_noft.pth",
        ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=999_999,
                    help="Cap per split (default: full). Small value = smoke test.")
    ap.add_argument("--tau_min", type=int, default=0,
                    help="PCK lower threshold in pixels (default 0).")
    ap.add_argument("--tau_max", type=int, default=30,
                    help="PCK upper threshold in pixels (default 30).")
    args = ap.parse_args()

    py = sys.executable
    for i, (ckpt, arch) in enumerate(configs, 1):
        cmd = [py, "evaluate_detnet.py",
               "--checkpoint", ckpt,
               "--device", "cpu",
               "--dataset", "both",          # FreiHAND eval + train (full)
               "--max_samples", str(args.max_samples),
               "--tau_min", str(args.tau_min),
               "--tau_max", str(args.tau_max),
               "--out_dir", OUT]
        if arch:
            cmd += ["--arch", arch]
        print(f"\n{'=' * 64}\n[{i}/{len(configs)}] {ckpt}\n{'=' * 64}")
        subprocess.run(cmd, cwd=q.REPO_ROOT, check=True)
    print(f"\nAll {len(configs)} CPU / FreiHAND (eval+train) runs done -> {OUT}/")


if __name__ == "__main__":
    main()
