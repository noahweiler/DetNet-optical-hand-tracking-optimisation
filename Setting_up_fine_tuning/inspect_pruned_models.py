"""
Report file size (MB) and parameter count for the L1-pruned DetNet models
in pruned_architectures/, alongside the unpruned baseline.

Run from the repo root with the conda env active:
    python inspect_pruned_models.py
"""
import os
import cv2  # noqa: F401  -- import-for-side-effect; keep above torch (Windows DLL order)
import torch

BASELINE_PARAMS = 11_503_445   # the unpruned 2D detnet (model/detnet/detnet.py)
PRUNE_RATIOS    = [10, 25, 40, 55, 70]
PRUNED_DIR      = "pruned_architectures"
PRUNED_FMT      = "detnet_l1_{r}pct_ep71_noft.pth"


def file_mb(path):
    """File size in MB (decimal megabytes — divisor 1000 × 1000)."""
    return os.path.getsize(path) / (1000 * 1000)


def n_params(path):
    m = torch.load(path, map_location="cpu", weights_only=False)
    return sum(p.numel() for p in m.parameters())


def main():
    print(f"\n{'Model':<34} {'File MB':>9} {'Params':>14} {'Reduction':>11}")
    print("-" * 72)

    # baseline reference (param count only; the .pth in new_check_point is a state_dict)
    print(f"{'baseline (unpruned detnet)':<34} {'-':>9} {BASELINE_PARAMS:>14,} {'0.0%':>11}")

    for r in PRUNE_RATIOS:
        path = os.path.join(PRUNED_DIR, PRUNED_FMT.format(r=r))
        if not os.path.isfile(path):
            print(f"{PRUNED_FMT.format(r=r):<34} {'MISSING':>9}")
            continue
        mb = file_mb(path)
        n = n_params(path)
        red = (1 - n / BASELINE_PARAMS) * 100
        name = f"l1_{r}pct ({PRUNED_FMT.format(r=r)})"
        print(f"{name:<34} {mb:>9.2f} {n:>14,} {red:>10.1f}%")

    print("-" * 72)
    print("(Reduction = 1 - params/baseline. Realised % is ~1-(1-r)^2 because a layer "
          "pruned on both\n in- and out-channels loses ~2r of its params.)\n")


if __name__ == "__main__":
    main()
