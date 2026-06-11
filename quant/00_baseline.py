"""
PHASE 1 - FP32 baseline & architecture inspection.

Establishes the reference the three INT8 methods are compared against:
  [1.1] load original 2D-only DetNet with ep71 weights, confirm ~11.5M params
  [1.2] print Conv/BN/ReLU module names (informs the fusion list later)
  [1.4] CPU latency (500 passes, 50 warmup) - same protocol as every method
  [1.5] state_dict size on disk
  [1.3] FreiHAND eval+train accuracy (the slow step - runs last)

Writes quant/baseline_fp32.json.

Usage:
  python quant\\00_baseline.py            # full FreiHAND (36,520 imgs, ~30-45 min CPU)
  python quant\\00_baseline.py 300        # quick smoke: 300 imgs/split (~1-2 min)
"""
import argparse
import json
import os

import qcommon as q
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_samples", nargs="?", type=int, default=999_999,
                    help="images per FreiHAND split (default: all)")
    args = ap.parse_args()

    print("=" * 64)
    print("  PHASE 1 - FP32 baseline")
    print("=" * 64)

    # [1.1] load -------------------------------------------------------------
    print("\n[1.1] Load original FP32 model (ckp_detnet_71.pth)")
    model = q.load_fp32("cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    assert 11_000_000 < n_params < 12_000_000, f"unexpected param count: {n_params}"
    print("  -> ~11.5M OK")

    # [1.2] architecture -----------------------------------------------------
    print("\n[1.2] Conv / BN / ReLU modules")
    convs = []
    for name, m in model.named_modules():
        cls = m.__class__.__name__
        if any(c in cls for c in ("Conv", "Batch", "ReLU")):
            print(f"  {name:55s} {cls}")
            if "Conv" in cls:
                convs.append(name)
    print(f"  First conv (FP32, exempt) : {convs[0]}")
    print(f"  Last conv  (FP32, head)   : {convs[-1]}")

    # [1.4] latency ----------------------------------------------------------
    print("\n[1.4] FP32 CPU latency (500 passes, 50 warmup)")
    lat, lat_std = q.measure_latency(model, device="cpu")
    print(f"  FP32: {lat:.2f} +/- {lat_std:.2f} ms/frame  "
          f"(threads={torch.get_num_threads()})")

    # [1.5] size -------------------------------------------------------------
    print("\n[1.5] FP32 state_dict size")
    size_mb = q.state_dict_size_mb(model)
    print(f"  FP32 size: {size_mb:.2f} MB")

    # [1.3] accuracy (slow - last) ------------------------------------------
    tag = "ALL" if args.max_samples >= 999_999 else f"{args.max_samples}/split"
    print(f"\n[1.3] FP32 accuracy on FreiHAND (max_samples={tag})")
    metrics = q.eval_model(model, "cpu", max_samples=args.max_samples)
    for split, d in metrics.items():
        print(f"  {split:15s} n={d['n_frames']:6d}  MPJPE={d['mpjpe']:.4f}  "
              f"AUC={d['auc_pck']:.2f}  PCK@20={d['pck20']:.2f}  PCK@50={d['pck50']:.2f}")

    # save -------------------------------------------------------------------
    out = {
        "checkpoint": "new_check_point/ckp_detnet_71.pth",
        "params": int(n_params),
        "size_mb": size_mb,
        "latency_ms": lat,
        "latency_std_ms": lat_std,
        "num_threads": torch.get_num_threads(),
        "max_samples": args.max_samples,
        "metrics": metrics,
    }
    out_path = os.path.join(q.QUANT_DIR, "baseline_fp32.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved baseline -> {out_path}")


if __name__ == "__main__":
    main()
