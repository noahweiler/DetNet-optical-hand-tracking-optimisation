"""
PHASE 4 - Method 2: full INT8 with MinMax activation calibration.

Activations: MinMax (per-tensor, asymmetric quint8). Weights: per-channel
symmetric, restricted range [-127,127]. The deliberate baseline (Gholami 2021):
MinMax is not production-recommended but isolates the gain from MSE in Method 3.

Usage:
  python quant\\method2.py            # full FreiHAND eval+train (~40 min)
  python quant\\method2.py 300        # quick smoke: 300 imgs/split
"""
import argparse

import qquant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_samples", nargs="?", type=int, default=999_999,
                    help="FreiHAND images per split (default: all)")
    args = ap.parse_args()
    qquant.run_method("method2", "Method 2 (full INT8, MinMax)",
                      qquant.M2_QCONFIG, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
