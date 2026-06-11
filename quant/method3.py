"""
PHASE 5 - Method 3: full INT8 with Histogram-MSE activation calibration.

Activations: HistogramObserver (2048-bin distribution; MSE-based clipping range,
Nagel 2021 Sec 3.5 - the recommended approach). Weights: per-channel symmetric,
restricted range [-127,127] (same as M2). Should beat M2's accuracy at the same
latency (MSE handles outliers better than MinMax).

Usage:
  python quant\\method3.py            # full FreiHAND eval+train (~40 min)
  python quant\\method3.py 300        # quick smoke: 300 imgs/split
"""
import argparse

import qquant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_samples", nargs="?", type=int, default=999_999,
                    help="FreiHAND images per split (default: all)")
    args = ap.parse_args()
    qquant.run_method("method3", "Method 3 (full INT8, Histogram-MSE)",
                      qquant.M3_QCONFIG, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
