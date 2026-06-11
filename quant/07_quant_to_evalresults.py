"""
Bridge the quantized models (QMM, QMSE) into eval_results_cpu/ using the same
summary.json (+errors.npz) schema evaluate_detnet.py writes, so
plot_pruning_results.py plots them alongside L1/Taylor/baseline/MediaPipe.

Why a bridge: a converted INT8 model can't be reloaded as a whole-model pickle
(quantized modules don't round-trip through torch.load), and evaluate_detnet
can't load it either. Quantization is deterministic, so we RE-QUANTIZE in-process
(qquant.quantize -> identical model), evaluate on the FULL FreiHAND set (eval +
the composited-background train set), and write the summary ourselves with
batch-1 CPU latency.

Usually launched via quant/run_all_eval.py. Can also be run standalone (after
08_eval_all_cpu.py, before plot_pruning_results.py):
  python quant\\07_quant_to_evalresults.py
"""
import argparse
import json
import os

import numpy as np
import torch

import qcommon as q
import qquant
from evaluate_detnet import compute_summary_stats

DEFAULT_OUT = os.path.join(q.REPO_ROOT, "eval_results_cpu")
SPLITS = ("eval", "train")
# (checkpoint tag -> parses to method in plot_pruning_results, qconfig). Re-quantized
# in-process (deterministic) - a torch.save(model) INT8 pickle doesn't round-trip.
MODELS = [("qmm", qquant.M2_QCONFIG), ("qmse", qquant.M3_QCONFIG)]
# Quantization preserves the multiply-accumulate count — only the per-op precision
# changes (FP32 → INT8). So the op count for QMM/QMSE equals the FP32 baseline; we
# read it from the baseline summary.json here so a single source of truth drives both.
# Pinned to eval_results_cpu/ because that's the canonical FP32 baseline location,
# regardless of where this run writes its own outputs (--out_dir).
BASELINE_GFLOPS_SUMMARY = os.path.join(q.REPO_ROOT, "eval_results_cpu",
                                       "ckp_detnet_71_both", "summary.json")


def bridge(tag, qconfig, max_samples, tau_min, tau_max, out_dir):
    model, _ = qquant.quantize(qconfig)        # deterministic re-quantization (== saved model)
    model.eval()

    # Single pass over FreiHAND eval + train recording both accuracy AND
    # per-frame whole-forward latency. This is the protocol-matched INT8
    # counterpart to evaluate_detnet's FP32 path (which uses LatencyAccumulator
    # inline during run_freihand). The previous synthetic measure_latency call
    # has been removed: the published CSV's inference_ms now comes from real
    # per-frame timing on the same FreiHAND frames the FP32 path uses.
    splits, errors_arrays = {}, {}
    all_times_ms = []                                 # concatenated across splits
    for split in SPLITS:
        results, times_ms = q.run_freihand_with_latency(
            model, q.DATA_ROOT, torch.device("cpu"),
            max_samples, split=split,
        )
        all_times_ms.extend(times_ms)
        stats = compute_summary_stats(results, tau_min=tau_min, tau_max=tau_max)
        key = f"freihand_{split}"
        splits[key] = {
            "n_frames": stats["n_frames"], "mpjpe": stats["mpjpe"],
            "pck_taus": stats["taus"].tolist(), "pck_curve": stats["pck_curve"],
            "auc_pck": stats["auc"],
        }
        errors_arrays[key]           = stats["all_errors"].astype(np.float32)
        errors_arrays[f"norm_{key}"] = stats["all_norms"].astype(np.float32)

    # 10-frame warm-up applied to the concatenated stream — identical to
    # evaluate_detnet.LatencyAccumulator._WARMUP, so the trimmed sample size
    # here matches the FP32 protocol exactly.
    _LATENCY_WARMUP = 10
    times_arr   = np.asarray(all_times_ms[_LATENCY_WARMUP:], dtype=np.float64)
    lat         = float(times_arr.mean()) if times_arr.size else float('nan')
    lat_std     = float(times_arr.std())  if times_arr.size > 1 else 0.0
    n_lat_eff   = int(times_arr.size)
    size_mb     = q.state_dict_size_mb(model)

    # Inherit op count from the FP32 baseline (invariant under quantization).
    baseline_gflops = None
    if os.path.isfile(BASELINE_GFLOPS_SUMMARY):
        with open(BASELINE_GFLOPS_SUMMARY, "r", encoding="utf-8") as f:
            baseline_gflops = json.load(f).get("compute", {}).get("gflops_forward")

    run_dir = os.path.join(out_dir, f"{tag}_both")
    os.makedirs(run_dir, exist_ok=True)
    summary = {
        "checkpoint": tag,                 # parse_method_ratio -> 'qmm' / 'qmse'
        "arch": None, "dataset": "both", "split": None, "device": "cpu",
        "model": {
            "param_count": 11503445, "param_count_M": 11.503445,
            "ram_gb": size_mb / 1024.0, "total_mb": size_mb,
        },
        "compute": {"gflops_forward": baseline_gflops,
                    "gflops_note": "same as FP32 baseline "
                                   "(op count invariant under quantization)"},
        "latency": {
            "frames_after_warmup": n_lat_eff,
            "warmup_frames":       _LATENCY_WARMUP,
            "total_ms_per_frame":  lat,
            "total_ms_std":        lat_std,
            "stages_ms":           {},   # whole-forward only; INT8 modules
                                         # have FloatFunctional/QuantStub ops
                                         # between stages that the FP32 per-
                                         # stage path cannot traverse
            "fps": (1000.0 / lat) if lat == lat else None,    # NaN-safe
            "protocol_note":       "Per-real-frame timing across FreiHAND "
                                   "eval+train at batch 1, 10-frame warm-up. "
                                   "Identical protocol to FP32 path "
                                   "(evaluate_detnet.LatencyAccumulator).",
        },
        "memory": {"peak_inference_gb": None, "peak_source": "CPU"},
        "splits": splits,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(run_dir, "errors.npz"), **errors_arrays)
    aucs = "  ".join(f"{k.split('_')[1]} AUC {v['auc_pck']:.2f}%" for k, v in splits.items())
    print(f"  {tag:5s}  {aucs}  lat {lat:.2f} ms  size {size_mb:.2f} MB  -> {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=999_999,
                    help="Cap per split (default: full). Small value = smoke test.")
    ap.add_argument("--tau_min", type=int, default=0,
                    help="PCK lower threshold in pixels (default 0).")
    ap.add_argument("--tau_max", type=int, default=30,
                    help="PCK upper threshold in pixels (default 30).")
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT,
                    help="Folder where qmm_both/ and qmse_both/ are written "
                         "(default: eval_results_cpu/). Use a separate dir "
                         "(e.g. eval_results_MPJPE) when re-running just for "
                         "bbox-normalised MPJPE without overwriting canonical "
                         "CPU eval results.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for tag, qconfig in MODELS:
        print(f"  quantizing + evaluating {tag} (eval + train) ...")
        bridge(tag, qconfig, args.max_samples, args.tau_min, args.tau_max,
               args.out_dir)
    print("Quant models bridged into", args.out_dir)


if __name__ == "__main__":
    main()
