# DetNet 2D — INT8 Post-Training Quantization

Post-training static INT8 quantization of the 2D DetNet hand-keypoint model on
CPU. Two methods are implemented, differing only in how activation ranges are
calibrated:

- **M2 — MinMax**: per-tensor asymmetric `quint8` activations.
- **M3 — Histogram-MSE**: per-tensor `HistogramObserver` with MSE clipping
  (Nagel et al. 2021 §3.5).

Weights are quantized identically in both methods (per-channel symmetric
`qint8`, restricted range [-127, 127]), so the difference between M2 and M3
is purely an activation-calibration difference.

For the broader compression study and how this pipeline plugs into the
overall evaluation, see the repository-root [`README.md`](../README.md).

---

## Contents

- [What is quantized](#what-is-quantized)
- [File map](#file-map)
- [How to run](#how-to-run)

---

## What is quantized

INT8 static quantization runs on a quantization-ready fork of DetNet
([`detnet_quant.py`](detnet_quant.py)). Of the 46 conv layers in DetNet, the
first conv stem (`resnet50.conv1` + its BN + ReLU) and the final prediction
conv (`hmap_0.prediction`) stay FP32; the other 44 `Conv+BN(+ReLU)` groups
are fused (BN folded) and quantized.

```
input (FP32)
  conv1 + bn1 + relu ............... FP32   (first conv, stem)
  QuantStub ........................ FP32 → INT8
  resnet50 blocks + squeeze ........ INT8
  pos-grid quant + concat .......... INT8
  hmap_0.project (conv+bn+relu) .... INT8
  DeQuantStub ...................... INT8 → FP32
  hmap_0.prediction + sigmoid ...... FP32   (last conv)
  map_to_uv (argmax) ............... FP32
```

Key parameters:

- **Weights** — per-channel symmetric `qint8`, restricted range [-127, 127].
- **Activations** — M2 = MinMax (per-tensor asymmetric `quint8`);
  M3 = `HistogramObserver` (MSE).
- **Fusion** — 44 `Conv+BN(+ReLU)` groups, BN folded into the conv.
- **Bias correction** — Nagel 2021 empirical correction
  (`b' = b - (E[Wq·x] - E[W·x])` per output channel, averaged over the
  calibration set), applied to all 44 INT8 convs.
- **Calibration** — 480 images (160 each from CMU + RHD + GAN), seed 42,
  cropped identically to the DetNet training pipeline (GT keypoints →
  centre/scale → 128×128).
- **Backend** — PyTorch `x86` (FBGEMM / AVX-512 VNNI).
- **FP32 reference** — `new_check_point/ckp_detnet_71.pth`, the epoch-71
  checkpoint all compression results compare against.

---

## File map

### The model

| File | Purpose |
|---|---|
| [`detnet_quant.py`](detnet_quant.py) | `DetNetQuant` — the quantization-ready fork. Mirror of `model/detnet/detnet.py` + `model/helper/resnet_helper.py` with a `QuantStub` after the FP32 stem, a `DeQuantStub` before the final prediction conv, `FloatFunctional` wrappers on every residual add and the feature/positional-grid cat, separate `nn.ReLU` instances per fusion group, and a `fusion_list()` enumerating the 44 fusion triplets. |

### Shared helpers (imported, not run)

| File | Purpose |
|---|---|
| [`qcommon.py`](qcommon.py) | Repo-root path, `load_fp32()`, `eval_model()`, `measure_latency()`, `state_dict_size_mb()`. Re-uses `load_model`, `run_freihand` and `compute_summary_stats` from `evaluate_detnet.py`. |
| [`qcalib.py`](qcalib.py) | Builds the 480-image calibration set and exposes `load_calib_loader()`. Reads `datasets/data/.cache/*.pkl` and crops via `utils/handutils.py`. |
| [`qquant.py`](qquant.py) | The shared quantization procedure — `prepare → calibrate → convert → bias-correct` — and the M2 / M3 qconfigs. Defines no model; operates on `DetNetQuant`. |

### Pipeline scripts

| File | Purpose |
|---|---|
| [`00_baseline.py`](00_baseline.py) | FP32 reference: params, latency, size, FreiHAND accuracy → `baseline_fp32.json`. |
| [`02_calibration.py`](02_calibration.py) | Builds `calib_tensors.pt` and `calibration_indices.json` (the 480-image calibration set with its seed and indices). |
| [`method2.py`](method2.py) | M2 (MinMax) → `method2.pth` + `method2_results.json`. |
| [`method3.py`](method3.py) | M3 (Histogram-MSE) → `method3.pth` + `method3_results.json`. |
| [`06_compare.py`](06_compare.py) | Reads `eval_results_cpu/{ckp_detnet_71_both, qmm_both, qmse_both}/` and produces `quantization_comparison.csv` + `figures/fig_quant_comparison.{png,pdf}`. |
| [`07_quant_to_evalresults.py`](07_quant_to_evalresults.py) | Re-quantizes M2 and M3 in-process via `qquant.quantize()`, evaluates each on the full FreiHAND set, and writes `eval_results_cpu/qmm_both/` and `eval_results_cpu/qmse_both/` in the same schema as `evaluate_detnet.py`. |
| [`08_eval_all_cpu.py`](08_eval_all_cpu.py) | DetNet evaluation orchestrator: runs `evaluate_detnet.py` on the FP32 baseline plus all 10 pruned + fine-tuned variants and writes one `eval_results_cpu/<run>_both/` directory per model. |
| [`run_all_eval.py`](run_all_eval.py) | Full evaluation orchestrator: wipes `eval_results_cpu/`, then runs `08_eval_all_cpu.py`, `07_quant_to_evalresults.py` and `evaluate_mediapipe.py` in sequence. One command to regenerate every accuracy/latency number behind the thesis figures. |

### Generated artefacts

| File | Contents |
|---|---|
| `baseline_fp32.json` | FP32 size, latency and FreiHAND accuracy. |
| `calibration_indices.json` | Seed + the exact 480 sampled indices. |
| `calib_tensors.pt` | The 480 preprocessed `(3, 128, 128)` calibration tensors. |
| `method2.pth`, `method3.pth` | INT8 quantized model state_dicts. |
| `method2_results.json`, `method3_results.json` | Per-method size / latency / accuracy / speedup. |
| `quantization_comparison.csv` | Side-by-side FP32 vs. M2 vs. M3 numbers. |

---

## How to run

The minimal sequence to produce the two INT8 models:

```bash
python quant/00_baseline.py        # FP32 reference → baseline_fp32.json
python quant/02_calibration.py     # build calib_tensors.pt + calibration_indices.json
python quant/method2.py            # MinMax → method2.pth + method2_results.json
python quant/method3.py            # Histogram-MSE → method3.pth + method3_results.json
python quant/06_compare.py         # FP32 / M2 / M3 → quantization_comparison.csv
```

For the full evaluation sweep across baseline + pruned variants + INT8 +
MediaPipe, use `run_all_eval.py` (see the repository-root
[`README.md`](../README.md) for details).
