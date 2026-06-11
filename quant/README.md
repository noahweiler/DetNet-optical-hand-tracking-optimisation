# DetNet 2D — INT8 Post-Training Quantization

Post-training static INT8 quantization of the 2D DetNet hand-keypoint model,
benchmarked against the FP32 baseline on CPU (the deployment / measurement
target). Two methods are compared, differing only in how activation ranges are
calibrated:

- **Method 2 — MinMax** (per-tensor, asymmetric): the deliberate weak baseline.
- **Method 3 — Histogram-MSE**: the recommended approach (Nagel 2021 §3.5).

> A weight-only INT8 "Method 1" was **dropped**: PyTorch's eager `convert()`
> cannot build a weight-only conv (its quantized conv kernels are integer-in /
> integer-out, so `NoopObserver` activations crash `convert`). M2/M3 use the
> real `convert` path and give genuine INT8 inference + speedup.

---

## What is quantized

Eager-mode static quantization on a **quantization-ready fork** of DetNet
(`detnet_quant.py`). The INT8 region is the whole network **except the first and
last conv** (Decision 9 — keep first/last layers FP32):

```
input (FP32)
  conv1 + bn1 + relu .............. FP32   (first conv, stem)
  QuantStub ....................... FP32 -> INT8
  resnet50 blocks + squeeze ....... INT8
  pos-grid quant + concat ......... INT8
  hmap_0.project (conv+bn+relu) ... INT8
  DeQuantStub ..................... INT8 -> FP32
  hmap_0.prediction + sigmoid ..... FP32   (last conv)
  map_to_uv (argmax) .............. FP32
```

Only **2 of the conv layers stay FP32** (`resnet50.conv1`, `hmap_0.prediction`);
the other 44 Conv+BN(+ReLU) groups are fused (BN folded) and quantized.

### Key decisions
- **Weights**: per-channel symmetric, restricted range **[-127, 127]** (both methods).
- **Activations**: M2 = MinMax (asymmetric quint8, per-tensor); M3 = HistogramObserver (MSE).
- **Fusion**: 44 Conv+BN(+ReLU) groups; BN folding is exact (verified to 6e-7).
- **Bias correction**: empirical (Nagel 2021), applied to all 44 INT8 convs.
- **Calibration**: 480 images (160 each from CMU + RHD + GAN), seed 42, cropped
  identically to the original DetNet pipeline (GT keypoints → center/scale → 128×128).
- **Evaluation**: FreiHAND eval+train only (RHD is excluded — it was the
  fine-tuning validation set → leakage). Accuracy = MPJPE + PCK-AUC.
- **Latency**: CPU, batch size 1 (live single-frame inference target).
- **FP32 baseline**: `new_check_point/ckp_detnet_71.pth` (the ep71 checkpoint all
  pruning + quantization results compare against).

---

## File map

Dependency order (who imports whom):

```
detnet_quant.py ─┐
qcommon.py ──────┼─→ qcalib.py ─→ qquant.py ─→ method2.py / method3.py
                 └──────────────────────────→ 00_baseline.py, 01_verify_fork.py, 02_calibration.py
```

### The model
| File | Purpose | Maps to in original DetNet |
|---|---|---|
| `detnet_quant.py` | `DetNetQuant` — the only network we quantize. Mirror of the original with 5 `# QUANT:` ops (QuantStub/DeQuantStub, FloatFunctional add+cat, split ReLUs) + `fusion_list()`. | `model/detnet/detnet.py` + `model/helper/resnet_helper.py` (line-for-line, verified bit-identical) |

### Shared helper modules (imported, not run)
| File | Purpose | Maps to |
|---|---|---|
| `qcommon.py` | Paths, `load_fp32()`, `eval_model()`, `measure_latency()`, `state_dict_size_mb()`. | `evaluate_detnet.py` (reuses `load_model`, `run_freihand`, `compute_summary_stats`) |
| `qcalib.py` | Builds the 480-image calibration set + `load_calib_loader()`. | The deleted `datasets/*.py` loaders — substitutes by reading `datasets/data/.cache/*.pkl` + cropping via `utils/handutils.py` |
| `qquant.py` | The shared quantization procedure (`prepare → calibrate → convert → bias-correct`) + the M2/M3 qconfigs. Defines **no model**. | New PTQ logic; operates on `DetNetQuant` |

### Runnable scripts (pipeline steps)
| File | Step | Purpose |
|---|---|---|
| `00_baseline.py` | Phase 1 | FP32 reference: params, latency, size, FreiHAND accuracy → `baseline_fp32.json` |
| `01_verify_fork.py` | Phase 1.5 | Proves `DetNetQuant` == original FP32 (max diff 0) |
| `02_calibration.py` | Phase 2 | Builds + saves `calib_tensors.pt`, `calibration_indices.json` |
| `method2.py` | Phase 4 | Method 2 (MinMax) → `method2.pth`, `method2_results.json` |
| `method3.py` | Phase 5 | Method 3 (Histogram-MSE) → `method3.pth`, `method3_results.json` |

### Generated artifacts
| File | Contents |
|---|---|
| `baseline_fp32.json` | FP32 size / latency / FreiHAND accuracy |
| `calibration_indices.json` | seed + exact 480 sampled indices (reproducibility) |
| `calib_tensors.pt` | the 480 preprocessed `(480,3,128,128)` calibration tensors |
| `method2.pth` / `method3.pth` | converted INT8 models (whole module — reloadable) |
| `method2_results.json` / `method3_results.json` | per-method size / latency / accuracy / speedup |

---

## How to run

All commands use the env Python directly (no `conda activate` needed):
`C:\Users\noahw\anaconda3\envs\minimal-hand-torch\python.exe`. Append a number
(e.g. `300`) to any script to run a quick smoke on N images/split.

```powershell
# 1. FP32 baseline  (accuracy reused from eval_results/ckp_detnet_71_both; ~instant otherwise ~40 min)
python quant\00_baseline.py
# 2. verify the fork == FP32
python quant\01_verify_fork.py
# 3. build calibration data
python quant\02_calibration.py
# 4. Method 2 (MinMax)        — ~40 min full, ~3 min smoke
python quant\method2.py
# 5. Method 3 (Histogram-MSE) — ~40 min full
python quant\method3.py
```

Reload a quantized model:
```python
import torch  # detnet_quant must be importable (run from quant/ or add it to sys.path)
model = torch.load("quant/method2.pth", weights_only=False).eval()
```

---

## Results

FP32 baseline (full FreiHAND, CPU, batch-1):

| | Size | Latency (ms) | FreiHAND-eval MPJPE | FreiHAND-eval AUC-PCK |
|---|---|---|---|---|
| **FP32** | 44.09 MB | 63.30 ± 9.14 | 0.0451 | 95.77 % |

INT8 methods (size / latency are final; **accuracy from the full run is pending**):

| | Size | Latency (ms) | Speedup | Bias-corr. | Accuracy |
|---|---|---|---|---|---|
| **M2 (MinMax)** | 11.38 MB | 19.50 ± 2.61 | 3.25× | 44/44 | full run pending (smoke@300: eval MPJPE 0.058 / AUC 91.5) |
| **M3 (Histogram-MSE)** | pending | pending | pending | pending | pending |

Expected: M2 ≈ M3 latency (~3×), M3 accuracy > M2 (MSE handles outliers better),
both ~4× smaller than FP32.

---

## Notes
- **Environment**: conda env `minimal-hand-torch`, PyTorch 2.6.0, quantized
  backend `x86` (FBGEMM / AVX-512 VNNI on Ryzen 9 8945HS).
- **`size_mb`** is the `state_dict` parameter-storage figure (comparable to FP32);
  the `.pth` *file* is larger because it pickles the whole module for reloading.
- Nothing under `datasets/` or the upstream model files is modified — the fork is
  a new file and the `.cache/*.pkl` is read-only.
