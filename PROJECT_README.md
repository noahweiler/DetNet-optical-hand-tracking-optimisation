# DetNet Optical Hand-Tracking Optimisation

Compression study of the DetNet 2D hand-pose estimator for real-time CPU
deployment, plus a single-camera Box-and-Block Test (BBT) harness for
benchmarking the practical impact of each compression decision.

Implementation repository for my Final-Year Project. Forked from
[`MengHao666/Minimal-Hand-pytorch`](https://github.com/MengHao666/Minimal-Hand-pytorch)
(kept as the `upstream` git remote). Adds:

- structured pruning (L1-norm and 1st-order Taylor) at 10 / 25 / 40 / 55 / 70 %,
- post-training INT8 quantization (MinMax and Histogram-MSE) with empirical bias correction,
- a MediaPipe Hands evaluator on the same FreiHAND benchmark for a
  different-architecture comparison,
- a gamified single-camera BBT app + a hand-to-mouse cursor controller.

Original upstream `LICENSE` and `README.md` are preserved under
[`License + README/`](License%20%2B%20README/).

---

## What's in the repo

### Compression
| Component | File | Notes |
|---|---|---|
| L1 pruning | `prune_l1.py` | torch_pruning DepGraph; per-layer uniform allocation |
| Taylor pruning | `prune_taylor.py` | 1st-order Taylor importance, calibrated over RHD |
| Fine-tune | `finetune_pruned.py` | RHD-validation best-checkpoint selection (`*_rhdbest.pth`) |
| INT8 PTQ pipeline | `quant/qquant.py` | shared fuse → prepare → calibrate → convert → empirical bias correction |
| M2 (MinMax) | `quant/method2.py` | per-tensor asymmetric quint8 activations |
| M3 (Histogram-MSE) | `quant/method3.py` | Nagel 2021 §3.5 MSE-clipped activations |
| Quant-ready DetNet fork | `quant/detnet_quant.py` | bit-identical to upstream FP32, plus QuantStub/DeQuantStub + fusion list |

Pipeline notes in [`quant/README.md`](quant/README.md). Backend is PyTorch's
`x86` (FBGEMM) on Ryzen Zen 4 / AVX-512 VNNI.

### Evaluation
| Component | File | Output |
|---|---|---|
| DetNet evaluator (FP32 / pruned / PTQ) | `evaluate_detnet.py` | `eval_results_cpu/<run>/summary.json` + `errors.npz` |
| MediaPipe Hands evaluator | `evaluate_mediapipe.py` | same schema, `mediapipe_hands_both/` |
| FreiHAND + RHD GT loaders | `utils/handutils.py`, etc. | pre-cropped 128×128 + Simon-2017 bbox-norm |

Both evaluators use identical metrics (MPJPE, PCK, AUC-PCK over τ ∈ [0, 30] px,
bbox-side normalised per Simon et al. 2017) so MediaPipe and DetNet numbers
are directly comparable. RHD is *not* an eval target — it was the fine-tune
validation split, so scoring it would leak.

### Clinical application
| Component | File | Purpose |
|---|---|---|
| Box-and-Block Test game | `bbt/box_block_test.py` | 60-second single-camera BBT, pinch-to-grab, MP4 + landmark CSV recording |
| Cursor control | `bbt/cursor_control.py` | hand → OS mouse via PyAutoGUI, pinch = click-and-hold |
| Shared model dispatch | `bbt/hand_tracking.py` | wraps MediaPipe / DetNet baseline / pruned / quantized behind one interface |

Decision rationale in [`analysis/bbt_decisions.md`](analysis/bbt_decisions.md).

### Figures
| Script | Outputs |
|---|---|
| `thesis_figures_clinical.py` | `pareto_auc_size`, `figure4_perjointMPJPE`, `figure4_perjointMPJPE_2x2` |
| `plot_pruning_results.py` | `fig05_pck_curves` … `fig15_heatmap_comparison` |

Decision logs in [`analysis/`](analysis/) (`pruning_decisions.md`,
`quantization_decisions.md`, `bbt_decisions.md`).

---

## Setup

```bash
conda env create -f environment.yml
conda activate minimal-hand-conda-env
```

PyTorch 2.6, Python 3.11. CPU-only is enough for evaluation + BBT.
GPU (CUDA 12.x) is needed for fine-tuning after pruning.

MediaPipe's `hand_landmarker.task` model bundle is auto-downloaded on first use.

---

## Data (not in this repo)

FreiHAND and RHD are **research-license-only** and are not redistributed.
Download separately and either symlink or pass `--data_root <path>`.

| Dataset | Source | Used for |
|---|---|---|
| FreiHAND | https://lmb.informatik.uni-freiburg.de/projects/freihand/ | Held-out evaluation (eval + train, ~36 k frames) |
| RHD | https://lmb.informatik.uni-freiburg.de/projects/hand3d/ | Pruning fine-tune validation only |

Expected layout under `<data_root>`:
```
FreiHAND/
  evaluation/rgb/*.jpg
  evaluation_xyz.json
  evaluation_K.json
  training/rgb/*.jpg
  training_xyz.json
  training_K.json
RHD/RHD_published_v2/
  training/...
  evaluation/...
```

---

## Model weights (not in this repo)

The FP32 baseline checkpoint, all 10 pruned + fine-tuned variants, and the
2 INT8 PTQ artifacts together total ~5 GB and are not pushed. Reproduce from
scratch:

```bash
# 1. FP32 baseline checkpoint
#    Download new_check_point/ckp_detnet_71.pth from the upstream release.

# 2. Pruning + fine-tune (GPU recommended; ~5 h per ratio on a single RTX 4080)
python prune_l1.py
python prune_taylor.py
python finetune_pruned.py --method l1     --ratio 10
python finetune_pruned.py --method l1     --ratio 25
# ... 10 / 25 / 40 / 55 / 70 for both methods

# 3. INT8 quantization (CPU, ~40 min per method)
python quant/00_baseline.py
python quant/01_verify_fork.py
python quant/02_calibration.py
python quant/method2.py
python quant/method3.py
```

---

## Evaluating

```bash
# FP32 baseline on the full FreiHAND eval + train (default --dataset both)
python evaluate_detnet.py --checkpoint new_check_point/ckp_detnet_71.pth

# A pruned + fine-tuned model — needs BOTH the state_dict AND the
# _noft architecture shell.
python evaluate_detnet.py \
  --checkpoint checkpoints_finetuned/l1_55pct/ft_l1_55pct_rhdbest.pth \
  --arch       pruned_architectures/detnet_l1_55pct_ep71_noft.pth

# MediaPipe Hands on the same data + metrics
python evaluate_mediapipe.py
```

Each run writes `eval_results_cpu/<run_name>/summary.json` (full numbers) and
`errors.npz` (per-joint per-frame pixel errors + bbox norms; gitignored).

---

## Box-and-Block Test + cursor control

Requires a webcam; defaults to `--camera 0`.

```bash
# 60-second BBT game, choose any backend
python bbt/box_block_test.py --model mediapipe
python bbt/box_block_test.py --model detnet-baseline
python bbt/box_block_test.py --model detnet-pruned \
  --weights checkpoints_finetuned/l1_25pct/ft_l1_25pct_rhdbest.pth \
  --arch    pruned_architectures/detnet_l1_25pct_ep71_noft.pth
python bbt/box_block_test.py --model detnet-quantized --quant-tag qmse

# Hand -> OS mouse cursor (pinch = click-and-hold)
python bbt/cursor_control.py --model mediapipe
```

Recordings save to `BBT recordings/` at the repo root: a composite MP4 of the
playthrough and a per-frame CSV with all 21 hand landmarks. Quit with `q` or `ESC`.

---

## Figures

```bash
python thesis_figures_clinical.py
# figures/pareto_auc_size.{pdf,png}
# figures/figure4_perjointMPJPE.{pdf,png}
# figures/figure4_perjointMPJPE_2x2.{pdf,png}

python plot_pruning_results.py
# figures/fig05_*..fig15_*.{pdf,png}
```

PDFs are vector; PNGs render at 1000 DPI.

---

## Repository layout

```
.
├── analysis/                  Design-decision logs (pruning, quant, BBT)
├── bbt/                       BBT game + cursor control + tracker dispatch
├── eval_results_cpu/          summary.json per run (errors.npz gitignored)
├── figures/                   Thesis figures (PDF + PNG)
├── License + README/          Upstream LICENSE + README
├── model/                     DetNet network (upstream)
├── Old_training/              Original DetNet training scripts (upstream)
├── P_files_no_longer_used/    Archived experimental code
├── quant/                     INT8 PTQ pipeline + quant-ready DetNet fork
├── Setting_up_fine_tuning/    RHD checkpoint-scoring utilities
├── tracking/                  Shared tracker abstractions
├── utils/                     Helpers carried from upstream
├── evaluate_detnet.py         DetNet evaluator (FP32 + pruned + PTQ)
├── evaluate_mediapipe.py      MediaPipe Hands evaluator
├── finetune_pruned.py         Pruned-model fine-tuner
├── prune_l1.py                L1-norm structured pruning
├── prune_taylor.py            Taylor structured pruning
├── thesis_figures_clinical.py Pareto + per-joint MPJPE figures
├── plot_pruning_results.py    All other thesis figures (Fig 5–15)
├── webcam_detnet.py           Live webcam demo (used by bbt/)
└── environment.yml            conda env spec
```

---

## Attribution

- **DetNet** — `MengHao666/Minimal-Hand-pytorch` (upstream remote). FP32
  network architecture, training data pipeline, and pretrained baseline
  weights come from this project.
- **Structured pruning** — `torch_pruning` (Fang et al. 2023, DepGraph).
- **INT8 quantization** — PyTorch `torch.ao.quantization` + FBGEMM x86
  backend. Histogram-MSE observer + empirical bias correction follow
  Nagel et al. 2021.
- **MediaPipe Hands** — Google MediaPipe Tasks API (Zhang et al. 2020).
  Closed tflite graph, used as a different-architecture comparator.
- **Datasets** — FreiHAND (Zimmermann et al. 2019), RHD (Zimmermann &
  Brox 2017).

---

## License

Upstream license preserved at [`License + README/LICENSE`](License%20%2B%20README/LICENSE).
The additional FYP code in this repo is released under the same terms.
