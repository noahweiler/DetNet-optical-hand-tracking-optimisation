# DetNet Optical Hand-Tracking Optimisation

Final-year project repository: structured pruning and post-training INT8
quantization of the 2D **DetNet** hand-keypoint estimator, evaluated against
**MediaPipe Hands** on FreiHAND, with a single-camera **Box-and-Block Test
(BBT)** harness for measuring the impact of each compression decision on a
real clinical assessment.

Forked from
[`MengHao666/Minimal-Hand-pytorch`](https://github.com/MengHao666/Minimal-Hand-pytorch)
— the upstream is preserved as the `upstream` git remote. The DetNet network,
dataset pipeline and baseline weights are inherited verbatim; everything to do
with **compression, evaluation, the clinical app and the figures is original
work for this project**. See [What was inherited vs. what was added](#what-was-inherited-vs-what-was-added)
below for the precise split.

---

## Contents

- [What this repo does](#what-this-repo-does)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [The DetNet model](#the-detnet-model)
- [Pruning](#pruning)
- [Quantization](#quantization)
- [Evaluation](#evaluation)
- [Box-and-Block Test + cursor control](#box-and-block-test--cursor-control)
- [Figures](#figures)
- [What was inherited vs. what was added](#what-was-inherited-vs-what-was-added)
- [Data and weights](#data-and-weights)
- [Attribution](#attribution)
- [License](#license)

---

## What this repo does

The single research question is: **how far can DetNet be compressed before it
stops being clinically useful for a Box-and-Block-Test?** To answer that the
repo provides:

1. **Two structured-pruning pipelines** (L1-norm magnitude and 1st-order
   Taylor) at five compression ratios — 10 %, 25 %, 40 %, 55 %, 70 % — each
   fine-tuned to recover accuracy.
2. **A post-training INT8 quantization pipeline** with two activation-calibration
   methods (per-tensor MinMax and Histogram-MSE) plus empirical bias correction.
3. **A unified evaluator** that scores DetNet, every pruned + fine-tuned
   variant, every quantized variant and MediaPipe Hands on the same FreiHAND
   test set with the same MPJPE / PCK / PCK-AUC metrics (bbox-normalised per
   Simon et al. 2017).
4. **A gamified single-camera BBT app** and a hand-to-mouse cursor controller
   that can swap any of the above models in behind one interface, so
   compression decisions can be tested end-to-end on a live webcam.
5. **All thesis figure code**, reproducible end-to-end from the JSON results
   emitted by the evaluator.

---

## Quick start

```bash
conda env create -f environment.yml
conda activate minimal-hand-conda-env   # or `minimal-hand-torch` on Windows
```

PyTorch 2.6, Python 3.11. CPU is enough for evaluation, quantization and the
BBT app. A CUDA GPU is only needed for pruning fine-tunes.

```bash
# 1. Score the FP32 baseline
python evaluate_detnet.py --checkpoint new_check_point/ckp_detnet_71.pth

# 2. Score a pruned model (state_dict + matching architecture shell)
python evaluate_detnet.py \
  --checkpoint checkpoints_finetuned/l1_25pct/ft_l1_25pct_rhdbest.pth \
  --arch       pruned_architectures/detnet_l1_25pct_ep71_noft.pth

# 3. Score the MediaPipe comparator on the same data + metrics
python evaluate_mediapipe.py

# 4. Live BBT game on a webcam
python bbt/box_block_test.py --model detnet-pruned \
  --weights checkpoints_finetuned/l1_25pct/ft_l1_25pct_rhdbest.pth \
  --arch    pruned_architectures/detnet_l1_25pct_ep71_noft.pth
```

---

## Repository layout

```
.
├── BBT recordings/          Example MP4 + per-frame CSV from a BBT run
├── Old_training/            Upstream training pipeline (datasets/, losses/)  -- INHERITED
├── model/                   DetNet 2D network + ResNet-50 backbone           -- INHERITED, lightly modified
├── new_check_point/         FP32 baseline checkpoint  (ckp_detnet_71.pth)
├── pruned_architectures/    Pruned-but-not-fine-tuned model shells (*_noft.pth)
├── checkpoints_finetuned/   Fine-tuned per-ratio checkpoints (only *_rhdbest.pth tracked)
├── quant/                   INT8 PTQ pipeline (own README inside)
├── eval_results_cpu/        summary.json + errors.npz per evaluation run
├── figures/                 All thesis figures (PDF + 1000 DPI PNG)
├── bbt/                     BBT game, cursor control, shared tracker dispatch
├── utils/                   Shared helpers (image / heatmap / IK / eval)        -- INHERITED
├── prune_l1.py              L1-norm structured pruning (5 ratios)
├── prune_taylor.py          Taylor structured pruning + calibration cache
├── finetune_pruned.py       RHD-validated fine-tuner (writes *_rhdbest.pth)
├── evaluate_detnet.py       DetNet evaluator (FP32 + pruned + INT8)
├── evaluate_mediapipe.py    MediaPipe Hands evaluator (same metrics)
├── webcam_detnet.py         Live webcam demo + shared inference helpers
├── visualise_qualitative.py Figure 14: per-frame overlay grid
├── visualise_heatmaps.py    Figure 15: 21-joint heatmap grid
├── plot_pruning_results.py  Figures 5–12, 15 (all pruning figures)
├── thesis_figures_clinical.py  Per-joint MPJPE + pareto figures
├── build_results_csv.py     Collates eval_results_cpu/*/summary.json → results.csv
├── train_detnet.py          Upstream from-scratch trainer (kept for reference) -- INHERITED
├── config.py                Upstream training config                            -- INHERITED
└── environment.yml          conda env spec
```

---

## The DetNet model

The network being compressed is the 2D head of DetNet (Zhou et al. 2020,
"Monocular Real-time Hand Shape and Motion Capture using Multi-modal Data"),
implemented in [`model/detnet/detnet.py`](model/detnet/detnet.py):

- **Backbone**: ResNet-50 truncated to a `(B, 256, 32, 32)` feature map —
  [`model/helper/resnet_helper.py`](model/helper/resnet_helper.py).
- **Positional encoding**: a fixed 2-channel `(B, 2, 32, 32)` x/y grid is
  concatenated onto the features, giving `(B, 258, 32, 32)` — see
  `get_pose_tile_torch` in [`model/detnet/detnet.py`](model/detnet/detnet.py#L17).
- **Heatmap head** (`hmap_0`): a 3×3 conv + BN + ReLU `project` block, followed
  by a 1×1 `prediction` conv into **21 channels** (one per hand joint) and a
  sigmoid. The 21 32×32 heatmaps are argmax-ed to give `(u, v)` per joint.
- **3D heads disabled**: upstream's `dmap_0` and `lmap_0` are commented out in
  the model file — the FP32 baseline checkpoint still carries the unused
  weights, so the loaders are tolerant about extra keys.

This is the only network in the repo; all compression operates on a copy of
this exact module.

---

## Pruning

Both pruning scripts use the **DepGraph framework**
([`torch_pruning`](https://github.com/VainF/Torch-Pruning), Fang et al. 2023) to
resolve residual-block, BatchNorm and concat coupling automatically — any
filter we drop in a Conv2d also drops the matched BN channel(s) and the
matched input channel(s) on every downstream consumer.

### `prune_l1.py` — L1-norm magnitude pruning (Li et al. 2017)

Generates five pruned DetNet architectures from the epoch-71 baseline at
ratios r ∈ {0.10, 0.25, 0.40, 0.55, 0.70}. Importance is the per-output-channel
L1 norm (`tp.importance.MagnitudeImportance(p=1)`); allocation is per-layer
uniform (`global_pruning=False`); the 21-channel heatmap output is protected
via `ignored_layers=[model.hmap_0.prediction]`. Outputs are written as whole
`nn.Module` objects to
`pruned_architectures/detnet_l1_{ratio}pct_ep71_noft.pth`. The `_noft` suffix
marks them as not-yet-fine-tuned shells, used downstream for their architecture
shape only.

```bash
python prune_l1.py
```

### `prune_taylor.py` — 1st-order Taylor pruning (Molchanov et al. 2017/2019)

The Taylor counterpart to `prune_l1.py`: same five ratios, same per-layer
uniform allocation, same protected output, same DepGraph framework. Importance
is single-variate Taylor (`tp.importance.TaylorImportance()`, `|W ⊙ ∂L/∂W|` per
channel). Gradients used as importance are populated by a one-pass
forward-backward over the full RHD + CMU + GAN training set with
`lambda_hm = 100 × MSE` heatmap loss and no optimiser step. The calibration
gradients are persisted to `taylor_calibration_grads.pt` and re-used on
subsequent runs; deleting that file forces a fresh calibration. Outputs are
written to `pruned_architectures/detnet_taylor_{ratio}pct_ep71_noft.pth`.

```bash
python prune_taylor.py
```

### `finetune_pruned.py` — recovery fine-tune

Fine-tunes a pruned architecture shell back to a usable model. A near-superset
of the upstream `train_detnet.py`, with three additions for this project:

- A 2D-only `DetLoss2D` (heatmap-only MSE), reimplemented from the upstream
  `losses.DetLoss` heatmap term verbatim — the 2D-only DetNet does not produce
  `d_map` / `l_map`, so the upstream loss does not apply.
- `--method {l1, taylor}` and `--ratio {10, 25, 40, 55, 70}` flags that select
  the matching `pruned_architectures/*_noft.pth` shell and remap the epoch-71
  baseline weights onto it.
- Best-on-RHD checkpoint selection: each epoch the model is evaluated on the
  RHD split and the best is written to
  `checkpoints_finetuned/{method}_{ratio}pct/ft_{method}_{ratio}pct_rhdbest.pth`.
  That is the only fine-tuned checkpoint per ratio that is ever evaluated and
  the only one tracked by git.

Training data is the same RHD + CMU + GAN mix as upstream; the RHD eval split
is the validation set, so RHD is excluded from final evaluation
(see [Evaluation](#evaluation)).

```bash
python finetune_pruned.py --method l1     --ratio 25
python finetune_pruned.py --method taylor --ratio 55
```

### Pruned model lifecycle, end-to-end

```
ckp_detnet_71.pth                  baseline weights (FP32, full DetNet)
        │
        ├──> prune_l1.py / prune_taylor.py
        │        ↓
        ▼
pruned_architectures/detnet_{method}_{ratio}pct_ep71_noft.pth    smaller architecture, NOT fine-tuned
        │
        ├──> finetune_pruned.py  (RHD = validation set, GPU recommended)
        │        ↓
        ▼
checkpoints_finetuned/{method}_{ratio}pct/ft_{method}_{ratio}pct_rhdbest.pth   final, evaluated
        │
        └──> evaluate_detnet.py  (--checkpoint ft_..._rhdbest.pth  --arch ..._noft.pth)
```

---

## Quantization

INT8 post-training **static** quantization of the DetNet 2D head, evaluated on
CPU (the deployment target). All quant code lives in [`quant/`](quant/) — see
[`quant/README.md`](quant/README.md) for the longer write-up. Headline points:

### Two methods (only the activation observer differs)

| Method | Activation observer | Weights | Backend |
|---|---|---|---|
| **M2 — MinMax** (`quant/method2.py`) | Per-tensor asymmetric `quint8` (default `MinMax`) | Per-channel symmetric `qint8`, restricted range [-127, 127] | x86 (FBGEMM, AVX-512 VNNI) |
| **M3 — Histogram-MSE** (`quant/method3.py`) | Per-tensor `HistogramObserver` with MSE clipping (Nagel 2021 §3.5) | same | same |

Weights are quantized identically in both methods. The whole story of
"M2 vs. M3" is therefore an **activation-calibration** story, by design.

### Quant-ready DetNet fork — `quant/detnet_quant.py`

The upstream `model/detnet/detnet.py` is left untouched as the FP32 reference.
`detnet_quant.py` is a mirror of it with the additions PyTorch's static-quant
API requires: a `QuantStub` after the FP32 stem (`conv1 + bn1 + relu`), a
`DeQuantStub` before the final FP32 `hmap_0.prediction + sigmoid`,
`FloatFunctional` wrappers on every residual `add` and the
feature/positional-grid `cat`, separate `nn.ReLU` instances per fusion group
(replacing the shared upstream ReLU), and a `fusion_list()` enumerating the
44 `Conv+BN(+ReLU)` groups to fold. Of the 46 conv layers in DetNet, the
`resnet50.conv1` stem and the final `hmap_0.prediction` stay FP32; the other
44 are quantized.

### The shared quantization procedure — `quant/qquant.py`

Both M2 and M3 use the same procedure, parameterised only by the qconfig:
load the FP32 fork, assign the method's qconfig, mark `resnet50.conv1 + bn1 +
relu` and `hmap_0.prediction` as FP32, fuse the 44 `Conv+BN(+ReLU)` groups,
prepare, calibrate on the 480 calibration images, convert to INT8, and apply
Nagel-2021 empirical bias correction (`b' = b - (E[Wq·x] - E[W·x])`, per
output channel, averaged over the calibration set). Output is a state_dict
written to `method{2,3}.pth` plus a per-method results JSON.

Calibration data is built once by `quant/02_calibration.py` and persisted as
`calibration_indices.json` (seed + indices) and `calib_tensors.pt` (the 480
`(3, 128, 128)` preprocessed tensors), so M2 and M3 calibrate on byte-for-byte
the same images.

### Running

```bash
python quant/00_baseline.py        # FP32 reference: size / latency / accuracy
python quant/02_calibration.py     # build calib_tensors.pt + calibration_indices.json
python quant/method2.py            # MinMax  → method2.pth + method2_results.json
python quant/method3.py            # Hist-MSE → method3.pth + method3_results.json
```

---

## Evaluation

### `evaluate_detnet.py` — DetNet evaluator (FP32 + pruned + INT8)

Scores a DetNet variant on FreiHAND with bbox-normalised metrics. Given just
`--checkpoint`, it loads the upstream FP32 DetNet; with `--checkpoint` plus
`--arch`, it loads a pruned + fine-tuned variant (the architecture shape comes
from the `_noft` shell, the `*_rhdbest.pth` state_dict is applied on top). Each
FreiHAND image is GT-centre-cropped to 128×128, fed through the model, and the
32×32 per-joint heatmaps are argmax-ed to `(u, v)` and un-cropped back to the
original image. Stage-level timings (feature extractor, heatmap head, UV
extraction, UV → pixel conversion) are recorded per frame using
`time.perf_counter()`, with the first 10 frames dropped as warm-up.

Metrics use Simon et al. 2017 bbox-side normalisation:

- **MPJPE** — mean per-joint pixel error / (bbox-side / 2)
- **PCK@τ** — fraction of joints within τ px, for τ ∈ [0, 30]
- **PCK-AUC** — area under the PCK curve over that τ range

Output is written to `eval_results_cpu/<run_name>/summary.json` (all metrics
plus FLOPs, parameter count, model size, latency statistics) and `errors.npz`
(per-joint per-frame errors + bbox norms; gitignored). Defaults are full
FreiHAND (`--dataset both`) and CPU (`--device cpu`).

```bash
# FP32 baseline
python evaluate_detnet.py --checkpoint new_check_point/ckp_detnet_71.pth

# Pruned + fine-tuned (state_dict + matching architecture shell)
python evaluate_detnet.py \
  --checkpoint checkpoints_finetuned/l1_55pct/ft_l1_55pct_rhdbest.pth \
  --arch       pruned_architectures/detnet_l1_55pct_ep71_noft.pth
```

### `evaluate_mediapipe.py` — MediaPipe Hands comparator

Scores MediaPipe Hands on the same data, GT and metric maths as
`evaluate_detnet.py`, so MediaPipe and DetNet numbers are directly comparable.
MediaPipe runs on the full image (no GT-centred crop) and returns 0–N detected
hands; the closest centroid to the GT is matched. Accuracy is computed over
detected frames only, and a per-split `detection_rate` is also recorded.
Latency is the MediaPipe `process()` call only, mirroring DetNet's model-only
timing.

```bash
python evaluate_mediapipe.py
```

### Evaluation policy

- **FreiHAND only.** RHD's eval split is the validation set used for pruning
  fine-tunes, so scoring it would be train/validation leakage. FreiHAND is
  the only held-out benchmark.
- **CPU, batch size 1.** The deployment target is single-frame webcam
  inference on a laptop CPU; both flags default to that.

### `quant/run_all_eval.py` — full evaluation orchestrator

One command to regenerate every accuracy/latency number behind the thesis
figures. Wipes `eval_results_cpu/` and runs three phases on CPU:

1. DetNet — FP32 baseline + 5 L1-pruned + 5 Taylor-pruned (via
   `evaluate_detnet.py`).
2. INT8 — M2 MinMax + M3 Histogram-MSE (via `07_quant_to_evalresults.py`,
   which re-quantizes from `qquant.py` then scores).
3. MediaPipe Hands (via `evaluate_mediapipe.py`).

Outputs land in `eval_results_cpu/<run_name>/` — one directory per model. The
full sweep is long (~6 hours, ~36 k forward passes per model); a smoke run is
available via `--max_samples 500` (~10 min).

```bash
python quant/run_all_eval.py
```

### `build_results_csv.py`

Collates every `eval_results_cpu/<run>/summary.json` into one `results.csv`
for the figure scripts.

---

## Box-and-Block Test + cursor control

A clinical Box-and-Block Test counts how many blocks a patient can move
across a centre partition in 60 seconds. The repo includes a single-camera
gamified version: pinch (thumb + index) to grab a block, drag across the
partition, release to drop. Score = blocks delivered.

### `bbt/hand_tracking.py` — shared dispatch

One interface, four backends, all selected by a `--model` flag:

| `--model`            | Backend |
|---|---|
| `mediapipe`          | Google MediaPipe Hands (Tasks API, `hand_landmarker.task` bundle, auto-downloaded) |
| `detnet-baseline`    | FP32 DetNet (`new_check_point/ckp_detnet_71.pth`) |
| `detnet-pruned`      | Any pruned + fine-tuned variant (`--weights` + `--arch`) |
| `detnet-quantized`   | INT8 model (`--quant-tag qmm` or `qmse` → `quant/method2.pth` / `method3.pth`) |

The shared module also defines a `PinchState` helper (single-threshold +
debounce for engage/release) and a `pinch_distance` function used by both the
game and the cursor controller.

### `bbt/box_block_test.py` — the BBT game

60 seconds of active play with a 3-second countdown and a 5-second end screen.
Visuals overlaid in the webcam frame:

- A 4-block-tall centre partition the hand cannot pass through while holding.
- Blocks on the start side that respawn when delivered.
- Live score + timer + model label.

Recordings are written to `BBT recordings/`:

- `<timestamp>_<model-label>_score<N>.mp4` — composite mirrored frame.
- `<timestamp>_<model-label>_score<N>.csv` — per-frame timestamp, score, pinch
  state, all 21 landmarks (`index_tip_x` etc.) and per-frame inference latency.

The label encodes the compression method and ratio (e.g.
`detnet-pruned_L1-25pct`, `detnet-quantized_QMM`).

### `bbt/cursor_control.py` — hand-to-mouse

Maps the index-tip pixel to the OS mouse cursor via PyAutoGUI. Pinch =
click-and-hold. Uses inverted hysteresis (engage at 20 px, release at 30 px,
sticky 20–30 px dead-zone) for a clean grab/release without flicker. Same
backend dispatch as the BBT game.

### `webcam_detnet.py` — live demo + shared inference

The original webcam demo, kept for direct inspection, also acts as the shared
inference layer for `bbt/hand_tracking.py` — `load_model`, `load_quantized`,
`preprocess`, `get_hand_detection` and the skeleton-drawing helper all live
here.

---

## Figures

### `thesis_figures_clinical.py`

- **Pareto figure** (`figures/fig12_pareto.{pdf,png}`): AUC-PCK vs. model
  size, every method/ratio overlaid + the baseline + the two INT8 points.
- **Per-joint MPJPE** (`figures/figure4_perjointMPJPE.{pdf,png}` and the 2×2
  variant): 21 hand joints on the x-axis, MPJPE on the y, one panel per
  compression family.

### `plot_pruning_results.py`

All other pruning figures (Fig 5 PCK curves, Fig 6 accuracy-vs-ratio per
method, Fig 7 inference-time-vs-ratio, Fig 8 model-size, Fig 9 GFLOPs, Fig 10
L1-vs-Taylor head-to-head). Each script reads `eval_results_cpu/<run>/summary.json`
+ `results.csv` directly, so the figures regenerate from a fresh evaluation
sweep without any intermediate state.

### `visualise_qualitative.py` and `visualise_heatmaps.py`

- **Fig 14** (`fig14_qualitative_overlay.{pdf,png}` + `fig_overlay_mpjpe_v2`):
  a grid of FreiHAND frames with the skeleton overlay from each compressed
  model, per-frame bbox-normalised MPJPE annotated in a corner.
- **Fig 15** (`fig15_heatmap_comparison.{pdf,png}`): a 21-joint heatmap grid
  per model, for a single hand image — shows directly how aggressive pruning
  smears the heatmaps.

All figures are written as **PDF (vector)** + **PNG at 1000 DPI**, with serif
fonts and the publication rcParams set scoped via `plt.rc_context`.

---

## What was inherited vs. what was added

### Inherited from `MengHao666/Minimal-Hand-pytorch` (unchanged or near-unchanged)

| Path | What it is |
|---|---|
| [`model/detnet/`](model/detnet/) | The DetNet network. The 3D `dmap_0` / `lmap_0` heads are commented out (not deleted) to stay reproducible against the upstream baseline checkpoint. Frozen for this project. |
| [`model/helper/`](model/helper/) | The ResNet-50 backbone and 3×3 conv helper. |
| [`Old_training/datasets/`](Old_training/datasets/) | All dataset loaders: `handataset.py`, `rhd.py`, `ganerated_hands.py`, `hand143_panopticdb.py`, `stb.py`, `dexter_object.py`, `egodexter.py`, `hand_labels.py`. Used unchanged by both fine-tune and Taylor calibration. |
| [`Old_training/losses/`](Old_training/losses/) | `detloss.py` (the 3D-aware `DetLoss`) and `shape_loss.py`. The 2D-only `DetLoss2D` in `finetune_pruned.py` is reimplemented from `DetLoss.compute_loss`'s heatmap term verbatim. |
| [`utils/`](utils/) (whole directory) | All helper modules from upstream — image / crop / heatmap utilities (`handutils.py`, `heatmaputils.py`, `imgutils.py`, `func.py`, `misc.py`), the inverse-kinematics + Levenberg-Marquardt + bone-length stack (`AIK.py`, `LM.py`, `LM_new.py`, `bone.py`, `align.py`), the EMA `smoother.py`, evaluation helpers (`utils/eval/`) and `vis.py`. The IK / LM / bone modules belong to the upstream 3D pipeline; they're not actively called in the 2D-only path here but kept for parity with upstream. |
| [`train_detnet.py`](train_detnet.py), [`config.py`](config.py) | Original from-scratch trainer + its config. Not used in the project — kept for reference only. |
| [`new_check_point/ckp_detnet_71.pth`](new_check_point/) | The FP32 baseline weights from the upstream release. The single reference checkpoint all results compare against. |

### Added for this project

| Path | What it is |
|---|---|
| `prune_l1.py`, `prune_taylor.py` | Both pruning pipelines + the Taylor calibration cache. |
| `finetune_pruned.py` | Near-superset of `train_detnet.py` with `DetLoss2D`, `--method/--ratio` flags, RHD-best checkpoint selection, Windows `__main__` guard. |
| `evaluate_detnet.py` | Whole evaluator: bbox-normalised metrics, per-stage timing, CPU defaults, FreiHAND-only. |
| `evaluate_mediapipe.py` | MediaPipe Hands evaluator on the same data + metrics. |
| `quant/` (whole directory) | INT8 PTQ pipeline: `detnet_quant.py` fork, shared `qquant.py` procedure, M2 and M3 methods, calibration, FP32 verification, results JSON. |
| `webcam_detnet.py` | Live webcam demo + shared inference helpers used by `bbt/`. |
| `bbt/` (whole directory) | BBT game (`box_block_test.py`), cursor control (`cursor_control.py`), shared tracker dispatch (`hand_tracking.py`). |
| `visualise_qualitative.py`, `visualise_heatmaps.py`, `thesis_figures_clinical.py`, `plot_pruning_results.py` | Every thesis figure. |
| `build_results_csv.py` | Collates `eval_results_cpu/*/summary.json` → `results.csv`. |
| `pruned_architectures/`, `checkpoints_finetuned/`, `eval_results_cpu/`, `figures/` | Generated artefacts (see [Data and weights](#data-and-weights) for what's tracked). |

### Modified vs. upstream

- [`model/detnet/detnet.py`](model/detnet/detnet.py) — 3D `dmap_0` / `lmap_0`
  heads commented out (not deleted) so the FP32 checkpoint still loads. No
  numerical change to the 2D path.
- [`environment.yml`](environment.yml) — bumped to PyTorch 2.6 / Python 3.11
  and merged with the quantization deps.

The upstream pickled `__pycache__/` and `.idea/` trees were deleted from the
working tree but are visible in early commits.

---

## Data and weights

### Datasets — not in this repo

FreiHAND and RHD are research-licence-only and not redistributed. Download
separately and pass `--data_root <path>` to the relevant scripts.

| Dataset | Source | Used for |
|---|---|---|
| FreiHAND | https://lmb.informatik.uni-freiburg.de/projects/freihand/ | Held-out evaluation only (eval + train splits, ~36 k frames) |
| RHD | https://lmb.informatik.uni-freiburg.de/projects/hand3d/ | Pruning fine-tune validation only (excluded from evaluation) |
| CMU Panoptic / GAN-erated | upstream pipeline | Fine-tune training |

### Weights — in this repo

All model weights needed to reproduce evaluation are included:

- `new_check_point/ckp_detnet_71.pth` — the FP32 baseline checkpoint, the
  single reference all results compare against.
- `pruned_architectures/detnet_{l1,taylor}_{10,25,40,55,70}pct_ep71_noft.pth`
  — the 10 pruned architecture shells (one per method × ratio).
- `checkpoints_finetuned/{method}_{ratio}pct/ft_*_rhdbest.pth` — the
  best-on-RHD fine-tuned checkpoint per ratio (one per method × ratio).
- `quant/method2.pth`, `quant/method3.pth` — INT8 quantized model state_dicts
  (M2 MinMax and M3 Histogram-MSE).

---

## Attribution

- **DetNet** — `MengHao666/Minimal-Hand-pytorch` (upstream remote). FP32
  network architecture, training data pipeline, and pretrained baseline
  weights come from this project.
- **Structured pruning** — `torch_pruning` (Fang et al. 2023, DepGraph),
  with importance criteria from Li et al. 2017 (L1) and Molchanov et al.
  2017/2019 (Taylor).
- **INT8 quantization** — PyTorch `torch.ao.quantization` + FBGEMM x86
  backend. Histogram-MSE observer + empirical bias correction follow
  Nagel et al. 2021.
- **MediaPipe Hands** — Google MediaPipe Tasks API (Zhang et al. 2020).
  Closed tflite graph, used as a different-architecture comparator.
- **Datasets** — FreiHAND (Zimmermann et al. 2019), RHD (Zimmermann &
  Brox 2017).

---

## License

This repository is a fork of
[`MengHao666/Minimal-Hand-pytorch`](https://github.com/MengHao666/Minimal-Hand-pytorch)
and is released under the same terms as the upstream project (see the upstream
`LICENSE` file at that repository). All additional FYP code added here is
released under the same terms.
