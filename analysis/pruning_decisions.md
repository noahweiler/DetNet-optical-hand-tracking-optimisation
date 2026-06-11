# Pruning Design Decisions — Implementation Reference

This document is a complete recap of every design choice made for the DetNet
pruning experiments in this thesis, with citations to the literature each
choice is grounded in and direct references to where the choice is realised
in the codebase. It is intended as a self-contained companion to the thesis
chapter on pruning — every numbered decision below should map to a one- or
two-sentence justification in the main text plus a paragraph in the
methodology appendix.

The complete bibliography of cited works is given at the end. Inline cites
use first-author-and-year form (e.g. *Li 2017*) and link to the
corresponding entry in [§ References](#references).

## Context

DetNet's deployment target in this work is single-frame CPU inference for a
real-time hand-tracking cursor-control application. Throughput on commodity
laptop CPUs is the primary efficiency metric, so the pruning programme is
designed around **wall-clock latency**, not parameter count or sparsity. Two
structured filter-pruning criteria — **L1-norm** (Li 2017) and **Taylor
expansion** (Molchanov 2017, 2019) — are compared on the same DetNet
backbone at five pruning ratios.

All experimental decisions below are made so that the **only variables**
across the 10 pruned models (2 criteria × 5 ratios) are the choice of
importance criterion and the target ratio. Everything else — baseline
checkpoint, dependency-graph handling, layer-allocation strategy, calibration
loss, fine-tuning recipe, evaluation protocol — is fixed identically.

---

## 1. Foundational Decisions

### D1. Structured filter pruning (whole-channel removal)

**Decision.** Remove entire output filters (and their downstream
dependencies) rather than individual weights. The resulting models have
denser, smaller architectures.

**Rationale.** Unstructured weight pruning produces sparse weight tensors
that require specialised runtime support (block-sparse kernels, sparse BLAS)
to deliver wall-clock speedup; on commodity CPU runtimes (PyTorch with
FBGEMM / oneDNN) sparse weights produce *zero* speedup and often slow
inference down because of fragmentation overhead. Structured pruning, in
contrast, yields a smaller dense network whose forward pass benefits from
exactly the same kernels as the original — so any FLOP reduction translates
directly into latency reduction (Li 2017; Han 2015 establishes the broader
prune-then-fine-tune paradigm but uses unstructured pruning).

**Implementation.** `torch_pruning.pruner.MagnitudePruner` in
[`prune_l1.py:57-64`](../prune_l1.py#L57-L64) and
[`prune_taylor.py:111-118`](../prune_taylor.py#L111-L118). The library
removes filters in-place; the saved `_noft.pth` files in
[`pruned_architectures/`](../pruned_architectures) hold the resulting
smaller models as whole `nn.Module` pickles.

**Literature.** Li 2017 introduces structured L1-norm filter pruning for
ConvNets and demonstrates its wall-clock benefits on VGG / ResNet
architectures; the structured paradigm has since become standard for
deployment-oriented pruning.

---

### D2. One-shot prune-then-fine-tune (single round)

**Decision.** For each (method, ratio) pair, perform exactly one pruning
step followed by a long fine-tuning phase. No iterative pruning (alternating
pruning + recovery rounds).

**Rationale.** Iterative pruning (Han 2015, Frankle 2019) can find marginally
better sparse subnetworks but at ~5-10× the training cost. For a thesis-scale
empirical study comparing two criteria at five ratios, the one-shot recipe
keeps the total fine-tuning budget manageable while preserving fair
comparison: both criteria face the same fine-tuning regime, so any accuracy
gap reflects the importance criterion rather than a training-budget
imbalance. Liu 2019 ("Rethinking the Value of Network Pruning") provides
important context here — for many ConvNets, the resulting *architecture*
matters more than the inherited weights, and one-shot fine-tune from inherited
weights matches or exceeds training the same architecture from scratch.

**Implementation.** `prune_{l1,taylor}.py` produce the `_noft.pth` (no
fine-tune) architectures; [`finetune_pruned.py`](../finetune_pruned.py) then
runs the fine-tuning loop on each.

**Literature.** Han 2015 (the prune-then-fine-tune paradigm in general);
Li 2017 (one-shot structured filter pruning specifically); Liu 2019
(caveats — fine-tuned and from-scratch can converge similarly).

---

### D3. Identical pre-trained baseline for all pruning runs

**Decision.** Every pruned model — all 10 of them — starts from the same
upstream DetNet checkpoint, `ckp_detnet_71.pth` (epoch 71 of the
pre-training). No re-pre-training between L1 and Taylor.

**Rationale.** Isolates the pruning criterion + ratio as the only independent
variables. Any difference between the L1-55% and Taylor-55% models is
attributable to filter selection, not to a difference in starting weights.

**Implementation.**
[`prune_l1.py:30`](../prune_l1.py#L30) and
[`prune_taylor.py:56`](../prune_taylor.py#L56):
```python
CHECKPOINT = 'new_check_point/ckp_detnet_71.pth'
```
The deepcopy at the start of every per-ratio loop
([`prune_l1.py:82`](../prune_l1.py#L82),
[`prune_taylor.py:195`](../prune_taylor.py#L195)) ensures the baseline
itself is never mutated between ratios.

---

## 2. Importance Criteria

### D4. L1-norm magnitude importance (Method 1)

**Decision.** Rank filters by the L1 norm of their weights:

$$
\mathrm{importance}(f_i) \;=\; \|W_i\|_1 \;=\; \sum_{c, k_h, k_w} |W_{i, c, k_h, k_w}|.
$$

**Rationale.** Magnitude-based importance is the simplest and most widely
benchmarked criterion in the structured-pruning literature. It is *data-free*:
no forward or backward pass is required, only the weights themselves. This
gives it three properties valuable for a comparative study: (i) trivially
reproducible — no random calibration set or stochastic gradient noise; (ii)
near-zero compute cost — pruning a 50-layer ResNet takes milliseconds; (iii)
the standard baseline against which essentially every other criterion in the
field is compared.

**Implementation.** `tp.importance.MagnitudeImportance(p=1)` at
[`prune_l1.py:56`](../prune_l1.py#L56). The `p=1` argument selects the
L1 norm specifically (the class also supports `p=2` for L2-norm pruning).

**Literature.** Li 2017 introduces this exact formulation for filter pruning
and demonstrates it on VGG-16 / ResNet-56; the technique itself traces back
to Han 2015's unstructured magnitude pruning, but Li was the first to apply
it at filter granularity.

---

### D5. Taylor expansion importance (Method 2)

**Decision.** Rank filters by the absolute first-order Taylor expansion of
the loss with respect to a "zero-out" perturbation of that filter:

$$
\mathrm{importance}(f_i) \;=\; \sum_{c, k_h, k_w}
    \bigg| W_{i, c, k_h, k_w} \cdot
           \frac{\partial \mathcal{L}}{\partial W_{i, c, k_h, k_w}} \bigg|.
$$

**Rationale.** Loss-aware. Unlike L1, this criterion uses the gradient signal
to estimate how *removing* a filter would change the deployment loss — a
filter with a large weight that the loss is insensitive to (large $|W|$, small
$|\partial \mathcal{L}/\partial W|$) is considered less important than a
filter with a moderate weight whose removal would substantially perturb the
loss. The intuition is the first-order Taylor expansion:

$$
\mathcal{L}(W \setminus W_i) - \mathcal{L}(W)
    \;\approx\; -W_i^\top \frac{\partial \mathcal{L}}{\partial W_i},
$$

so the absolute value of $W_i^\top \nabla_{W_i} \mathcal{L}$ approximates the
loss increase from removing $W_i$. This is the most-cited *loss-aware*
criterion in the structured-pruning literature.

**Implementation.** `tp.importance.TaylorImportance()` at
[`prune_taylor.py:110`](../prune_taylor.py#L110). The pruner requires
`.grad` to be populated on each Parameter; the calibration pass that
populates these is discussed in [D12-D15](#5-taylor-calibration).

**Literature.** Molchanov 2017 introduces the single-variate first-order
Taylor criterion for filter pruning; Molchanov 2019 extends to a
multi-variate formulation. Both build on the much older Optimal Brain
Damage (LeCun 1990) and Optimal Brain Surgeon (Hassibi 1993) line, which
used second-order Taylor expansion for unstructured weight pruning.

---

### D6. Single-variate Taylor (not multi-variate)

**Decision.** Treat each weight's contribution to importance independently
(single-variate Taylor), summing the per-weight $|W \cdot \partial \mathcal{L}/\partial W|$
within each filter. Do *not* use Molchanov 2019's multi-variate formulation
that incorporates cross-weight interaction terms.

**Rationale.** The single-variate version is (i) what most practical
implementations use, (ii) the default in `torch_pruning`, and (iii) shown
empirically in Molchanov 2019 to match the multi-variate version within
noise on most ConvNet benchmarks while being substantially cheaper to
compute. The multi-variate version would require populating the diagonal
of the Hessian, multiplying both compute and memory cost.

**Implementation.** `tp.importance.TaylorImportance()` is instantiated
with `multivariable=False` (the library default), confirmed in
[`prune_taylor.py:110`](../prune_taylor.py#L110).

**Literature.** Molchanov 2017 (single-variate); Molchanov 2019
(multi-variate extension, with comparison to single-variate).

---

### D7. Why these two criteria and not others

**Decision.** Compare L1-norm and Taylor expansion. No other criteria
(network slimming, channel pruning via LASSO, HRank, geometric median, etc.).

**Rationale.** L1 and Taylor span the two major axes of importance-criterion
design — **data-free vs data-aware**, and **weights-only vs loss-derived** —
so a like-for-like comparison between them illuminates whether the
loss-awareness of Taylor justifies its additional cost (a full
forward-backward pass over the calibration set). Adding more criteria would
inflate the experimental matrix without changing the core methodological
question. Specific alternatives considered and rejected:

| Criterion | Reference | Why rejected |
|-----------|-----------|--------------|
| Channel pruning (LASSO) | He 2017 | Requires per-layer LASSO reconstruction step; adds a different optimisation problem to the pipeline. |
| Network slimming | Liu 2017a | Requires retraining the base network with L1 sparsity on BatchNorm γ parameters before pruning. Would change the baseline checkpoint, breaking [D3](#d3-identical-pre-trained-baseline-for-all-pruning-runs). |
| HRank | Lin 2020 | Requires computing the rank of feature maps over a calibration set; adds a per-layer SVD step. |
| Geometric median (FPGM) | He 2019 | An alternative magnitude-style criterion; expected to behave similarly to L1 and not informative as a third data point. |
| Lottery Ticket Hypothesis | Frankle 2019 | Addresses a different research question (training subnetworks from initialisation), not deployment-oriented post-training pruning. |
| AMC (RL-based) | He 2018 | RL training loop over per-layer ratios is ~50× the cost of the entire experimental matrix here. |
| Movement pruning | Sanh 2020 | Designed for fine-tuning transformer language models, not deployment pruning of pre-trained ConvNets. |

---

## 3. Allocation Strategy

### D8. Per-layer uniform allocation (`global_pruning=False`)

**Decision.** Apply the same target pruning ratio to every prunable layer
independently, rather than ranking all filters across all layers globally
against a single threshold.

**Rationale.** Per-layer uniform allocation has two important properties
for this study:

1. **Architectural equivalence between criteria.** At a fixed target ratio
   *r*, L1 and Taylor produce models with **identical channel counts in every
   layer** — only the *identity* of the kept filters differs. This is
   verified empirically by the architecture-match check in
   [`analysis/compare_filter_selections.py`](compare_filter_selections.py)
   (loaded `_noft.pth` `out_channels` match the reconstructed kept-channel
   counts for every conv at every ratio). As a result, FLOPs and parameter
   counts at a given ratio are constants of the *architecture*, not of the
   criterion, which makes side-by-side accuracy comparison clean.
2. **Avoids cross-layer score-scale issues.** L1 and Taylor scores both
   scale with layer width (more weights to sum). Global ranking would
   therefore bias removal toward the smallest layers, regardless of
   importance — a well-known artefact that requires per-layer score
   normalisation to correct (Molchanov 2019 discusses this). Per-layer
   uniform allocation sidesteps the issue entirely by never comparing
   scores across layers.

The trade-off is flexibility: global allocation can in principle allocate
"more pruning budget" to layers that tolerate it, but the cited normalisation
work has not converged on a consensus normaliser, and the per-layer recipe
remains the default in most published baselines.

**Implementation.** `global_pruning=False` at
[`prune_l1.py:62`](../prune_l1.py#L62) and
[`prune_taylor.py:116`](../prune_taylor.py#L116). Each layer's per-filter
importance is computed independently; the bottom `ceil(N × r)` filters per
group are removed.

**Literature.** Li 2017 uses per-layer ratios; Molchanov 2019 discusses
the cross-layer normalisation problem; `torch_pruning` (Fang 2023)
implements both modes.

---

### D9. Automatic dependency-graph handling

**Decision.** Use `torch_pruning`'s built-in `DependencyGraph` to resolve
all coupling between layers (residual sums, BatchNorm parameter coupling,
concatenations, downstream input-channel cascades) automatically — rather
than authoring per-architecture pruning rules by hand.

**Rationale.** DetNet contains a ResNet50 backbone with bottleneck blocks,
projection shortcuts, and a concatenation step where the position tile is
joined onto the backbone features. Each of these creates filter-dimension
constraints that span multiple layers:

- A residual `out = conv3(x) + shortcut(x)` requires `conv3.out_channels ==
  shortcut.out_channels`.
- BatchNorm's `(γ, β, μ, σ²)` vectors are coupled to the preceding conv's
  output channels — pruning a conv filter requires removing the
  corresponding BN slot.
- A downstream conv that takes another's output as input has its
  *input* channels determined by the upstream conv's output.

Manually maintaining these constraints is error-prone and would have to be
re-derived for any future architectural change. `torch_pruning`'s
`DependencyGraph` traces the dependencies once via a single forward pass
on a dummy input and produces "pruning groups" of coupled layers that
must be pruned together — so a single pruning decision (which filters
to drop from one conv's output) automatically cascades to every coupled
operation.

**Implementation.** Built implicitly by `tp.pruner.MagnitudePruner` from
the `example_inputs=torch.randn(1, 3, 128, 128)` argument
([`prune_l1.py:70`](../prune_l1.py#L70),
[`prune_taylor.py:128`](../prune_taylor.py#L128)). No explicit
`DependencyGraph` construction is required in our code.

**Literature.** Fang 2023 introduces DepGraph as a general-purpose
dependency analysis framework for structural pruning of arbitrary ConvNets;
prior work required per-architecture hand-coding (e.g. Li 2017 explicitly
listed ResNet bottleneck rules).

---

## 4. Protected Layers

### D10. The 2D heatmap-output conv must remain 21-channel

**Decision.** Exclude `model.hmap_0.prediction` (the final 1×1 conv that
produces 21 joint heatmaps) from pruning consideration.

**Rationale.** The output layer's filter count is determined by the task
specification: one heatmap per hand joint, 21 joints in the standard
hand-keypoint convention. Pruning this conv would change the number of
output heatmaps and break compatibility with the rest of the pipeline
(MPJPE metric, loss function, downstream argmax). This is a hard task
constraint, not an optimisation choice.

**Implementation.** `ignored_layers=[model.hmap_0.prediction]` passed
to `MagnitudePruner` in both
[`prune_l1.py:63`](../prune_l1.py#L63) and
[`prune_taylor.py:117`](../prune_taylor.py#L117).

---

### D11. The input stem (`resnet50.conv1`) input channels are unprunable

**Decision.** The 3-channel RGB input to the stem conv cannot have its
input dimension reduced. The output channels of the stem *can* be pruned
(and are).

**Rationale.** Pruning input channels of `conv1` would require dropping
RGB channels from the input image — meaningless, and unsupported by the
data pipeline. The output side is unconstrained and gets pruned normally.

**Implementation.** Handled implicitly by `torch_pruning`: a Conv2d whose
inputs are produced by no upstream model layer (i.e., the input boundary)
is automatically excluded from input-channel pruning by the dependency
graph. No explicit user code is needed.

---

## 5. Taylor Calibration

The Taylor criterion needs `.grad` populated on every weight before pruning.
A single calibration pass — one full epoch of forward-backward over the
training data with no optimiser step — accumulates the gradients used as the
$\partial \mathcal{L}/\partial W$ term in [D5](#d5-taylor-expansion-importance-method-2).

### D12. Calibration loss = task loss (2D-heatmap MSE only)

**Decision.** The calibration loss is identical to the loss used to fine-tune
the pruned models: the 2D-heatmap MSE term of the original DetNet objective,
$\mathcal{L}_{\text{heat}} = \lambda_{\text{hm}} \cdot \tfrac{1}{2}
\sum_{j=1}^{21} \mathrm{MSE}(\hat{H}_j, H_j^{\text{GT}})$, with
$\lambda_{\text{hm}} = 100$. The 3D-head losses (delta-map, location-map) are
omitted because the 3D heads themselves are not part of the deployment
pipeline.

**Rationale.** Taylor importance estimates each filter's contribution to a
specific loss; it must therefore use the deployment loss, not an arbitrary
proxy. Using a different loss (e.g. the original 3D-inclusive DetNet loss,
or a different reduction) would rank filters according to their
contribution to a quantity we don't care about, biasing the pruning
decisions away from the actual deployment objective.

**Implementation.**
[`prune_taylor.py:84-101`](../prune_taylor.py#L84-L101) reimplements the
heatmap term verbatim from
[`finetune_pruned.py`](../finetune_pruned.py)'s `DetLoss2D.compute_loss`,
so the calibration loss is byte-identical to the fine-tuning loss.

**Literature.** Molchanov 2017 prescribes that the Taylor importance loss
must match the deployment objective; using a different loss would
correspond to a different importance ranking entirely.

---

### D13. Calibration data distribution = fine-tuning data distribution

**Decision.** Calibration gradients are accumulated over the same dataset
mixture used to fine-tune the pruned models: `rhd + cmu + gan` (the union
of RHD, CMU-Hands and GANerated-Hands, with the same training-time
augmentation as the original DetNet training pipeline).

**Rationale.** The gradient
$\mathbb{E}_{(x, y) \sim \mathcal{D}}\big[\partial \mathcal{L}(x, y; W)/\partial W\big]$
is a property of both the model and the distribution it's evaluated against.
Using a calibration distribution that differs from the deployment / fine-tuning
distribution would produce a biased importance estimate. The
`rhd + cmu + gan` mixture is what the model is fine-tuned on (and what the
original DetNet was trained on), so it is the most faithful proxy for the
gradients the model would see during fine-tuning.

**Implementation.**
[`prune_taylor.py:61`](../prune_taylor.py#L61):
```python
DATASETS = ['rhd', 'cmu', 'gan']  # matches L1 --datasets_train rhd cmu gan
```
The `HandDataset` instantiation
([`prune_taylor.py:150-158`](../prune_taylor.py#L150-L158)) uses the same
augmentation parameters as `finetune_pruned.py`'s training-set
instantiation.

---

### D14. Single-pass calibration (no optimiser step, no multiple epochs)

**Decision.** Run exactly one epoch of forward-backward over the
calibration set, without invoking `optimizer.step()`. Gradients are
accumulated by PyTorch's autograd into `param.grad` and read out directly
for the Taylor importance computation.

**Rationale.** Taylor importance is an *estimate of the gradient*, not the
basis for parameter updates. Running multiple epochs of accumulation
without zeroing gradients would integrate them rather than approximate
$\nabla_W \mathbb{E}[\mathcal{L}]$. Running a single epoch with batch-wise
gradient zeroing would be equivalent to a single SGD step's worth of
gradients — noisy. A single epoch with un-zeroed accumulation is the standard
recipe (the gradient accumulator integrates contributions from every batch),
producing $\sim 11{,}000$-batch averaged gradients.

**Implementation.** [`prune_taylor.py:167-179`](../prune_taylor.py#L167-L179):
```python
model_orig.zero_grad()
for metas in pbar:
    ...
    loss = compute_heatmap_loss(out['h_map'], hm, hm_veil)
    loss.backward()
    # no optimiser.step()
```
After the loop, each parameter's `.grad` contains the summed gradient
across all batches; the per-filter importance is computed directly from
these.

---

### D15. Calibration gradients are cached and reused

**Decision.** After a successful calibration pass, the per-parameter
gradient tensors are pickled to `taylor_calibration_grads.pt` (~44 MB).
On subsequent runs of `prune_taylor.py` (or `prune_one.py`), the cached
file is loaded directly and the calibration pass is skipped.

**Rationale.** The calibration pass takes several minutes (forward-backward
over the full ~389k-image training set). Caching reduces that to a ~3-second
torch-load on subsequent runs, and — more importantly — guarantees that
every per-ratio pruning decision sees *bit-identical* gradients. Without
caching, gradient noise from a fresh calibration pass (DataLoader shuffling,
PyTorch deterministic-flag-dependent numerics) could nominally produce
slightly different importance rankings, complicating reproducibility.

**Implementation.** [`prune_taylor.py:137-184`](../prune_taylor.py#L137-L184).
The cache stores `{'grads': dict[str, Tensor], 'n_images': int,
'datasets': list[str]}` so a future change to the dataset list is detected
as a stale cache. The deepcopy-then-restore pattern at
[`prune_taylor.py:195-198`](../prune_taylor.py#L195-L198) works around
PyTorch's `Parameter.__deepcopy__` only cloning `.data` and not `.grad`
(see the corresponding memory note in
[`memory/feedback_pytorch_deepcopy_drops_grad.md`](../../memory/feedback_pytorch_deepcopy_drops_grad.md)).

---

## 6. Ratio Sweep

### D16. Pruning ratios — `[10, 25, 40, 55, 70] %`

**Decision.** Generate one pruned model per criterion at each of five target
ratios: 10 %, 25 %, 40 %, 55 % and 70 % per-layer channel removal.

**Rationale.** This sweep covers four operating regimes of interest:

| Ratio | Realised parameter reduction | Operating regime |
|------:|-----------------------------:|------------------|
| 10 % | ~19 % | "free" pruning — near-baseline accuracy expected |
| 25 % | ~44 % | mild pruning — measurable but small accuracy loss |
| 40 % | ~64 % | moderate pruning — clear FLOPs reduction |
| 55 % | ~80 % | aggressive pruning — start of accuracy degradation |
| 70 % | ~91 % | extreme pruning — large compression, substantial loss |

A wider sweep with smaller steps would inflate the cost of the experimental
matrix (each of 10 pruned models requires a multi-hour fine-tune). The
chosen five-point sweep is dense enough to characterise the accuracy /
efficiency Pareto curve and identify the regime where the L1-vs-Taylor
choice matters most.

**Implementation.**
[`prune_l1.py:32`](../prune_l1.py#L32) and
[`prune_taylor.py:59`](../prune_taylor.py#L59):
```python
RATIOS = [0.10, 0.25, 0.40, 0.55, 0.70]
```
A separate script [`prune_one.py`](../prune_one.py) supports a single
additional ratio without re-running the full sweep — this was used to
investigate one extra-aggressive ratio (85 %) for exploratory purposes,
though that point is excluded from the main results.

---

## 7. Fine-Tuning Recipe

### D17. 26-epoch fine-tune (≈¼ of original training duration)

**Decision.** Fine-tune each `_noft.pth` for **exactly 26 epochs** on the
same `rhd + cmu + gan` training mixture, with RHD-validated best-checkpoint
selection over those 26 epochs.

The 26 follows from invoking `--epochs 25`: the underlying training loop
runs `range(start_epoch, args.epochs + 1)` inherited from
`train_detnet.py`, so `--epochs N` actually executes N+1 epochs (see the
known off-by-one in
[`memory/feedback_epochs_off_by_one.md`](../../memory/feedback_epochs_off_by_one.md)).
The CLI default at [`finetune_pruned.py:611`](../finetune_pruned.py#L611)
is 500, but is explicitly overridden for every fine-tune launched in the
experimental matrix.

**Rationale.** This duration is set to approximately **one-quarter of the
original DetNet training time** (~104 epochs), following Li 2017's recovery
recipe (§ 4.1). Li et al. explicitly recommend retraining for "1/4 of the
original training time" with the same optimiser and hyperparameters as
the baseline. Their published experiments use:

- **40 epochs** for CIFAR-10 fine-tuning, versus ~160 epochs of original
  training (≈ ¼);
- **20 epochs** for ImageNet fine-tuning, versus ~80 epochs of original
  training (≈ ¼).

They report that this duration is sufficient to recover the bulk of the
accuracy lost to pruning without over-fitting the smaller subnetwork.
The 1/4 heuristic generalises from their VGG / ResNet experiments to
our DetNet setting; the empirically observed validation AUC curves during
fine-tuning confirm that 26 epochs is enough for the pruned models at all
five ratios to plateau before the budget is exhausted.

Fine-tuning at all is essential — the literature is unanimous that
one-shot pruning without recovery training loses substantial accuracy
(Han 2015, Li 2017). The 1/4 budget is the published "minimum sufficient"
duration, not an arbitrary ceiling.

**Literature.** Li 2017 § 4.1 (the 1/4-of-original-training heuristic
specifically); Han 2015 (prune-then-fine-tune in general).

---

### D18. Fine-tune training data: `rhd + cmu + gan`

**Decision.** Identical to the original DetNet training data and identical
to the Taylor calibration set ([D13](#d13-calibration-data-distribution--fine-tuning-data-distribution)).

**Rationale.** Maintains distributional consistency end-to-end: same data
trained the baseline, same data calibrated the Taylor gradients, same data
fine-tunes the pruned models. Avoids domain-shift artefacts when comparing
to the baseline.

**Implementation.**
[`finetune_pruned.py:527-528`](../finetune_pruned.py#L527-L528):
```python
default=['cmu', 'rhd', 'gan']
```
for `--datasets_train`. The `HandDataset` instantiation at
[`finetune_pruned.py:207-215`](../finetune_pruned.py#L207-L215) applies the
same scale/centre jittering and rotation augmentation as the original
DetNet training pipeline.

---

### D19. Validation set: RHD (FreiHAND held out)

**Decision.** Validation during fine-tuning uses the RHD test split for
best-checkpoint selection. FreiHAND is *never* used during fine-tuning or
checkpoint selection — it is reserved exclusively as a held-out test set
for the final reported accuracy.

**Rationale.** Prevents test-set leakage. Selecting checkpoints based on
FreiHAND performance during fine-tuning would optimistically bias the
reported FreiHAND accuracy. Using RHD (which was also in the training
mixture) as validation is acceptable because the validation split is
disjoint from the training split, and FreiHAND remains genuinely
held-out.

This is the basis for the subsequent decision (separate from pruning) to
evaluate exclusively on FreiHAND in
[`evaluate_detnet.py`](../evaluate_detnet.py) — see the corresponding
memory note in
[`memory/feedback_rhd_excluded_eval.md`](../../memory/feedback_rhd_excluded_eval.md).

---

### D20. Hyperparameter inheritance from original DetNet training

**Decision.** Every fine-tuning hyperparameter is inherited verbatim from
the original DetNet training recipe. The *only* hyperparameter changed
from the baseline-training configuration is the **number of epochs**
([D17](#d17-26-epoch-fine-tune--of-original-training-duration)). All other
controls — optimiser, learning rate, scheduler, batch size, loss
formulation, augmentation parameters — are passed through unchanged. This
principle is the implicit motivation behind several of the per-knob
defaults in `finetune_pruned.py` and underlies why
[D12](#d12-calibration-loss--task-loss-2d-heatmap-mse-only),
[D13](#d13-calibration-data-distribution--fine-tuning-data-distribution)
and [D18](#d18-fine-tune-training-data-rhd--cmu--gan) all read the same way.

**What is inherited.**

| Hyperparameter | Value | Code |
|----------------|-------|------|
| Learning rate | $10^{-3}$ | [`finetune_pruned.py:641`](../finetune_pruned.py#L641) |
| LR scheduler | StepLR (`lr_decay_step`, `gamma`) | [`finetune_pruned.py:231-234`](../finetune_pruned.py#L231-L234) |
| Optimiser | Adam (`finetune_pruned.py` default) | [`finetune_pruned.py`](../finetune_pruned.py) |
| Loss formulation | $\mathcal{L}_{\text{heat}}$ with $\lambda_{\text{hm}} = 100$ | [`finetune_pruned.py:58-100`](../finetune_pruned.py#L58-L100) (`DetLoss2D`) |
| Training data | `rhd + cmu + gan` | [`finetune_pruned.py:527-528`](../finetune_pruned.py#L527-L528) |
| Scale jittering | 0.1 | [`finetune_pruned.py:212`](../finetune_pruned.py#L212) |
| Centre jittering | 0.1 | [`finetune_pruned.py:213`](../finetune_pruned.py#L213) |
| Max rotation augmentation | $0.5\pi$ | [`finetune_pruned.py:214`](../finetune_pruned.py#L214) |
| Heatmap target $\sigma$ | 1.0 (heatmap cells) | [`Old_training/datasets/handataset.py:38`](../Old_training/datasets/handataset.py#L38) |

**Rationale.** Two reasons:

1. **Eliminate confounding variables.** Changing hyperparameters between
   the baseline-training phase and the recovery fine-tune introduces an
   ambiguity: any accuracy delta between baseline and pruned models could
   then be attributed to either the pruning *or* the hyperparameter change.
   Inheriting the original recipe collapses that to a single source —
   the pruning step is the only thing that can explain the gap.
2. **Matches Li 2017's recommendation.** Li et al. explicitly retrain
   their pruned models "using the original optimisation method and
   hyperparameters". The implicit assumption is that the pruned
   subnetwork lives near the same region of the loss landscape as the
   baseline (because the architecture is a strict subset of it), so
   the optimisation procedure that found the baseline should still be a
   reasonable recovery procedure.

The same principle applies in reverse to the Taylor calibration setup
([D12–D13](#d12-calibration-loss--task-loss-2d-heatmap-mse-only)): the
calibration loss and data must match the fine-tuning loss and data so
that the gradients used to estimate filter importance reflect the
optimisation pressure those filters would face during recovery training.

**Literature.** Li 2017 § 4.1.

---

## 8. Summary Table

| # | Decision | Key code | Primary citation |
|--:|----------|----------|------------------|
| D1 | Structured filter pruning (not unstructured) | [`prune_{l1,taylor}.py`](../prune_l1.py) (via `torch_pruning`) | Li 2017 |
| D2 | One-shot prune then fine-tune | `finetune_pruned.py` | Han 2015, Li 2017 |
| D3 | Same baseline (ep71) for all pruning runs | `prune_l1.py:30`, `prune_taylor.py:56` | (Standard) |
| D4 | L1-norm importance | `prune_l1.py:56` | Li 2017 |
| D5 | Taylor expansion importance | `prune_taylor.py:110` | Molchanov 2017, 2019 |
| D6 | Single-variate Taylor (not multi-variate) | `prune_taylor.py:110` | Molchanov 2017 |
| D7 | L1 + Taylor only (no other criteria) | (Methodology) | (Comparative-study scope) |
| D8 | Per-layer uniform allocation (`global_pruning=False`) | `prune_l1.py:62`, `prune_taylor.py:116` | Li 2017 |
| D9 | Auto dependency-graph coupling | `MagnitudePruner` `example_inputs` | Fang 2023 |
| D10 | Output head excluded | `prune_*.py` `ignored_layers` | (Task constraint) |
| D11 | Stem input unprunable | (Implicit in DepGraph) | (Boundary constraint) |
| D12 | Calibration loss = 2D heatmap MSE | `prune_taylor.py:84-101` | Molchanov 2017 |
| D13 | Calibration data = fine-tune data | `prune_taylor.py:61` | (Distribution match) |
| D14 | Single-pass calibration | `prune_taylor.py:167-179` | Molchanov 2017 |
| D15 | Cached calibration gradients | `prune_taylor.py:137-184` | (Reproducibility) |
| D16 | Ratios = [10, 25, 40, 55, 70] % | `prune_l1.py:32`, `prune_taylor.py:59` | (Sweep design) |
| D17 | 26 epochs ≈ ¼ of original training duration | `--epochs 25` to `finetune_pruned.py` | Li 2017 § 4.1 |
| D18 | Fine-tune data = rhd + cmu + gan | `finetune_pruned.py:527-528` | (Distribution match) |
| D19 | RHD validation (FreiHAND held out) | `finetune_pruned.py`; `evaluate_detnet.py` | (Leakage prevention) |
| D20 | Hyperparameter inheritance from original training | `finetune_pruned.py` defaults (LR, scheduler, loss, augmentation) | Li 2017 § 4.1 |

---

## References

Citations are listed alphabetically by first author. For each, the bibtex-ready
fields are given (author, year, title, venue, arXiv ID).

#### Fang 2023
Gongfan Fang, Xinyin Ma, Mingli Song, Michael Bi Mi, Xinchao Wang.
*DepGraph: Towards Any Structural Pruning.*
CVPR 2023. arXiv:2301.12900.
**Used for:** [D1](#d1-structured-filter-pruning-whole-channel-removal),
[D8](#d8-per-layer-uniform-allocation-global_pruningfalse),
[D9](#d9-automatic-dependency-graph-handling) — the underlying framework
implemented by the `torch_pruning` library.

#### Frankle 2019
Jonathan Frankle, Michael Carbin.
*The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.*
ICLR 2019 (best paper). arXiv:1803.03635.
**Used for:** [D2](#d2-one-shot-prune-then-fine-tune-single-round)
(contextual — discusses iterative pruning vs train-from-scratch),
[D7](#d7-why-these-two-criteria-and-not-others) (rejected as scope).

#### Han 2015
Song Han, Jeff Pool, John Tran, William Dally.
*Learning both Weights and Connections for Efficient Neural Networks.*
NeurIPS 2015. arXiv:1506.02626.
**Used for:** [D1](#d1-structured-filter-pruning-whole-channel-removal),
[D2](#d2-one-shot-prune-then-fine-tune-single-round),
[D17](#d17-long-fine-tune-with-rhd-validated-best-checkpoint-selection) —
foundational prune-then-fine-tune paradigm (Han's version is unstructured;
the structured generalisation is Li 2017).

#### Hassibi 1993
Babak Hassibi, David Stork.
*Second order derivatives for network pruning: Optimal Brain Surgeon.*
NeurIPS 1992 (published 1993).
**Used for:** [D5](#d5-taylor-expansion-importance-method-2) — historical
foundation for Taylor-based pruning (Hessian-based second-order extension
of LeCun 1990).

#### He 2017
Yihui He, Xiangyu Zhang, Jian Sun.
*Channel Pruning for Accelerating Very Deep Neural Networks.*
ICCV 2017. arXiv:1707.06168.
**Used for:** [D7](#d7-why-these-two-criteria-and-not-others) — LASSO-based
channel pruning, considered as alternative criterion and rejected as
out-of-scope.

#### He 2018 (AMC)
Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han.
*AMC: AutoML for Model Compression and Acceleration on Mobile Devices.*
ECCV 2018. arXiv:1802.03494.
**Used for:** [D7](#d7-why-these-two-criteria-and-not-others) — RL-based
per-layer allocation, considered and rejected as scope.

#### He 2019 (FPGM)
Yang He, Ping Liu, Ziwei Wang, Zhilan Hu, Yi Yang.
*Filter Pruning via Geometric Median for Deep Convolutional Neural Networks
Acceleration.* CVPR 2019. arXiv:1811.00250.
**Used for:** [D7](#d7-why-these-two-criteria-and-not-others) — alternative
magnitude-style criterion, considered and rejected.

#### LeCun 1990
Yann LeCun, John Denker, Sara Solla.
*Optimal Brain Damage.*
NeurIPS 1989 (published 1990).
**Used for:** [D5](#d5-taylor-expansion-importance-method-2) — historical
foundation: introduces the second-order Taylor expansion for weight pruning
that Molchanov 2017's first-order single-variate criterion descends from.

#### Li 2017
Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, Hans Peter Graf.
*Pruning Filters for Efficient ConvNets.*
ICLR 2017. arXiv:1608.08710.
**Used for:** [D1](#d1-structured-filter-pruning-whole-channel-removal),
[D2](#d2-one-shot-prune-then-fine-tune-single-round),
[D4](#d4-l1-norm-magnitude-importance-method-1),
[D8](#d8-per-layer-uniform-allocation-global_pruningfalse),
[D17](#d17-26-epoch-fine-tune--of-original-training-duration),
[D20](#d20-hyperparameter-inheritance-from-original-detnet-training) —
the canonical reference for structured L1-norm filter pruning. § 4.1
specifically prescribes the "1/4 of original training time" recovery
fine-tune duration cited in [D17](#d17-26-epoch-fine-tune--of-original-training-duration)
(40 epochs on CIFAR-10 vs ~160 original; 20 epochs on ImageNet vs ~80
original) and the inheritance of original training hyperparameters cited
in [D20](#d20-hyperparameter-inheritance-from-original-detnet-training).

#### Lin 2020
Mingbao Lin, Rongrong Ji, Yan Wang, Yichen Zhang, Baochang Zhang, Yonghong Tian,
Ling Shao. *HRank: Filter Pruning using High-Rank Feature Map.*
CVPR 2020. arXiv:2002.10179.
**Used for:** [D7](#d7-why-these-two-criteria-and-not-others) — feature-map
rank as importance criterion, considered and rejected.

#### Liu 2017a
Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, Changshui Zhang.
*Learning Efficient Convolutional Networks through Network Slimming.*
ICCV 2017. arXiv:1708.06519.
**Used for:** [D7](#d7-why-these-two-criteria-and-not-others) — BN-γ-based
pruning with sparsity training, considered and rejected (would change the
baseline checkpoint, breaking [D3](#d3-identical-pre-trained-baseline-for-all-pruning-runs)).

#### Liu 2019
Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, Trevor Darrell.
*Rethinking the Value of Network Pruning.*
ICLR 2019. arXiv:1810.05270.
**Used for:** [D2](#d2-one-shot-prune-then-fine-tune-single-round) —
important caveat that, for many ConvNets, training the pruned architecture
from scratch matches inherited-weight fine-tuning.

#### Molchanov 2017
Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, Jan Kautz.
*Pruning Convolutional Neural Networks for Resource Efficient Inference.*
ICLR 2017. arXiv:1611.06440.
**Used for:** [D5](#d5-taylor-expansion-importance-method-2),
[D6](#d6-single-variate-taylor-not-multi-variate),
[D12](#d12-calibration-loss--task-loss-2d-heatmap-mse-only),
[D14](#d14-single-pass-calibration-no-optimiser-step-no-multiple-epochs) —
canonical reference for single-variate first-order Taylor filter importance.

#### Molchanov 2019
Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio, Jan Kautz.
*Importance Estimation for Neural Network Pruning.*
CVPR 2019. arXiv:1906.10771.
**Used for:** [D5](#d5-taylor-expansion-importance-method-2),
[D6](#d6-single-variate-taylor-not-multi-variate),
[D8](#d8-per-layer-uniform-allocation-global_pruningfalse) — extends Molchanov
2017 with multi-variate Taylor and discusses cross-layer normalisation
for global pruning.

#### Sanh 2020
Victor Sanh, Thomas Wolf, Alexander M. Rush.
*Movement Pruning: Adaptive Sparsity by Fine-Tuning.*
NeurIPS 2020. arXiv:2005.07683.
**Used for:** [D7](#d7-why-these-two-criteria-and-not-others) — designed
for transformer fine-tuning, considered and rejected as scope.

---

## Cross-references with other thesis-supporting documents

- The **filter-selection comparison analysis** ([`analysis/compare_filter_selections.py`](compare_filter_selections.py))
  uses the same `MagnitudePruner` configuration as `prune_l1.py` and
  `prune_taylor.py` (verifying it via architecture-match against the saved
  `_noft.pth` files) to replay the per-layer decisions and measure Jaccard
  overlap + Spearman correlation between L1 and Taylor.
- The **bbox-normalised MPJPE convention** used to report accuracy is
  documented in [`memory/project_bbox_norm_mpjpe.md`](../../memory/project_bbox_norm_mpjpe.md).
- The **PCK-AUC tau range** of `[0, 30]` px used in all accuracy reporting
  is documented in [`memory/feedback_auc_tau_range.md`](../../memory/feedback_auc_tau_range.md).
- The **leakage-prevention rationale** for evaluating only on FreiHAND
  ([D19](#d19-validation-set-rhd-freihand-held-out)) is documented in
  [`memory/feedback_rhd_excluded_eval.md`](../../memory/feedback_rhd_excluded_eval.md).
