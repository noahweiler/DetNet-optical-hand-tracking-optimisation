"""
STEP 8 — Generate and save all 5 structured-L1-pruned DetNet architectures.

All five are pruned from the SAME checkpoint (epoch 71), so the only variable is
the compression ratio. Each is saved as a WHOLE model object (torch.save(model)),
i.e. a self-contained pruned copy of DetNet — load it later with
torch.load(path, weights_only=False) for fine-tuning / evaluation / the webcam demo.

The "_noft" suffix means "no fine-tuning yet": these models have been pruned but
not retrained. Do NOT evaluate them as results — they must be fine-tuned first.

Methodology (Li et al. 2017, "Pruning Filters for Efficient ConvNets"):
  - L1-norm filter importance        -> tp.importance.MagnitudeImportance(p=1)
  - per-layer uniform ratio          -> global_pruning=False
  - one-shot prune (fine-tune later)
  - residual/BN/concat coupling       -> resolved automatically by torch-pruning's
                                         dependency graph (Fang et al. 2023, DepGraph)
  - hmap_0.prediction excluded via ignored_layers, so h_map stays (B, 21, 32, 32)
  - resnet50.conv1's 3 RGB input channels are never pruned (no upstream producer)

A requested per-layer channel ratio r removes ceil(N*r) channels per coupled group,
and each removed filter also drops a downstream input channel, so the realised
parameter reduction is well above r (e.g. r=0.10 -> ~19%, r=0.40 -> ~64%).
"""
import torch, copy, os
import torch_pruning as tp
import utils.misc as misc
from model.detnet import detnet

CHECKPOINT = 'new_check_point/ckp_detnet_71.pth'
SAVE_DIR   = 'pruned_architectures'
RATIOS     = [0.10, 0.25, 0.40, 0.55, 0.70]
os.makedirs(SAVE_DIR, exist_ok=True)


def load_fresh_model(ckp_path):
    """Fresh DetNet with epoch-71 weights; tolerant load (checkpoint carries 16
    extra dmap_0.*/lmap_0.* keys from the removed 3D heads). Same as
    evaluate_detnet.py::load_model."""
    m = detnet()
    state = misc.clean_state_dict(torch.load(ckp_path, map_location='cpu'))
    ms = m.state_dict()
    matched = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
    m.load_state_dict({**ms, **matched}, strict=False)
    return m


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def prune_structured_l1(model, ratio, example_input):
    """Structured L1 filter pruning (Li et al. 2017) on top of torch-pruning's
    auto dependency graph (Fang et al. 2023). hmap_0.prediction is excluded so
    the 21-channel heatmap output is preserved. Modifies the model in place."""
    imp = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_input,
        importance=imp,
        pruning_ratio=ratio,
        global_pruning=False,
        ignored_layers=[model.hmap_0.prediction],
    )
    pruner.step()
    return model


# ── Baseline ─────────────────────────────────────────────────────────────────
example = torch.randn(1, 3, 128, 128)
model_orig = load_fresh_model(CHECKPOINT)
n_before = count_params(model_orig)
print(f"Baseline checkpoint: {CHECKPOINT}")
print(f"Baseline parameters: {n_before:,}")
print()

# ── Prune at each ratio and save ─────────────────────────────────────────────
results = []
for ratio in RATIOS:
    print(f"Pruning at {ratio * 100:.0f}%...", flush=True)

    m = copy.deepcopy(model_orig)
    m = prune_structured_l1(m, ratio=ratio, example_input=example)
    m.eval()

    with torch.no_grad():
        out = m(example)

    n = count_params(m)
    shape_ok = (out['h_map'].shape == (1, 21, 32, 32) and out['uv'].shape == (1, 21, 2))

    fname = os.path.join(SAVE_DIR, f'detnet_l1_{int(ratio * 100):02d}pct_ep71_noft.pth')
    torch.save(m, fname)
    size_mb = os.path.getsize(fname) / (1000 * 1000)   # MB (decimal megabytes)

    results.append({
        'ratio': ratio,
        'params': n,
        'reduction': (1 - n / n_before) * 100,
        'shape_ok': shape_ok,
        'file': fname,
        'size_mb': size_mb,
    })

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print(f"{'Ratio':>7}  {'Params':>12}  {'Reduction':>10}  {'Size MB':>8}  {'Shape OK':>9}  File")
print("-" * 95)
all_ok = True
for r in results:
    status = "YES" if r['shape_ok'] else "NO - STOP"
    if not r['shape_ok']:
        all_ok = False
    print(f"{r['ratio'] * 100:6.0f}%  {r['params']:12,}  {r['reduction']:9.1f}%  "
          f"{r['size_mb']:7.1f}  {status:>9}  {r['file']}")

print()
if all_ok:
    print("=" * 60)
    print("ALL 5 PRUNED ARCHITECTURES SAVED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  Checkpoint used:    {CHECKPOINT}")
    print(f"  Baseline params:    {n_before:,}")
    print(f"  Pruning criterion:  L1 norm (Li et al. 2017)")
    print(f"  Strategy:           per-layer uniform (global_pruning=False)")
    print(f"  Dependency graph:   auto (Fang et al. 2023, DepGraph)")
    print(f"  Output protected:   h_map 21 channels (ignored_layers=[hmap_0.prediction])")
    print(f"  Saved as:           whole model objects -> load with torch.load(path, weights_only=False)")
    print(f"  Files saved in:     {SAVE_DIR}/")
    print()
    print("These models have NOT been fine-tuned yet (_noft suffix).")
    print("Do NOT evaluate them as results.")
    print("Next step: fine-tuning loop setup.")
else:
    print("ERROR: one or more models has wrong output shape.")
    print("Do NOT proceed. Tell me which ratio failed.")
