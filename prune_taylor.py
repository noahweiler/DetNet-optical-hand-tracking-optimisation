"""
Generate and save all 5 structured-Taylor-pruned DetNet architectures.

Direct counterpart to prune_l1.py — same checkpoint, same ratios, same per-layer
uniform allocation, same protected output, same dependency-graph framework
(Fang et al. 2023, DepGraph). The ONLY differences are:
  - Taylor expansion filter importance (Molchanov et al. 2017 arXiv:1611.06440,
    extended in Molchanov et al. 2019 arXiv:1906.10771) instead of L1 norm
  - A single forward-backward calibration pass over the same training data
    used for L1 fine-tuning (rhd + cmu + gan) populates .grad on every
    weight before pruning; NO optimiser.step(). Calibration loss = L_heat only
    (MSE × lambda_hm=100), the 2D heatmap term reimplemented verbatim from
    DetLoss2D.compute_loss in finetune_pruned.py.
  - tp.importance.TaylorImportance uses multivariable=False by default
    (Molchanov 2017 single-variate |W * grad|).
  - Calibration gradients are checkpointed to GRADS_FILE after a successful
    run. On subsequent runs, if that file exists, the script SKIPS calibration
    entirely and loads the saved gradients (~3 sec). Delete the file to force
    re-calibration (e.g. for a different DATASETS subset or different
    augmentation settings).
  - copy.deepcopy(model) does NOT preserve .grad on nn.Parameter objects:
    PyTorch's Parameter.__deepcopy__ clones .data only, not .grad. The per-
    ratio loop therefore manually restores gradients onto each deepcopy
    before calling the pruner.
  - DataLoader uses num_workers=8 by default (matches finetune_pruned.py).
    Safe on the DESKTOP env (Py 3.11.14): multiprocessing.reduction chunks
    large pickled objects across multiple pipe writes. On the LAPTOP env
    (Py 3.10.20), set NUM_WORKERS=0 instead — Py 3.10 issues a single
    oversized WriteFile that exceeds the Windows anonymous-pipe size limit
    and the spawn fails with `OSError: [Errno 22]` / pickle truncation.
    The script body sits inside `if __name__ == '__main__':` (matching
    finetune_pruned.py line 492) so this guard works for any num_workers.
  - Calibration loop uses tqdm for a live progress bar (current batch / total,
    rate, ETA, last-batch loss via set_postfix); replaces the prior
    print-every-N-batches block. Cosmetic only; no effect on gradients.
"""
# Suppress OpenMP "Error #15: libiomp5md.dll already initialized" on Windows + conda:
# numpy(MKL) ships libiomp5md.dll, torch ships libomp.dll, and the second-loaded
# OMP runtime aborts unless this env var is set. Must run BEFORE any import that
# pulls in an OpenMP-linked library (cv2 / numpy / torch).
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2  # noqa: F401  — must come before torch (Windows DLL load order)
import sys; sys.path.insert(0, 'old_training')  # for datasets.handataset

import torch, copy, os
import torch.nn.functional as torch_f
import torch_pruning as tp
import numpy as np
import utils.misc as misc
from model.detnet import detnet
from datasets.handataset import HandDataset
from tqdm import tqdm

CHECKPOINT  = 'new_check_point/ckp_detnet_71.pth'
SAVE_DIR    = 'pruned_architectures'
GRADS_FILE  = 'taylor_calibration_grads.pt'  # delete to force re-calibration
RATIOS      = [0.10, 0.25, 0.40, 0.55, 0.70]
DATA_ROOT   = 'C:/Imperial/FYP Data'
DATASETS    = ['rhd', 'cmu', 'gan']  # matches L1 --datasets_train rhd cmu gan
LAMBDA_HM   = 100.0
BATCH_SIZE  = 32
NUM_WORKERS = 8  # set to 0 on Py 3.10 (Windows pickle truncation); see docstring
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


def compute_heatmap_loss(pred_hm, targ_hm, hm_veil):
    """DetLoss2D heatmap term — verbatim from finetune_pruned.py::DetLoss2D.compute_loss
    (lines 62-74), reimplemented locally so prune_taylor.py has no import dependency
    on the training script. Returns LAMBDA_HM * sum_j 0.5 * MSE(pred_j, targ_j)."""
    batch_size = pred_hm.shape[0]
    hm_veil = hm_veil.unsqueeze(-1)
    njoints = pred_hm.size(1)
    pred_hm = pred_hm.reshape((batch_size, njoints, -1)).split(1, 1)
    targ_hm = targ_hm.reshape((batch_size, njoints, -1)).split(1, 1)
    hm_loss = torch.Tensor([0]).to(pred_hm[0].device)
    for idx in range(njoints):
        pred_hmapi = pred_hm[idx].squeeze()
        targ_hmi = targ_hm[idx].squeeze()
        hm_loss += 0.5 * torch_f.mse_loss(
            pred_hmapi.mul(hm_veil[:, idx]),
            targ_hmi.mul(hm_veil[:, idx])
        )
    return LAMBDA_HM * hm_loss


def prune_structured_taylor(model, ratio, example_input):
    """Structured Taylor filter pruning (Molchanov et al. 2017/2019) on top of
    torch-pruning's auto dependency graph (Fang et al. 2023). Assumes .grad is
    already populated on the model's weights (see calibration pass below).
    hmap_0.prediction is excluded so the 21-channel heatmap output is preserved.
    Modifies the model in place."""
    imp = tp.importance.TaylorImportance()  # multivariable=False (single-variate Molchanov)
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


# Windows multiprocessing: DataLoader workers re-import this module on spawn,
# so the script body must sit inside __main__ to avoid spawn-recursion.
# Same idiom as finetune_pruned.py (line 492).
if __name__ == '__main__':
    # ── Baseline ─────────────────────────────────────────────────────────────────
    example = torch.randn(1, 3, 128, 128)
    model_orig = load_fresh_model(CHECKPOINT)
    n_before = count_params(model_orig)
    print(f"Baseline checkpoint: {CHECKPOINT}")
    print(f"Baseline parameters: {n_before:,}")
    print()

    # ── Taylor calibration: load cached grads if available, else compute and save ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(GRADS_FILE):
        print(f"Loading pre-computed calibration gradients from {GRADS_FILE} (skipping calibration)...")
        ckpt = torch.load(GRADS_FILE, map_location='cpu', weights_only=False)
        saved_grads = ckpt['grads']
        n_train_images = int(ckpt.get('n_images', 0))
        saved_datasets = ckpt.get('datasets', DATASETS)
        if saved_datasets != DATASETS:
            print(f"  WARNING: saved grads were computed on DATASETS={saved_datasets}, "
                  f"current DATASETS={DATASETS}. Delete {GRADS_FILE} to force re-calibration.")
        print(f"  Loaded grads for {len(saved_grads)} parameters "
              f"(calibrated on {n_train_images:,} images from {saved_datasets}).")
    else:
        print(f"Loading training data: {DATASETS}")
        train_dataset = HandDataset(
            data_split='train',
            train=True,
            subset_name=DATASETS,
            data_root=DATA_ROOT,
            scale_jittering=0.1,
            center_jettering=0.1,
            max_rot=0.5 * np.pi,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
        )
        n_train_images = len(train_dataset)
        print(f"Training samples: {n_train_images:,}  |  Batches (bs={BATCH_SIZE}): {len(train_loader):,}  "
              f"|  Device: {device}  |  num_workers={NUM_WORKERS}")
        print("Running Taylor calibration over full training set (no optimiser.step())...", flush=True)
        model_orig.to(device).train()
        model_orig.zero_grad()
        pbar = tqdm(train_loader, desc="Taylor calibration", unit="batch", smoothing=0.05)
        for metas in pbar:
            clr = metas['clr'].to(device, non_blocking=True)
            hm = metas['hm'].to(device, non_blocking=True)
            hm_veil = metas['hm_veil'].to(device, non_blocking=True)
            out = model_orig(clr)
            loss = compute_heatmap_loss(out['h_map'], hm, hm_veil)
            loss.backward()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        model_orig.cpu()
        print(f"Calibration complete: {len(train_loader)} batches processed")
        # Persist gradients so future re-runs skip the long calibration pass.
        saved_grads = {name: p.grad.detach().clone().cpu()
                       for name, p in model_orig.named_parameters() if p.grad is not None}
        torch.save({'grads': saved_grads, 'n_images': n_train_images, 'datasets': DATASETS}, GRADS_FILE)
        print(f"Calibration gradients saved to {GRADS_FILE} ({len(saved_grads)} parameters).")
    print()

    # ── Prune at each ratio and save ─────────────────────────────────────────────
    # copy.deepcopy drops .grad on nn.Parameter (PyTorch's Parameter.__deepcopy__
    # only clones .data), so we manually restore gradients onto each deepcopy
    # before calling the pruner.
    results = []
    for ratio in RATIOS:
        print(f"Pruning at {ratio * 100:.0f}%...", flush=True)

        m = copy.deepcopy(model_orig)
        for name, p in m.named_parameters():
            if name in saved_grads:
                p.grad = saved_grads[name].clone()
        m = prune_structured_taylor(m, ratio=ratio, example_input=example)
        m.eval()

        with torch.no_grad():
            out = m(example)

        n = count_params(m)
        shape_ok = (out['h_map'].shape == (1, 21, 32, 32) and out['uv'].shape == (1, 21, 2))

        fname = os.path.join(SAVE_DIR, f'detnet_taylor_{int(ratio * 100):02d}pct_ep71_noft.pth')
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
        print(f"  Pruning criterion:  Taylor expansion (Molchanov et al. 2017/2019, |W * grad|)")
        print(f"  Calibration data:   {DATASETS} ({n_train_images:,} images, L_heat MSE x {LAMBDA_HM})")
        print(f"  Calibration cache:  {GRADS_FILE} (delete to force re-calibration)")
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
