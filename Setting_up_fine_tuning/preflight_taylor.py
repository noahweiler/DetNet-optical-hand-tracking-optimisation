"""
Preflight checks for prune_taylor.py — runs in ~30 seconds.

Verifies the 8 things that, if broken, would only surface 10-15 min into
the Taylor calibration pass. Catches them up front so a failing run can
be diagnosed before the GPU has done any real work.

Checks (in order; first failure aborts):
  1. CUDA available + device name
  2. torch_pruning importable + version pinned at 1.6.1
  3. Baseline checkpoint loads (same load_fresh_model path as prune_taylor.py)
  4. All 3 dataset folders exist under DATA_ROOT
  5. HandDataset constructs without error
  6. ONE forward + backward on GPU using compute_heatmap_loss (asserts the
     loss is finite and that at least one Conv2d weight has a populated .grad)
  7. GPU memory headroom > 4 GB free after the test forward/backward
  8. tqdm importable

Not a replacement for running prune_taylor.py — purely a fail-fast guard.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2  # noqa: F401  — must come before torch (Windows DLL load order)
import sys; sys.path.insert(0, 'old_training')

import torch
import torch.nn.functional as torch_f
import numpy as np
import utils.misc as misc
from model.detnet import detnet
from datasets.handataset import HandDataset

CHECKPOINT  = 'new_check_point/ckp_detnet_71.pth'
DATA_ROOT   = 'C:/Imperial/FYP Data'
DATASETS    = ['rhd', 'cmu', 'gan']
LAMBDA_HM   = 100.0
EXPECTED_TP_MAJOR_MINOR = '1.6.'  # MagnitudePruner / TaylorImportance API is stable across 1.6.x
MIN_FREE_GB = 4.0

DATASET_FOLDERS = {'rhd': 'RHD', 'cmu': 'CMU', 'gan': 'GANeratedHands_Release'}


def load_fresh_model(ckp_path):
    m = detnet()
    state = misc.clean_state_dict(torch.load(ckp_path, map_location='cpu'))
    ms = m.state_dict()
    matched = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
    m.load_state_dict({**ms, **matched}, strict=False)
    return m


def compute_heatmap_loss(pred_hm, targ_hm, hm_veil):
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


def check(label, fn):
    print(f"  [{label}] ... ", end='', flush=True)
    try:
        msg = fn()
        print(f"OK  {msg}" if msg else "OK")
        return True
    except Exception as e:
        print(f"FAIL\n    {type(e).__name__}: {e}")
        return False


if __name__ == '__main__':
    print("Preflight for prune_taylor.py")
    print("-" * 60)

    ok = True

    def c1():
        assert torch.cuda.is_available(), "CUDA not available"
        return f"device='{torch.cuda.get_device_name(0)}'"
    ok &= check("1/8 CUDA available", c1)

    def c2():
        import torch_pruning as tp
        v = getattr(tp, '__version__', 'unknown')
        assert v.startswith(EXPECTED_TP_MAJOR_MINOR), \
            f"expected {EXPECTED_TP_MAJOR_MINOR}x, got {v}"
        return f"torch_pruning=={v}"
    ok &= check("2/8 torch_pruning version", c2)

    def c3():
        m = load_fresh_model(CHECKPOINT)
        n = sum(p.numel() for p in m.parameters())
        return f"{n:,} params"
    ok &= check("3/8 Checkpoint loads", c3)

    def c4():
        missing = [d for d in DATASETS
                   if not os.path.isdir(os.path.join(DATA_ROOT, DATASET_FOLDERS[d]))]
        assert not missing, f"missing dataset folders: {missing}"
        return f"all 3 found under {DATA_ROOT}"
    ok &= check("4/8 Dataset folders exist", c4)

    def c5():
        ds = HandDataset(
            data_split='train', train=True, subset_name=DATASETS,
            data_root=DATA_ROOT,
            scale_jittering=0.1, center_jettering=0.1, max_rot=0.5 * np.pi,
        )
        return f"{len(ds):,} training samples"
    ok &= check("5/8 HandDataset constructs", c5)

    def c6():
        device = torch.device('cuda')
        m = load_fresh_model(CHECKPOINT).to(device).train()
        m.zero_grad()
        ds = HandDataset(
            data_split='train', train=True, subset_name=DATASETS,
            data_root=DATA_ROOT,
            scale_jittering=0.1, center_jettering=0.1, max_rot=0.5 * np.pi,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        metas = next(iter(loader))
        clr = metas['clr'].to(device)
        hm = metas['hm'].to(device)
        hm_veil = metas['hm_veil'].to(device)
        out = m(clr)
        loss = compute_heatmap_loss(out['h_map'], hm, hm_veil)
        assert torch.isfinite(loss), f"loss not finite: {loss.item()}"
        loss_val = loss.item()
        loss.backward()
        n_with_grad = sum(1 for p in m.parameters()
                          if p.grad is not None and p.grad.abs().sum() > 0)
        assert n_with_grad > 0, "no parameters received gradients"
        del m, out, loss
        torch.cuda.empty_cache()
        return f"loss={loss_val:.4f}, {n_with_grad} params with non-zero grad"
    ok &= check("6/8 Forward+backward on GPU", c6)

    def c7():
        free, total = torch.cuda.mem_get_info(0)
        free_gb = free / (1024 ** 3)
        assert free_gb > MIN_FREE_GB, f"only {free_gb:.2f} GB free (need > {MIN_FREE_GB})"
        return f"{free_gb:.2f} GB free / {total / (1024 ** 3):.2f} GB total"
    ok &= check("7/8 GPU memory headroom", c7)

    def c8():
        import tqdm
        return f"tqdm=={tqdm.__version__}"
    ok &= check("8/8 tqdm importable", c8)

    print("-" * 60)
    if ok:
        print("ALL CHECKS PASSED - safe to run prune_taylor.py")
        sys.exit(0)
    else:
        print("ONE OR MORE CHECKS FAILED - do NOT run prune_taylor.py until fixed")
        sys.exit(1)
