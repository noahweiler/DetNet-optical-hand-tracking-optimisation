"""
Preflight checks for the 6-run finetune queue (L1 10% resume + 5 Taylor).
Runs in ~60 sec. Verifies everything the queue depends on, including the
new NVIDIA driver under sustained load (since the prior driver TDR'd
mid-run).

Checks (in order; first failure aborts):
  1. CUDA + GPU name + NVIDIA driver version reported
  2. All 6 pruned baseline .pth files exist
  3. L1 10% epoch-17 checkpoint (ft_l1_10pct_17.pth) exists
  4. Resume-aliased copy at ckp_detnet_17.pth in place (created if missing)
     — workaround for finetune_pruned.py's hardcoded 'ckp_detnet_<N>.pth'
       prefix in the --resume code path (line 174)
  5. Strict state_dict load test: epoch-17 weights load cleanly into the
     pruned L1 10% architecture (catches any pickle / DataParallel / key
     mismatch before the queue starts)
  6. Dataset folders + HandDataset constructs
  7. Resume-loaded forward + backward on GPU (one batch, real loss formula)
  8. GPU stability test: 20 large matmuls under sustained load (~10 sec).
     Confirms the new driver doesn't TDR like the old one did.
  9. GPU memory headroom > 4 GB free
 10. tqdm + progress.bar imports

Not a replacement for actually running the queue; just fails fast on
anything that would otherwise surface hours in.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2  # noqa: F401  — Windows DLL load order
import sys; sys.path.insert(0, 'old_training')

import shutil
import subprocess
import torch
import torch.nn.functional as torch_f
import numpy as np
import utils.misc as misc
from datasets.handataset import HandDataset

DATA_ROOT = 'C:/Imperial/FYP Data'
DATASETS = ['rhd', 'cmu', 'gan']
LAMBDA_HM = 100.0
MIN_FREE_GB = 4.0

PRUNED_BASELINES = [
    'pruned_architectures/detnet_l1_10pct_ep71_noft.pth',
    'pruned_architectures/detnet_taylor_70pct_ep71_noft.pth',
    'pruned_architectures/detnet_taylor_55pct_ep71_noft.pth',
    'pruned_architectures/detnet_taylor_40pct_ep71_noft.pth',
    'pruned_architectures/detnet_taylor_25pct_ep71_noft.pth',
    'pruned_architectures/detnet_taylor_10pct_ep71_noft.pth',
]
L1_10PCT_BASELINE = PRUNED_BASELINES[0]

EPOCH17_FILE = 'checkpoints_finetuned/l1_10pct/ft_l1_10pct_17.pth'
RESUME_FILE  = 'checkpoints_finetuned/l1_10pct/ckp_detnet_17.pth'

DATASET_FOLDERS = {'rhd': 'RHD', 'cmu': 'CMU', 'gan': 'GANeratedHands_Release'}


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
    print("Preflight for 6-run finetune queue (L1 10% resume + 5 Taylor)")
    print("-" * 60)

    ok = True

    def c1():
        assert torch.cuda.is_available(), "CUDA not available"
        gpu = torch.cuda.get_device_name(0)
        try:
            drv = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                text=True, stderr=subprocess.DEVNULL
            ).strip().splitlines()[0]
        except Exception:
            drv = 'unknown (nvidia-smi failed)'
        return f"GPU='{gpu}', driver={drv}"
    ok &= check("1/10 CUDA + GPU + driver", c1)

    def c2():
        missing = [f for f in PRUNED_BASELINES if not os.path.isfile(f)]
        assert not missing, f"missing: {missing}"
        return "all 6 baseline .pth files present"
    ok &= check("2/10 Pruned baseline files", c2)

    def c3():
        assert os.path.isfile(EPOCH17_FILE), f"missing: {EPOCH17_FILE}"
        size_mb = os.path.getsize(EPOCH17_FILE) / (1000 * 1000)   # MB (decimal megabytes)
        return f"{size_mb:.1f} MB"
    ok &= check("3/10 L1 10% epoch-17 checkpoint", c3)

    def c4():
        if os.path.isfile(RESUME_FILE):
            return f"already present at {RESUME_FILE}"
        shutil.copy2(EPOCH17_FILE, RESUME_FILE)
        return f"copied -> {RESUME_FILE}"
    ok &= check("4/10 Resume-aliased copy", c4)

    def c5():
        m = torch.load(L1_10PCT_BASELINE, map_location='cpu', weights_only=False)
        sd = torch.load(RESUME_FILE, map_location='cpu', weights_only=False)
        sd = misc.clean_state_dict(sd)
        m.load_state_dict(sd)  # strict=True default — raises on any mismatch
        return "strict state_dict load into pruned L1 10% architecture OK"
    ok &= check("5/10 Resume state_dict shape match", c5)

    def c6():
        missing = [d for d in DATASETS
                   if not os.path.isdir(os.path.join(DATA_ROOT, DATASET_FOLDERS[d]))]
        assert not missing, f"missing dataset folders: {missing}"
        ds = HandDataset(
            data_split='train', train=True, subset_name=DATASETS,
            data_root=DATA_ROOT,
            scale_jittering=0.1, center_jettering=0.1, max_rot=0.5 * np.pi,
        )
        return f"all 3 folders found, HandDataset constructs ({len(ds):,} samples)"
    ok &= check("6/10 Dataset folders + HandDataset", c6)

    def c7():
        device = torch.device('cuda')
        m = torch.load(L1_10PCT_BASELINE, map_location='cpu', weights_only=False)
        sd = torch.load(RESUME_FILE, map_location='cpu', weights_only=False)
        sd = misc.clean_state_dict(sd)
        m.load_state_dict(sd)
        m = m.to(device).train()
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
    ok &= check("7/10 Resume-loaded fwd+bwd on GPU", c7)

    def c8():
        device = torch.device('cuda')
        x = torch.randn(8192, 8192, device=device)
        last = None
        for i in range(20):
            y = (x @ x.T).sum()
            assert torch.isfinite(y), f"matmul iter {i + 1} produced non-finite result"
            last = y.item()
        torch.cuda.synchronize()
        del x, y
        torch.cuda.empty_cache()
        return f"20 large matmuls completed, final sum={last:.3e}"
    ok &= check("8/10 GPU stability test", c8)

    def c9():
        free, total = torch.cuda.mem_get_info(0)
        free_gb = free / (1024 ** 3)
        assert free_gb > MIN_FREE_GB, f"only {free_gb:.2f} GB free (need > {MIN_FREE_GB})"
        return f"{free_gb:.2f} GB free / {total / (1024 ** 3):.2f} GB total"
    ok &= check("9/10 GPU memory headroom", c9)

    def c10():
        import tqdm
        from progress.bar import Bar  # noqa: F401
        return f"tqdm=={tqdm.__version__}, progress.bar.Bar OK"
    ok &= check("10/10 tqdm + progress imports", c10)

    print("-" * 60)
    if ok:
        print("ALL CHECKS PASSED - safe to run the 6-run finetune queue")
        sys.exit(0)
    else:
        print("ONE OR MORE CHECKS FAILED - do NOT run the queue until fixed")
        sys.exit(1)
