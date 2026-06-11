"""
prune_one.py — Generate ONE structured-pruned DetNet architecture at a single
ratio, for either the L1 or Taylor criterion.

prune_l1.py / prune_taylor.py are hardcoded to the original five ratios
[0.10, 0.25, 0.40, 0.55, 0.70] and have no CLI, so they cannot add a new ratio
(e.g. 85%) without re-running everything. This script adds exactly one ratio,
using the IDENTICAL importance + pruner configuration as those scripts:

  L1     : tp.importance.MagnitudeImportance(p=1)            (Li et al. 2017)
  Taylor : tp.importance.TaylorImportance()                  (Molchanov 2017/2019)
           — reuses the cached calibration gradients in
             taylor_calibration_grads.pt (written by prune_taylor.py), so NO
             re-calibration pass is needed.

Both: per-layer uniform (global_pruning=False), auto dependency graph
(Fang et al. 2023), hmap_0.prediction excluded so h_map stays (B,21,32,32).
Saved as a whole model object → load with torch.load(path, weights_only=False).
The forward output shape is validated; if pruning collapses a layer the script
EXITS NON-ZERO (so a chained `&&` aborts before a wasted fine-tune).

Usage:
  python prune_one.py --method l1     --ratio 85
  python prune_one.py --method taylor --ratio 85
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Windows cv2+torch OMP Error #15

import cv2  # noqa: F401 — keep ABOVE torch (Windows DLL load order)

import argparse
import copy
import sys

import torch
import torch_pruning as tp

import utils.misc as misc
from model.detnet import detnet

CHECKPOINT = 'new_check_point/ckp_detnet_71.pth'
SAVE_DIR   = 'pruned_architectures'
GRADS_FILE = 'taylor_calibration_grads.pt'   # written by prune_taylor.py


def load_fresh_model(ckp_path):
    """Fresh DetNet with epoch-71 weights; tolerant load (checkpoint carries 16
    extra dmap_0.*/lmap_0.* keys from the removed 3D heads). Verbatim from
    prune_l1.py / prune_taylor.py::load_fresh_model."""
    m = detnet()
    state = misc.clean_state_dict(torch.load(ckp_path, map_location='cpu'))
    ms = m.state_dict()
    matched = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
    m.load_state_dict({**ms, **matched}, strict=False)
    return m


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser(description='Prune ONE DetNet ratio (L1 or Taylor)')
    ap.add_argument('--method', required=True, choices=['l1', 'taylor'])
    ap.add_argument('--ratio', required=True, type=int,
                    help='per-layer pruning ratio as an integer percent, e.g. 85')
    args = ap.parse_args()

    if not 1 <= args.ratio <= 99:
        sys.exit(f'--ratio must be 1..99, got {args.ratio}')
    ratio = args.ratio / 100.0
    os.makedirs(SAVE_DIR, exist_ok=True)
    example = torch.randn(1, 3, 128, 128)

    model = load_fresh_model(CHECKPOINT)
    n_before = count_params(model)
    print(f'Baseline checkpoint : {CHECKPOINT}')
    print(f'Baseline parameters : {n_before:,}')
    print(f'Method / ratio      : {args.method} / {args.ratio}%')

    if args.method == 'l1':
        imp = tp.importance.MagnitudeImportance(p=1)
    else:
        # Taylor needs .grad populated. Reuse prune_taylor.py's cached grads
        # (deepcopy/Parameter.__deepcopy__ drops .grad, so restore explicitly).
        if not os.path.isfile(GRADS_FILE):
            sys.exit(f'{GRADS_FILE} not found. Run `python prune_taylor.py` once '
                     f'first (it computes + caches the calibration gradients).')
        print(f'Loading Taylor calibration grads : {GRADS_FILE}')
        ckpt = torch.load(GRADS_FILE, map_location='cpu', weights_only=False)
        saved_grads = ckpt['grads']
        print(f'  grads for {len(saved_grads)} params '
              f'(calibrated on {int(ckpt.get("n_images", 0)):,} images, '
              f'datasets={ckpt.get("datasets")})')
        model = copy.deepcopy(model)
        for name, p in model.named_parameters():
            if name in saved_grads:
                p.grad = saved_grads[name].clone()
        imp = tp.importance.TaylorImportance()

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example,
        importance=imp,
        pruning_ratio=ratio,
        global_pruning=False,
        ignored_layers=[model.hmap_0.prediction],
    )
    pruner.step()
    model.eval()

    with torch.no_grad():
        out = model(example)
    shape_ok = (out['h_map'].shape == (1, 21, 32, 32)
                and out['uv'].shape == (1, 21, 2))

    n_after = count_params(model)
    reduction = (1 - n_after / n_before) * 100
    print(f'Pruned parameters   : {n_after:,}  ({reduction:.1f}% reduction)')

    if not shape_ok:
        print(f'ERROR: pruning at {args.ratio}% broke the output shape '
              f"(h_map={tuple(out['h_map'].shape)}, uv={tuple(out['uv'].shape)}). "
              f'A layer likely collapsed to 0 channels at this ratio. '
              f'NOT saving. Pick a lower ratio.')
        sys.exit(1)

    fname = os.path.join(SAVE_DIR,
                         f'detnet_{args.method}_{args.ratio:02d}pct_ep71_noft.pth')
    torch.save(model, fname)
    size_mb = os.path.getsize(fname) / (1000 * 1000)   # MB (decimal megabytes)
    print(f'Saved               : {fname}  ({size_mb:.1f} MB)')
    print('Shape check         : OK (h_map (1,21,32,32), uv (1,21,2))')
    print('This model is NOT fine-tuned yet (_noft). Fine-tune before evaluating.')


if __name__ == '__main__':
    main()
