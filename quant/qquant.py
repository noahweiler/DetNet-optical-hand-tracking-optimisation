"""
Shared INT8 static-quantization PROCEDURE for Method 2 (MinMax) and Method 3
(Histogram-MSE). This defines NO model - it operates on the one model,
DetNetQuant (detnet_quant.py). M2 and M3 are identical except the activation
observer, so the pipeline lives here once.

Pipeline (eager static quantization):
  load fork -> set qconfig -> exempt FP32 layers (Decision 9) -> fuse (44 groups)
  -> prepare -> calibrate (480 imgs) -> convert -> empirical bias correction.

Weights: per-channel symmetric, restricted range [-127, 127] for BOTH methods
(Decisions 4/5/6). Only the conv1 stem (+ its bn1/relu) and the final prediction
conv stay FP32 (Decision 9).
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import json

import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic as nni

import qcommon as q
import qcalib
from detnet_quant import load_detnet_quant, fusion_list

from torch.ao.quantization import QConfig, default_qconfig, get_default_qconfig
from torch.ao.quantization.observer import PerChannelMinMaxObserver

torch.backends.quantized.engine = "x86"   # FBGEMM/AVX-512 VNNI on Ryzen Zen 4

# Decision 6: restricted-range symmetric per-channel weights for ALL methods.
RESTRICTED_WEIGHT = PerChannelMinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
    quant_min=-127, quant_max=127)

# Decision 9: only the conv1 stem (+ its bn1/relu) and the prediction conv stay FP32.
FP32_EXEMPT = ["resnet50.conv1", "resnet50.bn1", "resnet50.relu", "hmap_0.prediction"]

# Conv types present in the FP32 reference (after fuse+prepare, before convert).
FP32_CONV_TYPES = (nn.Conv2d, nni.ConvBnReLU2d, nni.ConvBn2d, nni.ConvReLU2d)


def make_qconfig(activation_observer):
    """Method qconfig: method-specific activation observer + shared restricted weight."""
    return QConfig(activation=activation_observer, weight=RESTRICTED_WEIGHT)


# Decision 4/5: M2 = MinMax (per-tensor, asymmetric quint8); M3 = Histogram (MSE).
M2_QCONFIG = make_qconfig(default_qconfig.activation)
M3_QCONFIG = make_qconfig(get_default_qconfig("x86").activation)


def prepare_model(qconfig):
    """Load fork, set qconfig, exempt FP32 layers, fuse (44 groups), prepare."""
    m = load_detnet_quant(device="cpu")
    m.qconfig = qconfig
    mods = dict(m.named_modules())
    for name in FP32_EXEMPT:
        mods[name].qconfig = None
    torch.ao.quantization.fuse_modules(m, fusion_list(m), inplace=True)
    torch.ao.quantization.prepare(m, inplace=True)
    return m


def calibrate(model, loader):
    model.eval()
    with torch.no_grad():
        for b in loader:
            model(b[0])


def apply_empirical_bias_correction(fp32_ref, int8_model, loader):
    """Nagel 2021 empirical bias correction: b_corrected = b - (E[Wq x] - E[W x]),
    means over the calibration set (per output channel). fp32_ref and int8_model
    share module names so corrections align by name."""
    fp32_batches, int8_batches = {}, {}

    hooks = []
    for name, mod in fp32_ref.named_modules():
        if isinstance(mod, FP32_CONV_TYPES):
            def mk(n):
                def hook(m, i, o):
                    fp32_batches.setdefault(n, []).append(o.detach().mean(dim=[0, 2, 3]))
                return hook
            hooks.append(mod.register_forward_hook(mk(name)))
    calibrate(fp32_ref, loader)
    for h in hooks:
        h.remove()
    fp32_means = {n: torch.stack(v).mean(0) for n, v in fp32_batches.items()}

    hooks = []
    for name, mod in int8_model.named_modules():
        if isinstance(mod, nnq.Conv2d):
            def mk(n):
                def hook(m, i, o):
                    out = o.dequantize() if o.is_quantized else o
                    int8_batches.setdefault(n, []).append(out.detach().mean(dim=[0, 2, 3]))
                return hook
            hooks.append(mod.register_forward_hook(mk(name)))
    calibrate(int8_model, loader)
    for h in hooks:
        h.remove()
    int8_means = {n: torch.stack(v).mean(0) for n, v in int8_batches.items()}

    n_corr = 0
    for name, mod in int8_model.named_modules():
        if isinstance(mod, nnq.Conv2d) and name in fp32_means and name in int8_means:
            b = mod.bias()
            if b is not None:
                delta = int8_means[name] - fp32_means[name]      # E[Wq x] - E[W x]
                mod.set_weight_bias(mod.weight(), b.clone() - delta)
                n_corr += 1
    return n_corr


def quantize(qconfig):
    """Full pipeline -> (int8_model, n_bias_corrected)."""
    loader = qcalib.load_calib_loader()
    m = prepare_model(qconfig)
    fp32_ref = copy.deepcopy(m)              # FP32 reference (same module names as int8 model)
    calibrate(m, loader)                     # observers learn activation ranges
    torch.ao.quantization.convert(m, inplace=True)
    n_corr = apply_empirical_bias_correction(fp32_ref, m, loader)
    return m, n_corr


def run_method(tag, method_name, qconfig, max_samples=999_999):
    """Quantize, save model, then measure size / latency / accuracy and save JSON."""
    print("=" * 64)
    print(f"  {method_name}  (tag={tag})")
    print("=" * 64)

    print("\n[quantize] fuse -> prepare -> calibrate(480) -> convert -> bias correct")
    model, n_corr = quantize(qconfig)
    n_q = sum(1 for _, md in model.named_modules() if isinstance(md, nnq.Conv2d))
    print(f"  INT8 conv layers: {n_q}   bias-corrected: {n_corr}")

    pth = os.path.join(q.QUANT_DIR, f"{tag}.pth")
    torch.save(model.state_dict(), pth)   # INT8 state_dict (a torch.save(model) pickle does NOT round-trip)
    print(f"  saved state_dict -> {pth}  (reload: re-run quantize(), or rebuild fork+convert then load_state_dict)")

    print("\n[latency] batch=1 CPU, 500 passes / 50 warmup")
    lat, lat_std = q.measure_latency(model, device="cpu")
    print(f"  {lat:.2f} +/- {lat_std:.2f} ms/frame")

    print("\n[size]")
    size_mb = q.state_dict_size_mb(model)
    print(f"  {size_mb:.2f} MB")

    tag_s = "ALL" if max_samples >= 999_999 else f"{max_samples}/split"
    print(f"\n[accuracy] FreiHAND eval+train (max_samples={tag_s})")
    metrics = q.eval_model(model, "cpu", max_samples=max_samples)

    with open(os.path.join(q.QUANT_DIR, "baseline_fp32.json")) as f:
        base = json.load(f)
    print(f"\n  {'split':15s} {'MPJPE  (FP32)':22s} {'AUC  (FP32)'}")
    for split, d in metrics.items():
        bm = base["metrics"][split]
        print(f"  {split:15s} {d['mpjpe']:.4f} ({bm['mpjpe']:.4f})      "
              f"{d['auc_pck']:.2f} ({bm['auc_pck']:.2f})")
    sp = base["latency_ms"] / lat
    print(f"\n  speedup vs FP32: {sp:.2f}x   |   size: {size_mb:.2f} MB (FP32 {base['size_mb']:.1f} MB)")

    out = {
        "method": method_name, "tag": tag,
        "size_mb": size_mb,
        "latency_ms": lat, "latency_std_ms": lat_std, "latency_batch_size": 1,
        "num_threads": torch.get_num_threads(),
        "speedup_vs_fp32": sp,
        "n_bias_corrected": n_corr,
        "max_samples": max_samples,
        "metrics": metrics,
    }
    jpath = os.path.join(q.QUANT_DIR, f"{tag}_results.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved results -> {jpath}")
    return out
