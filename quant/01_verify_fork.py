"""
PHASE 1.5 - verify the detnet_quant fork is numerically identical to FP32.

Loads ep71 weights into both the original detnet and DetNetQuant (strict load),
runs the same random input through both, and checks the heatmap output matches to
< 1e-6 (it should be exactly 0.0: every quant/cat/relu op is a float identity
before convert). This proves the fork preserves the FP32 baseline before any
quantization is applied.

Usage:
  python quant\\01_verify_fork.py
"""
import qcommon as q
import torch

from detnet_quant import load_detnet_quant


def main():
    torch.manual_seed(0)

    print("[1.5] Build fork + load ep71 weights (strict)")
    fp32 = q.load_fp32("cpu")
    mq = load_detnet_quant(fp32, device="cpu")    # strict load happens inside
    print("  strict load OK")
    print(f"  original params: {sum(p.numel() for p in fp32.parameters()):,}")
    print(f"  fork params    : {sum(p.numel() for p in mq.parameters()):,}")

    print("\n[1.5] Compare outputs (original FP32 vs un-quantized fork)")
    fp32.eval()
    mq.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 128, 128)
        o_orig = fp32(x)
        o_fork = mq(x)
    hd = (o_orig["h_map"] - o_fork["h_map"]).abs()
    uv_match = torch.equal(o_orig["uv"], o_fork["uv"])
    print(f"  h_map shape    : {tuple(o_fork['h_map'].shape)}")
    print(f"  h_map max diff : {hd.max().item():.3e}")
    print(f"  h_map mean diff: {hd.mean().item():.3e}")
    print(f"  uv identical   : {uv_match}")

    assert hd.max().item() < 1e-6, "fork is NOT numerically identical to FP32"
    assert uv_match, "uv mismatch between fork and original"
    print("\n  PASS - fork preserves the FP32 baseline (ready to quantize)")


if __name__ == "__main__":
    main()
