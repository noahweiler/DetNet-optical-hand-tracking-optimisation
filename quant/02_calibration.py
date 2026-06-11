"""
PHASE 2 - build the calibration set (160 CMU + 160 RHD + 160 GAN = 480 imgs).

Crops exactly like the original DetNet pipeline (GT kp2d -> center/scale -> 128x128
-> normalize), using the surviving .cache/*.pkl files only as the annotation source
(read-only). Saves:
  quant/calibration_indices.json  - seed + selected indices (reproducibility)
  quant/calib_tensors.pt          - the 480 preprocessed (480,3,128,128) tensors
Then smoke-tests the loader through the FP32 model.

Usage:
  python quant\\02_calibration.py
"""
import qcommon as q
import qcalib
import torch


def main():
    print("=" * 64)
    print("  PHASE 2 - calibration data")
    print("=" * 64)

    print("\n[2.1/2.2] Sample 160 imgs/dataset (seed 42) + crop (original DetNet pipeline)")
    X = qcalib.build_and_save()
    print(f"  calibration tensor : {tuple(X.shape)}  dtype={X.dtype}")
    print(f"  value range        : [{X.min():.3f}, {X.max():.3f}]  (expect ~[-0.5, 0.5])")
    print(f"  indices saved      : {qcalib.CALIB_INDICES}")
    print(f"  tensors saved      : {qcalib.CALIB_TENSORS}")

    print("\n[2.2] Build calibration loader")
    loader = qcalib.load_calib_loader()
    print(f"  {len(loader.dataset)} images, {len(loader)} batches (batch={qcalib.BATCH})")

    print("\n[2.3] Smoke test one batch through the FP32 model")
    model = q.load_fp32("cpu")
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0]
            out = model(imgs)
            print(f"  input  : {tuple(imgs.shape)}  dtype={imgs.dtype}")
            print(f"  output : h_map {tuple(out['h_map'].shape)}, uv {tuple(out['uv'].shape)}")
            break

    print("\n  Calibration ready.")


if __name__ == "__main__":
    main()
