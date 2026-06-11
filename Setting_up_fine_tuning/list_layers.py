"""
STEP 4 — Diagnostic: list every Conv2d layer in the (2D-only) DetNet.

For our own understanding before pruning. torch-pruning's MagnitudePruner
builds and uses the dependency graph internally (DepGraph, Fang et al. 2023,
arXiv:2301.12900) — we do NOT build or pass it manually. This is inspection only.

Same tolerant checkpoint load as the earlier steps (the checkpoint carries
3D-head keys the current model doesn't have).
"""
import torch
import utils.misc as misc
from model.detnet import detnet

CHECKPOINT = 'new_check_point/ckp_detnet_71.pth'

model = detnet()
clean_state = misc.clean_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
model_state = model.state_dict()
matched = {k: v for k, v in clean_state.items()
           if k in model_state and v.shape == model_state[k].shape}
model.load_state_dict({**model_state, **matched}, strict=False)

print(f"{'#':<4} {'Layer name':<40} {'out_ch':>7} {'in_ch':>7} {'weight shape':<22} cfg")
print("-" * 100)
conv_count = 0
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        conv_count += 1
        flag = ""
        if module.out_channels == 21:
            flag = "  <<< HEATMAP OUTPUT (21 channels - auto-protected as model output)"
        if module.in_channels == 3:
            flag = "  <<< FIRST CONV (3 input channels - input dim protected)"
        cfg = f"k{module.kernel_size[0]} s{module.stride[0]} d{module.dilation[0]}"
        print(f"{conv_count:<4} {name:<40} {module.out_channels:>7} "
              f"{module.in_channels:>7} {str(tuple(module.weight.shape)):<22} {cfg}{flag}")

print()
print(f"Total Conv2d layers: {conv_count}")
print()
print("NOTES:")
print("- 1 conv has out_channels=21 (hmap_0.prediction) -> the heatmap output;")
print("  torch-pruning auto-protects it as a model-output dimension.")
print("- 1 conv has in_channels=3 (resnet50.conv1) -> RGB input; auto-protected.")
print("- resnet50.block1/block2/block3 + resnet50.squeeze.0 = the backbone (main targets).")
print("- hmap_0.project.0 = the 2D head's intermediate conv (also prunable).")
print("- The 3D detector heads (dmap_0/lmap_0) are commented out in detnet.py,")
print("  so there are NO 3D-detector conv layers in this model.")
