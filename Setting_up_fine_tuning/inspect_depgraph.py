"""
Diagnostic: build torch-pruning's dependency graph for DetNet and print the
coupled-channel "groups" — i.e. the relationships pruning must respect
(residual adds, BatchNorms, the features+pos_tile concat).

This is the explicit, low-level view of what tp.pruner.MagnitudePruner does
internally inside prune_l1.py. It is NOT needed for pruning — inspection only.
Read-only: does not modify the model, prune_l1.py, detnet.py, or any checkpoint.

Console  -> a compact one-line-per-group summary.
depgraph_groups.txt -> the full, verbose group listing (every dependency edge).
"""
import torch
import torch.nn as nn
import torch_pruning as tp
import utils.misc as misc
from model.detnet import detnet

CHECKPOINT = 'new_check_point/ckp_detnet_71.pth'
OUT_TXT    = 'depgraph_groups.txt'

# Weights don't affect the graph (only shapes do), but load the real checkpoint
# for consistency with the other steps (tolerant load: skip the 3D-head keys).
model = detnet()
clean = misc.clean_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
ms = model.state_dict()
model.load_state_dict({**ms, **{k: v for k, v in clean.items() if k in ms and v.shape == ms[k].shape}},
                      strict=False)
model.eval()

name_of = {m: n for n, m in model.named_modules()}

# 1) Build the dependency graph. This is the call that "automatically handles
#    ResNet-50 skip connection dependencies" (and BN, and the concat).
example_input = torch.randn(1, 3, 128, 128)
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_input)
print("Dependency graph built for DetNet.\n")

# 2) Enumerate the pruning groups. Each group = a set of (layer, channel-indices)
#    that MUST be pruned together. Exclude hmap_0.prediction (the 21-ch output),
#    exactly as prune_l1.py does.
groups = list(DG.get_all_groups(
    ignored_layers=[model.hmap_0.prediction],
    root_module_types=[nn.Conv2d, nn.Linear],
))

print(f"DetNet has {len(groups)} independent pruning groups "
      f"(hmap_0.prediction excluded as the model output).\n")
print(f"{'#':>3}  {'root layer (out-channels pruned here)':<34} {'#ch':>5}  members (every layer tied to this group)")
print("-" * 130)

with open(OUT_TXT, 'w', encoding='utf-8') as fh:
    fh.write(f"DetNet pruning dependency groups ({len(groups)} groups; hmap_0.prediction excluded)\n")
    fh.write("=" * 130 + "\n")
    for i, g in enumerate(groups):
        # root = the layer whose out-channels this group is keyed on
        root_module = g[0][0].target.module
        root_name = name_of.get(root_module, type(root_module).__name__)
        root_ch = getattr(root_module, 'out_channels', getattr(root_module, 'num_features', '?'))

        # collect distinct member layers + whether their in- or out-channels are touched
        members = []
        seen = set()
        for dep, _idxs in g:
            mod = dep.target.module
            nm = name_of.get(mod, type(mod).__name__)
            fn = getattr(dep.handler, '__name__', str(dep.handler))
            side = 'out' if 'out_channel' in fn else ('in' if 'in_channel' in fn else fn)
            key = (nm, side)
            if key not in seen:
                seen.add(key)
                members.append(f"{nm}[{side}]")

        print(f"{i:>3}  {root_name:<34} {str(root_ch):>5}  {', '.join(members)}")

        fh.write(f"\n############  GROUP {i}  (root: {root_name}, channels: {root_ch})  ############\n")
        fh.write(str(g))
        fh.write("\n")

print("-" * 130)
print(f"\nFull verbose listing (every dependency edge of every group) written to: {OUT_TXT}")
print("\nHow to read this:")
print(" - A 'group' is a set of layers whose channels are locked together by the architecture:")
print("   a residual add forces conv3's + the shortcut's out-channels to match the next stage's in-channels;")
print("   a BatchNorm shares the channel axis of the conv before it; the features+pos_tile concat ties")
print("   resnet50.squeeze.0's out-channels to part of hmap_0.project.0's in-channels.")
print(" - When you prune k filters from a group's root layer, torch-pruning removes the SAME k positions")
print("   from every member: out-channels for some layers (and their BNs), in-channels for the consumers.")
print(" - hmap_0.prediction is excluded, so its 21 output channels (the heatmap output) are never touched.")
print(" - resnet50.conv1's 3 input channels never appear as a prunable side — nothing upstream produces them.")
