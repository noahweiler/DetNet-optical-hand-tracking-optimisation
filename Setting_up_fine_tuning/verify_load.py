"""
STEP 2 — Load checkpoint ckp_detnet_71.pth into the (2D-only) DetNet and confirm it loads.

The checkpoint was saved back when DetNet still had its 3D detector heads
(dmap_0 / lmap_0), so it carries 16 extra keys the current 2D-only detnet()
does not have. We therefore load only the matching keys (strict=False),
exactly as evaluate_detnet.py::load_model does. misc.clean_state_dict still
runs first to strip any DataParallel 'module.' prefix (this particular file
has none, so it is a no-op here).
"""
import torch
import utils.misc as misc
from model.detnet import detnet

CHECKPOINT = 'new_check_point/ckp_detnet_71.pth'

model = detnet()
raw_state   = torch.load(CHECKPOINT, map_location='cpu')
clean_state = misc.clean_state_dict(raw_state)            # strips 'module.' prefix (none here -> no-op)

model_state = model.state_dict()
matched = {k: v for k, v in clean_state.items()
           if k in model_state and v.shape == model_state[k].shape}
model.load_state_dict({**model_state, **matched}, strict=False)
model.eval()

skipped = sorted(set(clean_state) - set(matched))
n_params = sum(p.numel() for p in model.parameters())
print(f"Checkpoint loaded: {CHECKPOINT}")
print(f"Matched keys:      {len(matched)} / {len(clean_state)} in checkpoint  "
      f"({len(model_state)} in model)")
print(f"Skipped from ckpt: {skipped}")
print(f"Total parameters:  {n_params:,}")
print("Status: OK")
