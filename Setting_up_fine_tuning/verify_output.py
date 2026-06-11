"""
STEP 3 — Verify the original (pre-pruning) DetNet output shape.

Same tolerant checkpoint load as verify_load.py (the checkpoint carries 3D-head
keys the 2D-only model doesn't have). Runs one forward pass on a random
128x128 RGB input and prints every key/shape in the output dict.

The current detnet.forward() returns only {'h_map', 'uv'} — the 3D outputs
(d_map / l_map / xyz / delta) are commented out in model/detnet/detnet.py.
The critical invariant is h_map == (1, 21, 32, 32): 21 joint heatmaps at 32x32.
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
model.eval()

example = torch.randn(1, 3, 128, 128)
with torch.no_grad():
    output = model(example)

print("Output keys and shapes:")
for key, val in output.items():
    print(f"  output['{key}'] : {tuple(val.shape)}")

assert output['h_map'].shape == (1, 21, 32, 32), \
    f"WRONG SHAPE: {output['h_map'].shape}"
print()
print("h_map shape verified: (1, 21, 32, 32)")
print("Status: OK")
