"""STEP 7 — confirm a pruned model survives a forward+backward+step (architecture
preserved, heatmap loss computes cleanly, no NaN). Runs on CPU; no data needed."""
import torch
import torch.nn.functional as F

MODEL = 'pruned_architectures/detnet_l1_10pct_ep71_noft.pth'
LAMBDA_HM = 100  # from train_detnet.py: losses.DetLoss(lambda_hm=100., ...)

# torch >= 2.6 -> weights_only=True by default refuses to unpickle nn.Module objects
model = torch.load(MODEL, map_location='cpu', weights_only=False)
model.train()

def count(m): return sum(p.numel() for p in m.parameters())

n_before = count(model)
print(f'Parameters before gradient step: {n_before:,}')

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
x = torch.randn(2, 3, 128, 128)

with torch.no_grad():
    out0 = model(x)
target = torch.rand_like(out0['h_map'])

out = model(x)
lH = F.mse_loss(out['h_map'], target) * LAMBDA_HM
print(f'lH value: {lH.item():.4f}')
print(f'lH is NaN: {torch.isnan(lH).item()}')
print(f'Output keys: {list(out.keys())}')

lH.backward()
opt.step()
opt.zero_grad()

n_after = count(model)
print(f'Parameters after gradient step: {n_after:,}')

model.eval()
with torch.no_grad():
    out2 = model(x)
print(f'Output shape after step: {tuple(out2["h_map"].shape)}')

assert not torch.isnan(lH), 'NaN LOSS'
assert n_before == n_after, f'PARAM COUNT CHANGED: {n_before} -> {n_after}'
assert out2['h_map'].shape[1:] == (21, 32, 32), 'SHAPE CHANGED'
print()
print('Architecture preserved: VERIFIED OK')
print('Heatmap loss computable: VERIFIED OK')
