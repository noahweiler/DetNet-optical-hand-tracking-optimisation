"""
Print per-submodule parameter count and FLOPs for the 2D-only DetNet model.

Reproduces the style of the FYP architecture diagram:
    ResNet-50 backbone : x.x GFLOPs · xx.xM params
    Heatmap head       : x.x GFLOPs · xx.xM params
    Total              : x.x GFLOPs · xx.xM params

Run from project root with the conda env active:
    python compute_model_stats.py
"""
import torch
from thop import profile as thop_profile

from model.detnet import detnet

INPUT_SIZE   = 128                  # DetNet expects 128x128 RGB
FEATURE_DIM  = 32                   # spatial resolution after ResNet50
HMAP_IN_CH   = 256 + 2              # 256 ResNet features + 2 positional channels


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def count_gflops(module, dummy_input):
    """thop counts MACs (multiply-accumulate). FLOPs = 2 × MACs."""
    with torch.no_grad():
        macs, _ = thop_profile(module, inputs=(dummy_input,), verbose=False)
    return macs * 2 / 1e9


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model  = detnet().to(device).eval()

    # Submodule input shapes match the live forward pass:
    #   ResNet50    : (1, 3, 128, 128)        — raw image tensor
    #   Heatmap head: (1, 258, 32, 32)        — features + pos_tile concat
    rn_input = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    hm_input = torch.zeros(1, HMAP_IN_CH, FEATURE_DIM, FEATURE_DIM, device=device)

    sections = [
        ('ResNet-50 backbone', model.resnet50, rn_input),
        ('Heatmap head',       model.hmap_0,   hm_input),
    ]

    print(f'\nDevice: {device}')
    print(f'{"":-<54}')
    print(f'{"Component":<22} {"GFLOPs":>10} {"Params":>14}')
    print(f'{"":-<54}')

    total_gflops = 0.0
    total_params = 0
    for name, module, dummy in sections:
        gflops = count_gflops(module, dummy)
        params = count_params(module)
        total_gflops += gflops
        total_params += params
        print(f'{name:<22} {gflops:>10.3f} {params:>14,}')

    full_params = count_params(model)   # sanity check — should equal total_params
    print(f'{"":-<54}')
    print(f'{"Total":<22} {total_gflops:>10.3f} {total_params:>14,}')
    if full_params != total_params:
        print(f'(Note: full model has {full_params:,} params — '
              f'difference {full_params - total_params:,} lives outside the listed submodules.)')
    print()


if __name__ == '__main__':
    main()
