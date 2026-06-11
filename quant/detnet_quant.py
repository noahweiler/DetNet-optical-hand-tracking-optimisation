"""
quant/detnet_quant.py - quantization-enabled fork of DetNet 2D.

GOAL: behave EXACTLY like the original (model/detnet/detnet.py +
model/helper/resnet_helper.py) - same modules, same forward sequence, same
weights - with the ONLY differences being the minimal ops eager-mode static
quantization requires. Every such change is tagged `# QUANT:` below. The five
(and only five) changes vs the originals are:

  1. Bottleneck's single shared `self.relu` -> `self.relu1`, `self.relu2`
     (eager fusion needs a distinct ReLU module per Conv-BN-ReLU).
  2. `out += identity; out = self.relu(out)` -> `self.skip_add.add_relu(...)`
     via FloatFunctional (eager mode can't quantize `+=`; math is identical).
  3. QuantStub after the FP32 stem (FP32 -> INT8 boundary).
  4. DeQuantStub before the prediction conv (INT8 -> FP32 boundary).
  5. positional grid + concat quantized (quant_pos + FloatFunctional.cat),
     required because Decision 9 keeps only conv1 + prediction in FP32, so the
     head's `project` conv is INT8 and its concat input must be INT8.

Every other line - module names, layer construction, the forward pass, variable
names, the weight-init block, map_to_uv, the commented-out 3D heads - is the
original verbatim, so ep71 weights load `strict` and the un-quantized fork is
bit-identical to FP32 (proven by 01_verify_fork.py: max diff 0).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat

from model.helper.resnet_helper import conv1x1, conv3x3
from model.detnet.detnet import get_pose_tile_torch

try:
    from torch.ao.nn.quantized import FloatFunctional
except ImportError:                        # older torch fallback
    from torch.nn.quantized import FloatFunctional
QuantStub = torch.ao.quantization.QuantStub
DeQuantStub = torch.ao.quantization.DeQuantStub


class QuantBottleneck(nn.Module):
    # Mirror of resnet_helper.Bottleneck. Only QUANT-tagged lines differ.
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(QuantBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu1 = nn.ReLU(inplace=True)   # QUANT: was one shared self.relu (used 3x)
        self.relu2 = nn.ReLU(inplace=True)   # QUANT: distinct ReLU per Conv-BN-ReLU
        self.skip_add = FloatFunctional()    # QUANT: replaces `out += identity`
        self.shortcut = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)                # QUANT: was self.relu

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)                # QUANT: was self.relu

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out = self.skip_add.add_relu(out, identity)   # QUANT: was `out += identity; out = self.relu(out)`

        return out


class QuantResNet(nn.Module):
    # Mirror of resnet_helper.ResNet. Only QUANT-tagged lines differ.

    def __init__(self, block=QuantBottleneck, layers=(2, 4, 6), num_classes=1000,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(QuantResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.block1 = self._make_layer(block, 64, layers[0], stride=1)
        self.pool = self.__sample(block, 64, 1, stride=2)

        self.block2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=True)
        self.block3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=True)

        self.squeeze = nn.Sequential(conv3x3(1024, 256), nn.BatchNorm2d(256), nn.ReLU())

        self.quant = QuantStub()             # QUANT: FP32 -> INT8 boundary (used after the FP32 stem)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, QuantBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, self.dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def __sample(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.quant(x)            # QUANT: FP32 -> INT8 (conv1 stem stays FP32)
        x = self.block1(x)
        x = self.pool(x)

        x = self.block2(x)

        x = self.block3(x)
        x = self.squeeze(x)
        return x                     # QUANT: stays INT8 (dequant happens in the head)

    def forward(self, x):
        return self._forward_impl(x)


class QuantNet2D(nn.Module):
    # Mirror of detnet.net_2d. Only the QUANT-tagged line differs.
    def __init__(self, input_features, output_features, stride, joints=21):
        super().__init__()
        self.project = nn.Sequential(conv3x3(input_features, output_features, stride), nn.BatchNorm2d(output_features),
                                     nn.ReLU())

        self.prediction = nn.Conv2d(output_features, joints, 1, 1, 0)
        self.dequant = DeQuantStub()         # QUANT: INT8 -> FP32 boundary (before the final conv)

    def forward(self, x):
        x = self.project(x)
        x = self.dequant(x)                  # QUANT: INT8 -> FP32 (project is INT8, prediction is FP32)
        x = self.prediction(x).sigmoid()
        return x


class DetNetQuant(nn.Module):
    # Mirror of detnet.detnet. Only QUANT-tagged lines differ.
    def __init__(self, stacks=1):
        super().__init__()
        self.resnet50 = QuantResNet(QuantBottleneck, (2, 4, 6))   # QUANT: quantizable resnet50()

        self.hmap_0 = QuantNet2D(258, 256, 1)                     # QUANT: quantizable net_2d(258,256,1)
        # Only doing 2D analysis — dmap/lmap 3D heads below are not needed.
        # self.dmap_0 = net_3d(279, 256, 1)
        # self.lmap_0 = net_3d(342, 256, 1)
        self.quant_pos = QuantStub()         # QUANT: positional grid FP32 -> INT8
        self.cat = FloatFunctional()         # QUANT: replaces torch.cat (INT8 concat)
        self.stacks = stacks

    def forward(self, x):
        # ── Feature Extractor (ResNet50) ─────────────────────────────────────
        features = self.resnet50(x)

        device = x.device
        pos_tile = get_pose_tile_torch(features.shape[0]).to(device)

        pos_tile = self.quant_pos(pos_tile)              # QUANT: positional grid -> INT8
        x = self.cat.cat([features, pos_tile], dim=1)    # QUANT: was torch.cat([features, pos_tile], dim=1)
        # ── End Feature Extractor ─────────────────────────────────────────────

        hmaps = []
        # Only doing 2D analysis — dmap/lmap accumulators not needed.
        # dmaps = []
        # lmaps = []

        for _ in range(self.stacks):
            # ── 2D Detector (Heat Maps H) ─────────────────────────────────────
            heat_map = self.hmap_0(x)
            hmaps.append(heat_map)
            # 2D-only: heat_map is no longer fed into the 3D heads, so this
            # concat is redundant.
            # x = torch.cat([x, heat_map], dim=1)
            # ── End 2D Detector ───────────────────────────────────────────────

            # ── Only doing 2D analysis — 3D detector heads below not needed ───
            # ── 3D Detector: Delta Maps D ─────────────────────────────────────
            # dmap = self.dmap_0(x)
            # dmaps.append(dmap)
            #
            # x = torch.cat([x, rearrange(dmap, 'b j l h w -> b (j l) h w')], dim=1)
            # ── End Delta Maps ────────────────────────────────────────────────

            # ── 3D Detector: Location Maps L ──────────────────────────────────
            # lmap = self.lmap_0(x)
            # lmaps.append(lmap)
            # ── End Location Maps ─────────────────────────────────────────────

        hmap = hmaps[-1]
        # hmap, dmap, lmap = hmaps[-1], dmaps[-1], lmaps[-1]

        # ── Joint Locations X (argmax + delta/xyz readout) ────────────────────
        uv, argmax = self.map_to_uv(hmap)

        # Only doing 2D analysis — delta/xyz readout from 3D heads not needed.
        # delta = self.dmap_to_delta(dmap, argmax)
        # xyz = self.lmap_to_xyz(lmap, argmax)
        # ── End Joint Locations ───────────────────────────────────────────────

        det_result = {
            "h_map": hmap,
            # "d_map": dmap,
            # "l_map": lmap,
            # "delta": delta,
            # "xyz": xyz,
            "uv": uv
        }

        return det_result

    @property
    def pos(self):
        return self.__pos_tile

    @staticmethod
    def map_to_uv(hmap):
        b, j, h, w = hmap.shape
        hmap = rearrange(hmap, 'b j h w -> b j (h w)')
        argmax = torch.argmax(hmap, -1, keepdim=True)
        u = argmax // w
        v = argmax % w
        uv = torch.cat([u, v], dim=-1)

        return uv, argmax                   #uv are the coordinates of each joint from the heatmap space which is 32x32 so we need to convert this back to the image size which is 128x128

    @staticmethod
    def dmap_to_delta(dmap, argmax):
        return DetNetQuant.lmap_to_xyz(dmap, argmax)

    @staticmethod
    def lmap_to_xyz(lmap, argmax):
        lmap = rearrange(lmap, 'b j l h w -> b j (h w) l')
        index = repeat(argmax, 'b j i -> b j i c', c=3)
        xyz = torch.gather(lmap, dim=2, index=index).squeeze(2)
        return xyz


# ── quant fusion list + loader (helpers, not part of the mirrored model) ───────

def fusion_list(model):
    """Conv-BN-ReLU / Conv-BN groups for fuse_modules. Everything between the
    first and last conv (Decision 9): backbone blocks, squeeze, head project.
    conv1 stem and the prediction conv stay FP32, unfused. Reviewed at Step 3.5."""
    groups = []
    for stage in ("block1", "pool", "block2", "block3"):
        for i, blk in enumerate(getattr(model.resnet50, stage)):
            p = f"resnet50.{stage}.{i}."
            groups += [[p + "conv1", p + "bn1", p + "relu1"],
                       [p + "conv2", p + "bn2", p + "relu2"],
                       [p + "conv3", p + "bn3"]]
            if isinstance(blk.shortcut, nn.Sequential):     # MaxPool shortcut skipped
                groups.append([p + "shortcut.0", p + "shortcut.1"])
    groups.append(["resnet50.squeeze.0", "resnet50.squeeze.1", "resnet50.squeeze.2"])
    groups.append(["hmap_0.project.0", "hmap_0.project.1", "hmap_0.project.2"])
    return groups


def load_detnet_quant(fp32_model=None, device="cpu"):
    """Build DetNetQuant and load ep71 weights (strict)."""
    mq = DetNetQuant()
    if fp32_model is None:
        from qcommon import load_fp32
        fp32_model = load_fp32(device)
    mq.load_state_dict(fp32_model.state_dict(), strict=True)
    return mq.to(device).eval()
