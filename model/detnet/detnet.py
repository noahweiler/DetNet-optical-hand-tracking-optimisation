'''
detnet  based on PyTorch
this is modified from https://github.com/lingtengqiu/Minimal-Hand
'''
import sys

import torch

sys.path.append("./")
from torch import nn
from einops import rearrange, repeat
from model.helper import resnet50, conv3x3
import numpy as np


# my modification
def get_pose_tile_torch(N):
    pos_tile = np.expand_dims(
        np.stack(
            [
                np.tile(np.linspace(-1, 1, 32).reshape([1, 32]), [32, 1]),
                np.tile(np.linspace(-1, 1, 32).reshape([32, 1]), [1, 32])
            ], -1
        ), 0
    )
    pos_tile = np.tile(pos_tile, (N, 1, 1, 1))
    retv = torch.from_numpy(pos_tile).float()
    return rearrange(retv, 'b h w c -> b c h w')


class net_2d(nn.Module):
    def __init__(self, input_features, output_features, stride, joints=21):
        super().__init__()
        self.project = nn.Sequential(conv3x3(input_features, output_features, stride), nn.BatchNorm2d(output_features),
                                     nn.ReLU())

        self.prediction = nn.Conv2d(output_features, joints, 1, 1, 0)

    def forward(self, x):
        x = self.project(x)
        x = self.prediction(x).sigmoid()
        return x


class net_3d(nn.Module):
    def __init__(self, input_features, output_features, stride, joints=21, need_norm=False):
        super().__init__()
        self.need_norm = need_norm
        self.project = nn.Sequential(conv3x3(input_features, output_features, stride), nn.BatchNorm2d(output_features),
                                     nn.ReLU())
        self.prediction = nn.Conv2d(output_features, joints * 3, 1, 1, 0)

    def forward(self, x):
        x = self.prediction(self.project(x))

        dmap = rearrange(x, 'b (j l) h w -> b j l h w', l=3)

        return dmap


class detnet(nn.Module):
    def __init__(self, stacks=1):
        super().__init__()
        self.resnet50 = resnet50()

        self.hmap_0 = net_2d(258, 256, 1)
        # Only doing 2D analysis — dmap/lmap 3D heads below are not needed.
        # self.dmap_0 = net_3d(279, 256, 1)
        # self.lmap_0 = net_3d(342, 256, 1)
        self.stacks = stacks

    def forward(self, x):
        # ── Feature Extractor (ResNet50) ─────────────────────────────────────
        features = self.resnet50(x)

        device = x.device
        pos_tile = get_pose_tile_torch(features.shape[0]).to(device)

        x = torch.cat([features, pos_tile], dim=1)
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
        return detnet.lmap_to_xyz(dmap, argmax)

    @staticmethod
    def lmap_to_xyz(lmap, argmax):
        lmap = rearrange(lmap, 'b j l h w -> b j (h w) l')
        index = repeat(argmax, 'b j i -> b j i c', c=3)
        xyz = torch.gather(lmap, dim=2, index=index).squeeze(2)
        return xyz


if __name__ == '__main__':
    mydet = detnet()
    img_crop = torch.randn(10, 3, 128, 128)
    res = mydet(img_crop)

    hmap = res["h_map"]
    # Only doing 2D analysis — d_map/l_map/delta/xyz are not produced.
    # dmap = res["d_map"]
    # lmap = res["l_map"]
    # delta = res["delta"]
    # xyz = res["xyz"]
    uv = res["uv"]

    print("hmap.shape=", hmap.shape)
    # print("dmap.shape=", dmap.shape)
    # print("lmap.shape=", lmap.shape)
    # print("delta.shape=", delta.shape)
    # print("xyz.shape=", xyz.shape)
    print("uv.shape=", uv.shape)
