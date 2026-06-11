"""
Calibration data for INT8 PTQ - built to match the ORIGINAL DetNet input pipeline.

Each image is cropped to a hand-centered 128x128 exactly the way the rest of the
project does it:
    GT 2D keypoints -> center + scale (utils.handutils)
                    -> affine crop to 128x128
                    -> normalize (func.to_tensor; subtract 0.5)
This is the SAME function used to evaluate the FP32 baseline
(evaluate_detnet.training_crop_and_transform), so calibration sees the identical
input distribution as evaluation / live inference.

Annotation source: the surviving datasets/data/.cache/*.pkl files (the deleted
loaders' cached output) supply image paths + GT kp2ds only. Nothing under
datasets/ is written - read-only.

Calibration set (Decisions 13/14): 160 imgs each from CMU + RHD + GAN = 480,
batch 32 => 15 batches. CMU = hand143_panopticdb (14817) + hand_labels (1912).
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # before cv2/torch

import pickle
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from qcommon import REPO_ROOT
from evaluate_detnet import training_crop_and_transform

CACHE = os.path.join(REPO_ROOT, "datasets", "data", ".cache", "my-train")
CACHE_FILES = {
    "cmu_panoptic":   os.path.join(CACHE, "hand143_panopticdb", "train.pkl"),
    "cmu_handlabels": os.path.join(CACHE, "hand_labels", "train.pkl"),
    "rhd":            os.path.join(CACHE, "rhd", "train.pkl"),
    "gan":            os.path.join(CACHE, "GANeratedHands", "train.pkl"),
}
N_PER_DATASET = 160
SEED = 42
BATCH = 32

CALIB_TENSORS = os.path.join(REPO_ROOT, "quant", "calib_tensors.pt")
CALIB_INDICES = os.path.join(REPO_ROOT, "quant", "calibration_indices.json")


def _load_cache(name):
    with open(CACHE_FILES[name], "rb") as f:
        return pickle.load(f)


def load_calib_entries():
    """{source: (paths, kp2ds)} with CMU = panoptic + hand_labels concatenated."""
    cp, ch = _load_cache("cmu_panoptic"), _load_cache("cmu_handlabels")
    cmu_paths = list(cp["clr_paths"]) + list(ch["clr_paths"])
    cmu_kp = np.concatenate([cp["kp2ds"], ch["kp2ds"]], axis=0)
    rhd, gan = _load_cache("rhd"), _load_cache("gan")
    return {
        "cmu": (cmu_paths, cmu_kp),
        "rhd": (list(rhd["clr_paths"]), rhd["kp2ds"]),
        "gan": (list(gan["clr_paths"]), gan["kp2ds"]),
    }


def sample_indices(entries):
    """Deterministic per-dataset indices (seed 42), fixed order cmu -> rhd -> gan."""
    random.seed(SEED)
    return {src: random.sample(range(len(entries[src][0])), N_PER_DATASET)
            for src in ("cmu", "rhd", "gan")}


def crop_one(path, kp2d):
    """Original-pipeline crop: GT kp2d -> 128x128 normalized tensor (3,128,128)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    tensor, _, _ = training_crop_and_transform(img, kp2d, torch.device("cpu"))
    return tensor.squeeze(0)


def build_and_save():
    """Build all 480 calibration tensors, save tensors + indices. Returns the tensor."""
    import json
    entries = load_calib_entries()
    idx = sample_indices(entries)
    tensors, sources = [], []
    for src in ("cmu", "rhd", "gan"):
        paths, kps = entries[src]
        for i in idx[src]:
            tensors.append(crop_one(paths[i], kps[i]))
            sources.append(src)
    X = torch.stack(tensors)   # (480, 3, 128, 128)
    torch.save({"x": X, "sources": sources}, CALIB_TENSORS)
    with open(CALIB_INDICES, "w") as f:
        json.dump({
            "seed": SEED, "n_per_dataset": N_PER_DATASET, "batch": BATCH,
            "crop": "GT kp2d -> handutils center/scale -> 128x128 (original DetNet pipeline)",
            "cmu_note": "indices over concatenated [hand143_panopticdb(14817), hand_labels(1912)] = 16729",
            "cmu": idx["cmu"], "rhd": idx["rhd"], "gan": idx["gan"],
        }, f, indent=2)
    return X


def load_calib_loader(batch=BATCH):
    """DataLoader over the saved calibration tensors. shuffle=False (observer stats
    are order-independent) and num_workers=0 (Windows-safe, only 480 imgs)."""
    blob = torch.load(CALIB_TENSORS)
    return DataLoader(TensorDataset(blob["x"]), batch_size=batch,
                      shuffle=False, num_workers=0)
