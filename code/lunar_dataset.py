"""
LUNA-SITE | PyTorch DataLoader for DFSAR Stokes Tensors
=========================================================
Implements weak-supervised labeling: pixels are labelled 'Ice'
if they satisfy the physics gate (CPR > 1.0 AND DOP < 0.13),
so no human-annotated ground truth is required.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from generate_synthetic_isro_data import generate_dataset


class LunarDFSARDataset(Dataset):
    """
    PyTorch Dataset for 4-channel DFSAR Stokes patches.

    Weak-supervised labeling override: even if a patch is generated
    as 'regolith', if its CPR/DOP values satisfy the ice physics gate,
    the label is promoted to Ice (class 1). This removes the need for
    hand-labeled ground truth — critical for a real ISRO dataset.
    """

    def __init__(
        self,
        data_dir:    str   = "data",
        augment:     bool  = True,
        cpr_thresh:  float = 1.0,
        dop_thresh:  float = 0.13,
    ):
        self.augment    = augment
        self.cpr_thresh = cpr_thresh
        self.dop_thresh = dop_thresh

        # Load
        self.stokes = np.load(os.path.join(data_dir, "stokes.npy"))
        self.labels = np.load(os.path.join(data_dir, "labels.npy"))
        self.cpr    = np.load(os.path.join(data_dir, "cpr.npy"))
        self.dop    = np.load(os.path.join(data_dir, "dop.npy"))

        # Weak labeling override
        ice_gate = (self.cpr > self.cpr_thresh) & (self.dop < self.dop_thresh)
        self.labels = np.where(ice_gate, 1, self.labels).astype(np.int64)

        # Normalize each channel to [0, 1]
        for c in range(4):
            ch = self.stokes[:, c]
            vmin, vmax = ch.min(), ch.max()
            if vmax > vmin:
                self.stokes[:, c] = (ch - vmin) / (vmax - vmin)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        stokes = self.stokes[idx].copy()  # (4, H, W)
        label  = int(self.labels[idx])
        cpr    = float(self.cpr[idx])
        dop    = float(self.dop[idx])

        if self.augment:
            # Random horizontal/vertical flip
            if np.random.rand() > 0.5:
                stokes = stokes[:, :, ::-1].copy()
            if np.random.rand() > 0.5:
                stokes = stokes[:, ::-1, :].copy()
            # Random 90-degree rotation
            k = np.random.randint(0, 4)
            stokes = np.rot90(stokes, k=k, axes=(1, 2)).copy()
            # Small Gaussian noise
            stokes += np.random.normal(0, 0.005, stokes.shape).astype(np.float32)
            stokes = np.clip(stokes, 0.0, 1.0)

        return {
            "stokes": torch.tensor(stokes, dtype=torch.float32),
            "label":  torch.tensor(label,  dtype=torch.long),
            "cpr":    torch.tensor(cpr,    dtype=torch.float32),
            "dop":    torch.tensor(dop,    dtype=torch.float32),
        }


def generate_and_load_dataset(
    data_dir: str = "data",
    n_samples: int = 2000,
    force_regen: bool = False,
) -> LunarDFSARDataset:
    """Generate synthetic data if not present, then return Dataset."""
    stokes_path = os.path.join(data_dir, "stokes.npy")
    if not os.path.exists(stokes_path) or force_regen:
        generate_dataset(n_samples=n_samples, save_dir=data_dir)
    return LunarDFSARDataset(data_dir=data_dir)


if __name__ == "__main__":
    ds = generate_and_load_dataset()
    sample = ds[0]
    print(f"Dataset size: {len(ds)}")
    print(f"Stokes shape: {sample['stokes'].shape}")
    print(f"Label: {sample['label'].item()} | CPR: {sample['cpr']:.3f} | DOP: {sample['dop']:.3f}")
    classes, counts = np.unique(ds.labels, return_counts=True)
    for c, n in zip(classes, counts):
        names = {0: "Regolith", 1: "Ice", 2: "Rock"}
        print(f"  {names[c]}: {n} ({100*n/len(ds):.1f}%)")
