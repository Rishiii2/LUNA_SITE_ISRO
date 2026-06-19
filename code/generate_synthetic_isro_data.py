"""
LUNA-SITE | Layer 0: Synthetic ISRO DFSAR Data Generator
=========================================================
Generates realistic 4-channel Stokes vector patches mimicking
Chandrayaan-2 DFSAR polarimetric radar data for the lunar south pole.

Physics basis:
  S1 = total power
  S2 = linear polarization difference (H-V)
  S3 = linear polarization at 45 degrees
  S4 = circular polarization (RCP - LCP)

  CPR = (S1 - S4) / (S1 + S4)   [>1.0 => volumetric ice scattering]
  DOP = sqrt(S2^2+S3^2+S4^2)/S1 [<0.13 => depolarized = ice candidate]
"""

import numpy as np
import os
import json

SEED = 42
np.random.seed(SEED)


def _ice_patch(size: tuple, strength: float = 1.0) -> np.ndarray:
    """Create a 4-channel Stokes patch with ice-like signatures."""
    H, W = size
    stokes = np.zeros((4, H, W), dtype=np.float32)
    # S1: moderate total power
    stokes[0] = np.random.normal(0.6 * strength, 0.05, (H, W)).clip(0.1, 1.5)
    # S4: Must be negative for CPR (SC/OC) to exceed 1.0 (indicating strong same-sense SC power)
    stokes[3] = -stokes[0] * np.random.uniform(0.55, 0.70, (H, W))
    # S2, S3: small (depolarised => low DOP)
    stokes[1] = np.random.normal(0.02, 0.005, (H, W))
    stokes[2] = np.random.normal(0.02, 0.005, (H, W))
    return stokes


def _rock_patch(size: tuple) -> np.ndarray:
    """Rock/regolith: high CPR from surface roughness but HIGH DOP."""
    H, W = size
    stokes = np.zeros((4, H, W), dtype=np.float32)
    stokes[0] = np.random.normal(0.8, 0.08, (H, W)).clip(0.2, 2.0)
    stokes[3] = stokes[0] * np.random.uniform(0.52, 0.62, (H, W))
    # HIGH S2/S3 => DOP > 0.13 (distinguishes rock from ice)
    stokes[1] = np.random.normal(0.15, 0.02, (H, W))
    stokes[2] = np.random.normal(0.12, 0.02, (H, W))
    return stokes


def _regolith_patch(size: tuple) -> np.ndarray:
    """Plain regolith: CPR < 1, DOP > 0.13."""
    H, W = size
    stokes = np.zeros((4, H, W), dtype=np.float32)
    stokes[0] = np.random.normal(0.4, 0.06, (H, W)).clip(0.05, 1.2)
    stokes[3] = stokes[0] * np.random.uniform(0.30, 0.48, (H, W))
    stokes[1] = np.random.normal(0.08, 0.01, (H, W))
    stokes[2] = np.random.normal(0.06, 0.01, (H, W))
    return stokes


def compute_cpr_dop(stokes: np.ndarray):
    """Compute CPR and DOP from 4-channel Stokes array."""
    S1, S2, S3, S4 = stokes[0], stokes[1], stokes[2], stokes[3]
    eps = 1e-8
    SC = (S1 - S4) / 2.0
    OC = (S1 + S4) / 2.0
    cpr = SC / (OC + eps)
    dop = np.sqrt(S2**2 + S3**2 + S4**2) / (S1 + eps)
    return cpr, dop


def generate_dataset(
    n_samples: int = 2000,
    patch_size: int = 64,
    save_dir: str = "data",
    class_weights: tuple = (0.35, 0.35, 0.30),  # ice, rock, regolith
) -> dict:
    """
    Generate a labelled synthetic DFSAR dataset.

    Returns
    -------
    dict with keys: stokes, labels, cpr, dop, metadata
    """
    os.makedirs(save_dir, exist_ok=True)
    size = (patch_size, patch_size)

    n_ice      = int(n_samples * class_weights[0])
    n_rock     = int(n_samples * class_weights[1])
    n_regolith = n_samples - n_ice - n_rock

    patches, labels, cprs, dops = [], [], [], []

    generators = [
        (_ice_patch,      n_ice,      1),  # label 1 for ice
        (_rock_patch,     n_rock,     2),  # label 2
        (_regolith_patch, n_regolith, 0),  # label 0
    ]

    for gen_fn, count, label in generators:
        for _ in range(count):
            stokes = gen_fn(size) if gen_fn != _ice_patch else gen_fn(size)
            cpr, dop = compute_cpr_dop(stokes)
            # Apply physics-based weak labelling override
            ice_mask = (cpr > 1.0) & (dop < 0.13)
            final_label = 1 if (gen_fn == _ice_patch and ice_mask.mean() > 0.5) else label
            patches.append(stokes)
            labels.append(final_label)
            cprs.append(float(cpr.mean()))
            dops.append(float(dop.mean()))

    stokes_arr = np.stack(patches).astype(np.float32)
    labels_arr = np.array(labels, dtype=np.int64)
    cpr_arr    = np.array(cprs, dtype=np.float32)
    dop_arr    = np.array(dops, dtype=np.float32)

    # Save
    np.save(os.path.join(save_dir, "stokes.npy"), stokes_arr)
    np.save(os.path.join(save_dir, "labels.npy"), labels_arr)
    np.save(os.path.join(save_dir, "cpr.npy"), cpr_arr)
    np.save(os.path.join(save_dir, "dop.npy"), dop_arr)

    meta = {
        "n_samples": n_samples,
        "patch_size": patch_size,
        "class_distribution": {
            "ice": int((labels_arr == 1).sum()),
            "rock": int((labels_arr == 2).sum()),
            "regolith": int((labels_arr == 0).sum()),
        },
        "physics": {
            "cpr_ice_threshold": 1.0,
            "dop_ice_threshold": 0.13,
            "l_band_wavelength_cm": 24.0,
            "s_band_wavelength_cm": 10.0,
        },
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DataGen] Generated {n_samples} samples → {save_dir}")
    print(f"  Ice: {meta['class_distribution']['ice']}")
    print(f"  Rock: {meta['class_distribution']['rock']}")
    print(f"  Regolith: {meta['class_distribution']['regolith']}")
    return {"stokes": stokes_arr, "labels": labels_arr, "cpr": cpr_arr,
            "dop": dop_arr, "metadata": meta}


if __name__ == "__main__":
    generate_dataset(n_samples=2000, patch_size=64, save_dir="data/synthetic")
