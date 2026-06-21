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
    # S1: Gamma distributed for authentic SAR coherent speckle (shape=looks, scale=mean/looks)
    looks = 4.0
    stokes[0] = np.random.gamma(shape=looks, scale=(0.6 * strength) / looks, size=(H, W)).clip(0.1, 1.5)
    # S4: Highly negative for strong same-sense volume scattering (m-chi V component)
    stokes[3] = -stokes[0] * np.random.uniform(0.55, 0.70, (H, W))
    stokes[1] = np.random.normal(0.02, 0.005, (H, W))
    stokes[2] = np.random.normal(0.02, 0.005, (H, W))
    return stokes


def _rock_patch(size: tuple) -> np.ndarray:
    """Rock/regolith: high CPR from surface roughness but HIGH DOP."""
    H, W = size
    stokes = np.zeros((4, H, W), dtype=np.float32)
    looks = 4.0
    stokes[0] = np.random.gamma(shape=looks, scale=0.8 / looks, size=(H, W)).clip(0.2, 2.0)
    # S4 positive (opposite sense from double bounce)
    stokes[3] = stokes[0] * np.random.uniform(0.52, 0.62, (H, W))
    stokes[1] = np.random.normal(0.15, 0.02, (H, W))
    stokes[2] = np.random.normal(0.12, 0.02, (H, W))
    return stokes


def _regolith_patch(size: tuple) -> np.ndarray:
    """Plain regolith: low backscatter, surface scattering dominant."""
    H, W = size
    stokes = np.zeros((4, H, W), dtype=np.float32)
    looks = 4.0
    stokes[0] = np.random.gamma(shape=looks, scale=0.4 / looks, size=(H, W)).clip(0.05, 1.2)
    stokes[3] = stokes[0] * np.random.uniform(0.30, 0.48, (H, W))
    stokes[1] = np.random.normal(0.08, 0.01, (H, W))
    stokes[2] = np.random.normal(0.06, 0.01, (H, W))
    return stokes


def compute_m_chi_decomposition(stokes: np.ndarray):
    """
    Advanced m-chi Polarimetric Decomposition.
    Separates total power (S1) into Even (Double), Odd (Surface), and Diffuse (Volume) scattering.
    Volume scattering (V) > 0.4 is the SOTA indicator of volumetric water ice.
    """
    S1, S2, S3, S4 = stokes[0], stokes[1], stokes[2], stokes[3]
    eps = 1e-8
    m = np.sqrt(S2**2 + S3**2 + S4**2) / (S1 + eps)
    sin_2chi = -S4 / (S1 * m + eps)  # Convention: negative S4 -> positive chi -> volume
    
    # Stokes power inversion to m-chi bases
    V = S1 * (1 - m)                           # Diffuse / Volume scattering
    S = S1 * m * 0.5 * (1 + sin_2chi)          # Odd / Surface scattering
    D = S1 * m * 0.5 * (1 - sin_2chi)          # Even / Double bounce
    
    return V, S, D


def generate_dataset(
    n_samples: int = 2000,
    patch_size: int = 64,
    save_dir: str = "data",
    class_weights: tuple = (0.35, 0.35, 0.30),  # ice, rock, regolith
) -> dict:
    """
    Generate a labelled synthetic DFSAR dataset with DEM roughness.
    """
    os.makedirs(save_dir, exist_ok=True)
    size = (patch_size, patch_size)

    n_ice      = int(n_samples * class_weights[0])
    n_rock     = int(n_samples * class_weights[1])
    n_regolith = n_samples - n_ice - n_rock

    patches, roughnesses, labels, v_comps = [], [], [], []

    generators = [
        (_ice_patch,      n_ice,      1),  # label 1 for ice
        (_rock_patch,     n_rock,     2),  # label 2
        (_regolith_patch, n_regolith, 0),  # label 0
    ]

    for gen_fn, count, label in generators:
        for _ in range(count):
            stokes = gen_fn(size)
            V, S, D = compute_m_chi_decomposition(stokes)
            
            # Simulate corresponding DEM roughness (high roughness for rocks, low for ice/regolith)
            if label == 2:
                r = np.random.gamma(shape=3.0, scale=0.1, size=size).astype(np.float32)
            else:
                r = np.random.gamma(shape=1.5, scale=0.05, size=size).astype(np.float32)
                
            # Ice mask gated purely on SOTA Volume scattering (V > 0.4) and low roughness
            ice_mask = (V > 0.4) & (r < 0.2)
            final_label = 1 if (gen_fn == _ice_patch and ice_mask.mean() > 0.5) else label
            
            patches.append(stokes)
            roughnesses.append(r)
            labels.append(final_label)
            v_comps.append(float(V.mean()))

    stokes_arr = np.stack(patches).astype(np.float32)
    rough_arr  = np.stack(roughnesses).astype(np.float32)
    labels_arr = np.array(labels, dtype=np.int64)
    v_arr      = np.array(v_comps, dtype=np.float32)

    # Save
    np.save(os.path.join(save_dir, "stokes.npy"), stokes_arr)
    np.save(os.path.join(save_dir, "roughness.npy"), rough_arr)
    np.save(os.path.join(save_dir, "labels.npy"), labels_arr)
    np.save(os.path.join(save_dir, "v_vol.npy"), v_arr)

    meta = {
        "n_samples": n_samples,
        "patch_size": patch_size,
        "class_distribution": {
            "ice": int((labels_arr == 1).sum()),
            "rock": int((labels_arr == 2).sum()),
            "regolith": int((labels_arr == 0).sum()),
        },
        "physics": {
            "m_chi_volume_threshold": 0.4,
            "roughness_threshold": 0.2,
        },
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DataGen] Generated {n_samples} Multi-Modal samples -> {save_dir}")
    print(f"  Ice: {meta['class_distribution']['ice']}")
    print(f"  Rock: {meta['class_distribution']['rock']}")
    print(f"  Regolith: {meta['class_distribution']['regolith']}")
    return {"stokes": stokes_arr, "roughness": rough_arr, "labels": labels_arr, "v_vol": v_arr, "metadata": meta}


if __name__ == "__main__":
    generate_dataset(n_samples=2000, patch_size=64, save_dir="data/synthetic")
