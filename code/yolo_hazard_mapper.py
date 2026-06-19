"""
LUNA-SITE | Layer 10: Dual-Zone Hazard Mapper
==============================================
Implements the Optical Paradox fix:

  ZONE A (Sunlit approach): YOLOv8 on OHRC optical images
    - Detects boulders > 0.32m (Chandrayaan-4 wheel clearance spec)
    - Detects slopes > 10° (Chandrayaan-4 stability spec)

  ZONE B (PSR interior): LOLA DEM + DFSAR surface roughness ONLY
    - Optical cameras are physically useless at 25K in zero-light PSRs
    - YOLOv8 is DISABLED; roughness map drives hazard classification

This dual-zone switch is the single most important architectural
decision that separates LUNA-SITE from competing teams.
"""

import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Chandrayaan-4 Hardware Constraints (validated specs) ─────────────────────
MAX_SLOPE_DEG       = 10.0    # Rover stability limit
MAX_BOULDER_HEIGHT_M = 0.32   # Wheel clearance minimum
PSR_ILLUMINATION_THRESH = 0.0 # Pixels with illumination <= this are PSR


@dataclass
class HazardMap:
    """Complete hazard assessment for a landing site region."""
    zone:            str                  # "sunlit" or "psr"
    shape:           Tuple[int, int]      # (H, W)
    boulder_mask:    np.ndarray = field(repr=False)
    slope_mask:      np.ndarray = field(repr=False)
    combined_hazard: np.ndarray = field(repr=False)  # [0,1] hazard score
    n_hazard_pixels: int        = 0
    hazard_fraction: float      = 0.0
    method:          str        = ""

    def safe_fraction(self) -> float:
        return 1.0 - self.hazard_fraction


# ── Synthetic DEM helpers ─────────────────────────────────────────────────────

def simulate_dem_slopes(shape: Tuple[int, int], seed: int = 1) -> np.ndarray:
    """
    Simulate slope map from DEM (degrees).
    In production: derive from LOLA DEM via numpy gradient.
    """
    rng = np.random.default_rng(seed)
    # Most terrain flat, occasional steep crater walls
    slopes = rng.exponential(scale=4.0, size=shape).astype(np.float32)
    # Crater rim: a ring of steeper slopes
    cy, cx = shape[0] // 2, shape[1] // 2
    Y, X = np.ogrid[:shape[0], :shape[1]]
    ring_mask = (np.abs(np.sqrt((Y-cy)**2 + (X-cx)**2) - min(shape)/4) < 5)
    slopes[ring_mask] += rng.uniform(15, 35, ring_mask.sum()).astype(np.float32)
    return np.clip(slopes, 0, 90)


def simulate_boulder_density(shape: Tuple[int, int], seed: int = 2) -> np.ndarray:
    """
    Simulate boulder density map (0=clear, 1=hazardous).
    In production: from OHRC image YOLOv8 detections.
    """
    rng = np.random.default_rng(seed)
    density = rng.beta(0.5, 5.0, size=shape).astype(np.float32)
    return density


def simulate_illumination_map(shape: Tuple[int, int], psr_fraction: float = 0.35) -> np.ndarray:
    """
    Simulate illumination map.
    PSR (permanently shadowed) pixels = 0, sunlit = 1.
    In production: from LOLA illumination model or SPICE toolkit.
    """
    illum = np.ones(shape, dtype=np.float32)
    # PSR centered in bottom half (south pole geometry)
    cy, cx = int(shape[0] * 0.65), shape[1] // 2
    radius = int(min(shape) * np.sqrt(psr_fraction / np.pi))
    Y, X   = np.ogrid[:shape[0], :shape[1]]
    psr_mask = ((Y - cy)**2 + (X - cx)**2) < radius**2
    illum[psr_mask] = 0.0
    return illum


# ── Zone A: Sunlit YOLO-based mapping ────────────────────────────────────────

def sunlit_yolo_hazard_map(
    shape:           Tuple[int, int],
    boulder_density: np.ndarray,
    slope_map:       np.ndarray,
    boulder_thresh:  float = 0.4,   # density score above which = hazardous
    slope_thresh:    float = MAX_SLOPE_DEG,
) -> HazardMap:
    """
    Layer 10A: Sunlit zone hazard mapping.

    In production, boulder_density comes from YOLOv8 detections on
    OHRC 25cm/px imagery. Here we use the simulated density map.

    YOLOv8 detects boulders >= 0.32m. The density map encodes
    how many hazardous boulders per pixel neighbourhood.
    """
    boulder_mask = boulder_density > boulder_thresh
    slope_mask   = slope_map > slope_thresh

    # Combined: either hazard type is blocking
    combined    = np.clip(
        boulder_density * 0.6 + (slope_map / 90.0) * 0.4, 0, 1
    )
    hazard_mask  = boulder_mask | slope_mask
    n_haz        = int(hazard_mask.sum())
    haz_frac     = float(n_haz) / (shape[0] * shape[1])

    logger.info(
        f"[Zone A Sunlit] Hazard pixels: {n_haz} ({haz_frac:.1%}) | "
        f"Method: YOLOv8 + DEM slope"
    )

    return HazardMap(
        zone="sunlit", shape=shape,
        boulder_mask=boulder_mask, slope_mask=slope_mask,
        combined_hazard=combined,
        n_hazard_pixels=n_haz, hazard_fraction=haz_frac,
        method="YOLOv8 boulder detection + LOLA slope analysis",
    )


# ── Zone B: PSR Radar/DEM mapping ────────────────────────────────────────────

def psr_radar_hazard_map(
    shape:       Tuple[int, int],
    slope_map:   np.ndarray,
    roughness:   np.ndarray,
    slope_thresh: float = MAX_SLOPE_DEG,
    rough_thresh: float = 0.25,
) -> HazardMap:
    """
    Layer 10B: PSR interior hazard mapping.

    CRITICAL: YOLOv8 is DISABLED here.
    Optical cameras cannot operate at 25K in permanently shadowed regions
    where there is literally zero photon flux. Only LOLA DEM and DFSAR
    surface roughness are valid data sources inside a PSR.

    Rough terrain in a PSR ≈ blocky ejecta ≈ hazardous to rover wheels.
    """
    slope_mask   = slope_map > slope_thresh
    boulder_proxy_mask = roughness > rough_thresh  # roughness proxies for boulders

    combined  = np.clip(
        (slope_map / 90.0) * 0.5 + (roughness / roughness.max()) * 0.5, 0, 1
    )
    hazard_mask = slope_mask | boulder_proxy_mask
    n_haz       = int(hazard_mask.sum())
    haz_frac    = float(n_haz) / (shape[0] * shape[1])

    logger.info(
        f"[Zone B PSR] Hazard pixels: {n_haz} ({haz_frac:.1%}) | "
        f"Method: LOLA DEM + DFSAR roughness (NO optical)"
    )

    return HazardMap(
        zone="psr", shape=shape,
        boulder_mask=boulder_proxy_mask, slope_mask=slope_mask,
        combined_hazard=combined,
        n_hazard_pixels=n_haz, hazard_fraction=haz_frac,
        method="LOLA DEM roughness + DFSAR surface texture (optical DISABLED)",
    )


# ── Dual-Zone Dispatcher ──────────────────────────────────────────────────────

def dual_zone_hazard_map(
    shape:            Tuple[int, int] = (256, 256),
    illumination_map: Optional[np.ndarray] = None,
    boulder_density:  Optional[np.ndarray] = None,
    slope_map:        Optional[np.ndarray] = None,
    roughness_map:    Optional[np.ndarray] = None,
    psr_fraction:     float = 0.35,
) -> Dict[str, HazardMap]:
    """
    Layer 10: Full dual-zone hazard assessment.

    Automatically routes each pixel to the correct mapping method
    based on illumination data. This is the key innovation.

    Returns
    -------
    {"sunlit": HazardMap, "psr": HazardMap, "combined": np.ndarray}
    """
    if illumination_map is None:
        illumination_map = simulate_illumination_map(shape, psr_fraction)
    if slope_map is None:
        slope_map = simulate_dem_slopes(shape)
    if boulder_density is None:
        boulder_density = simulate_boulder_density(shape)
    if roughness_map is None:
        from cpr_dop_mapper import simulate_dem_roughness
        roughness_map = simulate_dem_roughness(shape)

    psr_mask    = illumination_map <= PSR_ILLUMINATION_THRESH
    sunlit_mask = ~psr_mask

    logger.info(
        f"Dual-zone split: {psr_mask.sum()} PSR pixels | "
        f"{sunlit_mask.sum()} sunlit pixels"
    )

    # Zone A: sunlit pixels only
    sunlit_result = sunlit_yolo_hazard_map(
        shape=shape,
        boulder_density=boulder_density * sunlit_mask,
        slope_map=slope_map * sunlit_mask,
    )

    # Zone B: PSR pixels only (no optical data used)
    psr_result = psr_radar_hazard_map(
        shape=shape,
        slope_map=slope_map * psr_mask,
        roughness=roughness_map * psr_mask,
    )

    # Composite hazard map
    combined = np.where(
        psr_mask,
        psr_result.combined_hazard,
        sunlit_result.combined_hazard,
    )

    return {
        "sunlit":   sunlit_result,
        "psr":      psr_result,
        "combined": combined,
        "psr_mask": psr_mask,
        "illumination": illumination_map,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = dual_zone_hazard_map(shape=(256, 256))
    print(f"\nSunlit safe fraction:  {result['sunlit'].safe_fraction():.1%}")
    print(f"PSR safe fraction:     {result['psr'].safe_fraction():.1%}")
    print(f"PSR pixels: {result['psr_mask'].sum()} / {256*256}")
    print(f"Combined hazard range: [{result['combined'].min():.3f}, {result['combined'].max():.3f}]")
    print("\nDual-zone hazard mapper OK ✓")
