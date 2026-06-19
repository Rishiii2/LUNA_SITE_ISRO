"""
LUNA-SITE | Layers 14-16: Multi-Objective Landing Site Optimizer
=================================================================
Implements NSGA-II (Non-dominated Sorting Genetic Algorithm II) to find
the Pareto front of optimal Chandrayaan-4 landing sites.

Objective functions (all to MINIMIZE):
  f1 = -ice_volume_score      (maximize ice)
  f2 = terrain_hazard_score   (minimize hazard)
  f3 = -solar_illumination    (maximize solar energy for rover)

Outputs the Pareto front as a set of non-dominated candidate sites,
each with a unified Cost/Risk Metric for mission controllers.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ── Candidate site ────────────────────────────────────────────────────────────

@dataclass
class LandingSite:
    """Represents a candidate Chandrayaan-4 landing site."""
    site_id:         int
    x_px:            float          # pixel x in south polar stereographic
    y_px:            float          # pixel y
    ice_volume_m3:   float          # estimated extractable ice volume
    terrain_hazard:  float          # 0=safe, 1=impassable
    solar_hours:     float          # avg illumination hours/day
    earth_visibility: float         # fraction of time with direct earth link
    slope_deg:       float          # average slope
    depth_prob:      float          # P(ice within drill limit)
    pareto_rank:     int = 0
    crowding_dist:   float = 0.0
    cost_risk_metric: float = 0.0

    def satisfies_constraints(self) -> bool:
        """Hard Chandrayaan-4 hardware constraints."""
        return (
            self.slope_deg      <= 10.0 and   # stability
            self.terrain_hazard <  0.6         # safe traversal
        )


# ── Synthetic site generation ─────────────────────────────────────────────────

def generate_candidate_sites(
    n_sites:   int  = 200,
    seed:      int  = 42,
    grid_size: int  = 512,
) -> List[LandingSite]:
    """Generate synthetic Chandrayaan-4 candidate landing sites."""
    rng = np.random.default_rng(seed)
    sites = []
    for i in range(n_sites):
        x = rng.uniform(50, grid_size - 50)
        y = rng.uniform(50, grid_size - 50)

        # Simulate geographic correlations:
        # Sites near PSR centre (y > 300) have more ice but less sun
        near_psr  = (y > 300)
        ice_base  = rng.exponential(0.5) + (2.0 if near_psr else 0.1)
        solar     = rng.uniform(0.5, 4.0) if not near_psr else rng.uniform(0.1, 1.5)
        hazard    = rng.beta(1.5, 4.0 if not near_psr else 2.5)
        slope     = rng.exponential(3.0) + (0 if not near_psr else 1)
        earth_vis = rng.uniform(0.3, 0.8) if not near_psr else rng.uniform(0.05, 0.4)
        depth_p   = rng.beta(3, 2) if near_psr else rng.beta(1, 3)

        sites.append(LandingSite(
            site_id=i,
            x_px=float(x), y_px=float(y),
            ice_volume_m3=float(np.clip(ice_base * 50, 1, 500)),
            terrain_hazard=float(hazard),
            solar_hours=float(solar),
            earth_visibility=float(earth_vis),
            slope_deg=float(np.clip(slope, 0, 45)),
            depth_prob=float(depth_p),
        ))
    return sites


# ── NSGA-II ──────────────────────────────────────────────────────────────────

def objective_vector(site: LandingSite) -> np.ndarray:
    """
    Convert site to minimisation objective vector.
    All objectives are to be MINIMISED.
    """
    f1 = -site.ice_volume_m3 / 500.0            # neg ice (maximise ice)
    f2 = site.terrain_hazard                     # minimise hazard
    f3 = -site.solar_hours / 6.0                # neg solar (maximise sun)
    return np.array([f1, f2, f3], dtype=float)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a dominates b (a <= b in all, a < b in at least one)."""
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(sites: List[LandingSite]) -> List[List[int]]:
    """
    NSGA-II Fast Non-Dominated Sort.
    Returns list of fronts, each front is list of site indices.
    """
    n = len(sites)
    objectives = [objective_vector(s) for s in sites]
    domination_count = [0] * n
    dominated_set    = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                dominated_set[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)
            sites[i].pareto_rank = 1

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
                    sites[j].pareto_rank = k + 2
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(sites: List[LandingSite], front: List[int]):
    """Assign crowding distance within a Pareto front."""
    if len(front) <= 2:
        for i in front:
            sites[i].crowding_dist = float("inf")
        return

    n_obj = 3
    for obj_idx in range(n_obj):
        sorted_front = sorted(front, key=lambda i: objective_vector(sites[i])[obj_idx])
        sites[sorted_front[0]].crowding_dist  = float("inf")
        sites[sorted_front[-1]].crowding_dist = float("inf")
        obj_range = (
            objective_vector(sites[sorted_front[-1]])[obj_idx]
            - objective_vector(sites[sorted_front[0]])[obj_idx]
            + 1e-10
        )
        for k in range(1, len(sorted_front) - 1):
            diff = (
                objective_vector(sites[sorted_front[k+1]])[obj_idx]
                - objective_vector(sites[sorted_front[k-1]])[obj_idx]
            )
            sites[sorted_front[k]].crowding_dist += diff / obj_range


def compute_cost_risk_metric(site: LandingSite) -> float:
    """
    Unified Cost/Risk Metric for mission controllers.
    Lower = better candidate.
    Weights: 40% ice, 35% hazard, 15% solar, 10% earth link.
    """
    ice_score    = 1.0 - (site.ice_volume_m3 / 500.0)   # lower = more ice
    hazard_score = site.terrain_hazard
    solar_score  = 1.0 - (site.solar_hours / 6.0)
    earth_score  = 1.0 - site.earth_visibility
    return (
        0.40 * ice_score +
        0.35 * hazard_score +
        0.15 * solar_score +
        0.10 * earth_score
    )


def run_nsga2(
    sites:      List[LandingSite],
    max_rank:   int = 1,   # Return only Pareto rank 1 by default
) -> Tuple[List[LandingSite], List[LandingSite]]:
    """
    Run NSGA-II and return (pareto_front, all_ranked_sites).

    Parameters
    ----------
    sites    : list of candidate LandingSites
    max_rank : sites with rank <= max_rank are returned as Pareto front

    Returns
    -------
    pareto_front : non-dominated sites
    all_sites    : all sites with rank and crowding assigned
    """
    # Filter to constraint-satisfying sites only
    feasible = [s for s in sites if s.satisfies_constraints()]
    logger.info(f"Feasible sites (constraint-satisfying): {len(feasible)}/{len(sites)}")

    if not feasible:
        logger.warning("No feasible sites found! Returning all sites without filtering.")
        feasible = sites

    fronts = fast_non_dominated_sort(feasible)
    for front in fronts:
        crowding_distance(feasible, front)

    for s in feasible:
        s.cost_risk_metric = compute_cost_risk_metric(s)

    pareto_front = [feasible[i] for i in fronts[0] if fronts]
    pareto_front.sort(key=lambda s: s.cost_risk_metric)

    logger.info(f"Pareto front: {len(pareto_front)} sites (rank 1)")
    for i, s in enumerate(pareto_front[:5]):
        logger.info(
            f"  Top {i+1}: Site#{s.site_id} | "
            f"Ice={s.ice_volume_m3:.0f}m³ | Hazard={s.terrain_hazard:.2f} | "
            f"Solar={s.solar_hours:.1f}h | CRM={s.cost_risk_metric:.3f}"
        )

    return pareto_front, feasible


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sites = generate_candidate_sites(n_sites=200)
    pareto, all_feasible = run_nsga2(sites)

    print(f"\n=== NSGA-II Results ===")
    print(f"Total candidates : {len(sites)}")
    print(f"Feasible         : {len(all_feasible)}")
    print(f"Pareto front     : {len(pareto)}")
    print(f"\nTop 3 mission sites:")
    for i, s in enumerate(pareto[:3], 1):
        print(
            f"  #{i} Site {s.site_id}: "
            f"Ice={s.ice_volume_m3:.0f}m³, "
            f"Hazard={s.terrain_hazard:.2f}, "
            f"Solar={s.solar_hours:.1f}h/day, "
            f"P(drill)={s.depth_prob:.1%}, "
            f"CRM={s.cost_risk_metric:.3f}"
        )
    print("\nNSGA-II optimizer OK ✓")
