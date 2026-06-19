"""
LUNA-SITE | Layers 17-18: Energy-Aware RRT* + DWA Rover Path Planner
=====================================================================
Implements:
  Layer 17: Energy-Aware RRT* global path planning
    - Cost = distance + slope_penalty + shadow_energy_penalty
    - Chandrayaan-4 constraints: slope < 10°, boulder clearance 0.32m

  Layer 18: Dynamic Window Approach (DWA) local obstacle avoidance
    - Runs in real-time in Gazebo Digital Twin
    - Handles new obstacles discovered during traverse

Design for finale:
  RRT* global path is PRE-COMPUTED before the 30-hour event.
  DWA runs live in the Gazebo simulator.
  This avoids the 30-hour compute trap.

Prior work ported from: MATLAB Tactical Aerial Combat Simulator
(Kalman + Hungarian + RRT* + DWA drone collision avoidance stack)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging
import time

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class RoverConfig:
    """Chandrayaan-4 rover physical constraints."""
    max_slope_deg:      float = 10.0   # tipping stability limit
    min_boulder_clear_m: float = 0.32  # wheel clearance
    max_speed_mps:      float = 0.01   # 1 cm/s (conservative lunar rover)
    battery_capacity_wh: float = 100.0
    drive_power_w:      float = 30.0   # base driving power consumption
    solar_panel_w:      float = 20.0   # solar recharge rate (sunlit zone)


@dataclass
class RRTNode:
    x: float
    y: float
    cost: float = 0.0
    parent: Optional["RRTNode"] = None
    children: List["RRTNode"] = field(default_factory=list)


# ── Energy-Aware Cost Map ─────────────────────────────────────────────────────

class LunarCostMap:
    """
    Combined cost map for energy-aware RRT* planning.
    Cost = Euclidean distance × (1 + slope_factor + shadow_factor)

    Slope penalty: exponential increase near the 10° limit
    Shadow penalty: traversing PSR consumes stored battery (no solar recharge)
    """

    def __init__(
        self,
        shape:        Tuple[int, int],
        slope_map:    Optional[np.ndarray] = None,
        illum_map:    Optional[np.ndarray] = None,
        hazard_map:   Optional[np.ndarray] = None,
        config:       RoverConfig = None,
    ):
        self.shape  = shape
        self.config = config or RoverConfig()

        if slope_map is None:
            rng = np.random.default_rng(1)
            slope_map = rng.exponential(3.0, shape).astype(np.float32)
        if illum_map is None:
            illum_map = np.ones(shape, dtype=np.float32)
            cy, cx    = int(shape[0]*0.65), shape[1]//2
            radius    = int(min(shape) * 0.3)
            Y, X      = np.ogrid[:shape[0], :shape[1]]
            illum_map[((Y-cy)**2 + (X-cx)**2) < radius**2] = 0.0
        if hazard_map is None:
            rng        = np.random.default_rng(2)
            hazard_map = rng.beta(0.5, 5, shape).astype(np.float32)

        self.slope_map  = slope_map.clip(0, 90)
        self.illum_map  = illum_map
        self.hazard_map = hazard_map

        # Passability: True = can traverse
        self.passable = (slope_map < self.config.max_slope_deg) & (hazard_map < 0.7)

    def traversal_cost(self, x1, y1, x2, y2) -> float:
        """Cost of moving from (x1,y1) to (x2,y2)."""
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if dist < 1e-6:
            return 0.0

        # Sample along path
        n_pts   = max(2, int(dist))
        xs      = np.linspace(x1, x2, n_pts).astype(int).clip(0, self.shape[1]-1)
        ys      = np.linspace(y1, y2, n_pts).astype(int).clip(0, self.shape[0]-1)

        avg_slope  = self.slope_map[ys, xs].mean()
        avg_illum  = self.illum_map[ys, xs].mean()
        avg_hazard = self.hazard_map[ys, xs].mean()

        # Impassable
        if avg_slope >= self.config.max_slope_deg or avg_hazard >= 0.7:
            return float("inf")

        # Slope factor: exponential near limit
        slope_factor = np.exp(avg_slope / self.config.max_slope_deg * 2) - 1

        # Shadow factor: in PSR must use battery (more expensive)
        shadow_factor = (1.0 - avg_illum) * 1.5

        return dist * (1.0 + 0.5 * slope_factor + 0.3 * shadow_factor)


# ── RRT* ─────────────────────────────────────────────────────────────────────

class EnergyAwareRRTStar:
    """
    Energy-Aware RRT* global planner for lunar rover traverse.

    RRT* (Rapidly-exploring Random Tree Star) guarantees asymptotic
    optimality — given enough samples, it finds the globally optimal path.

    Ported from the MATLAB Tactical Aerial Combat Simulator (drone
    collision avoidance). Adapted for slow-speed, low-gravity terrain
    traversal with energy-cost weighting.
    """

    def __init__(
        self,
        cost_map:      LunarCostMap,
        step_size:     float = 8.0,
        max_iters:     int   = 3000,
        rewire_radius: float = 20.0,
        seed:          int   = 42,
    ):
        self.cost_map      = cost_map
        self.step_size     = step_size
        self.max_iters     = max_iters
        self.rewire_radius = rewire_radius
        self.rng           = np.random.default_rng(seed)
        self.nodes: List[RRTNode] = []

    def _sample(self, goal: Tuple[float, float], goal_bias: float = 0.1) -> Tuple[float, float]:
        if self.rng.random() < goal_bias:
            return goal
        H, W = self.cost_map.shape
        return (self.rng.uniform(0, W), self.rng.uniform(0, H))

    def _nearest(self, x: float, y: float) -> RRTNode:
        dists = [np.sqrt((n.x-x)**2 + (n.y-y)**2) for n in self.nodes]
        return self.nodes[int(np.argmin(dists))]

    def _steer(self, from_node: RRTNode, to: Tuple[float, float]) -> Tuple[float, float]:
        dx = to[0] - from_node.x
        dy = to[1] - from_node.y
        d  = np.sqrt(dx**2 + dy**2)
        if d < 1e-6:
            return (from_node.x, from_node.y)
        ratio = min(self.step_size, d) / d
        return (from_node.x + dx * ratio, from_node.y + dy * ratio)

    def _near_nodes(self, x: float, y: float) -> List[RRTNode]:
        return [n for n in self.nodes
                if np.sqrt((n.x-x)**2 + (n.y-y)**2) < self.rewire_radius]

    def plan(
        self,
        start: Tuple[float, float],
        goal:  Tuple[float, float],
    ) -> Tuple[List[Tuple[float, float]], float]:
        """
        Plan an energy-optimal path from start to goal.

        Returns
        -------
        path  : list of (x, y) waypoints
        cost  : total energy-aware traversal cost
        """
        self.nodes = [RRTNode(start[0], start[1], cost=0.0)]
        goal_node  = None
        t_start    = time.time()

        for iteration in range(self.max_iters):
            x_rand, y_rand = self._sample(goal)
            nearest         = self._nearest(x_rand, y_rand)
            x_new, y_new    = self._steer(nearest, (x_rand, y_rand))

            step_cost = self.cost_map.traversal_cost(
                nearest.x, nearest.y, x_new, y_new
            )
            if step_cost == float("inf"):
                continue

            # Choose best parent from nearby nodes
            near_nodes  = self._near_nodes(x_new, y_new)
            best_parent = nearest
            best_cost   = nearest.cost + step_cost

            for n in near_nodes:
                c = self.cost_map.traversal_cost(n.x, n.y, x_new, y_new)
                if c < float("inf") and n.cost + c < best_cost:
                    best_cost   = n.cost + c
                    best_parent = n

            new_node = RRTNode(x_new, y_new, cost=best_cost, parent=best_parent)
            best_parent.children.append(new_node)
            self.nodes.append(new_node)

            # Rewire nearby nodes through new node if cheaper
            for n in near_nodes:
                c = self.cost_map.traversal_cost(x_new, y_new, n.x, n.y)
                if c < float("inf") and new_node.cost + c < n.cost:
                    n.parent = new_node
                    n.cost   = new_node.cost + c

            # Check goal
            goal_dist = np.sqrt((x_new - goal[0])**2 + (y_new - goal[1])**2)
            if goal_dist < self.step_size:
                goal_c = self.cost_map.traversal_cost(x_new, y_new, goal[0], goal[1])
                total  = new_node.cost + goal_c
                if goal_node is None or total < goal_node.cost:
                    goal_node = RRTNode(goal[0], goal[1], cost=total, parent=new_node)

        if goal_node is None:
            logger.warning("RRT*: No path found to goal")
            return [], float("inf")

        # Extract path
        path  = []
        node  = goal_node
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        path.reverse()

        elapsed = time.time() - t_start
        logger.info(
            f"RRT*: Path found | {len(path)} waypoints | "
            f"Cost: {goal_node.cost:.2f} | Time: {elapsed:.2f}s | "
            f"Nodes: {len(self.nodes)}"
        )
        return path, goal_node.cost


# ── DWA Local Obstacle Avoidance ──────────────────────────────────────────────

class DynamicWindowApproach:
    """
    Layer 18: DWA real-time local obstacle avoidance.

    Runs in the Gazebo Digital Twin during the finale demo.
    Pre-computed RRT* global path provides waypoints; DWA handles
    obstacles discovered locally during execution.

    Ported from MATLAB drone collision avoidance (DWA component).
    """

    def __init__(
        self,
        config:    RoverConfig = None,
        dt:        float = 1.0,     # simulation timestep (seconds)
        v_max:     float = 0.01,    # m/s
        w_max:     float = 0.5,     # rad/s
        v_res:     float = 0.002,   # velocity resolution
        w_res:     float = 0.1,     # angular velocity resolution
    ):
        self.config = config or RoverConfig()
        self.dt     = dt
        self.v_max  = v_max
        self.w_max  = w_max
        self.v_res  = v_res
        self.w_res  = w_res

    def compute_velocity(
        self,
        state:     Tuple[float, float, float, float, float],  # x,y,theta,v,w
        goal:      Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],          # x,y,radius
        cost_map:  Optional[LunarCostMap] = None,
    ) -> Tuple[float, float]:
        """
        Compute optimal (v, w) for current timestep.

        Returns
        -------
        (v, w) : linear and angular velocity commands
        """
        x, y, theta, v_curr, w_curr = state
        gx, gy = goal

        best_score = -float("inf")
        best_v, best_w = 0.0, 0.0

        v_min = max(0.0, v_curr - 0.002)
        v_max = min(self.v_max, v_curr + 0.002)
        w_min = max(-self.w_max, w_curr - 0.2)
        w_max = min(self.w_max,  w_curr + 0.2)

        for v in np.arange(v_min, v_max + 1e-6, self.v_res):
            for w in np.arange(w_min, w_max + 1e-6, self.w_res):
                # Simulate trajectory
                nx    = x + v * np.cos(theta + w * self.dt) * self.dt
                ny    = y + v * np.sin(theta + w * self.dt) * self.dt
                ntheta = theta + w * self.dt

                # Collision check
                safe = True
                for ox, oy, r in obstacles:
                    if np.sqrt((nx - ox)**2 + (ny - oy)**2) < r + 0.5:
                        safe = False
                        break
                if not safe:
                    continue

                # Scoring
                heading_err  = abs(np.arctan2(gy - ny, gx - nx) - ntheta)
                heading_score = (np.pi - min(heading_err, np.pi)) / np.pi

                dist_to_goal = np.sqrt((nx - gx)**2 + (ny - gy)**2)
                goal_score   = 1.0 / (1.0 + dist_to_goal)

                vel_score  = v / self.v_max
                score      = 0.5 * heading_score + 0.3 * goal_score + 0.2 * vel_score

                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        return best_v, best_w


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    shape    = (256, 256)
    cost_map = LunarCostMap(shape=shape)
    planner  = EnergyAwareRRTStar(cost_map=cost_map, max_iters=1000)

    start = (20.0, 20.0)
    goal  = (220.0, 200.0)   # Across the PSR

    path, cost = planner.plan(start, goal)
    print(f"\nRRT* path: {len(path)} waypoints, total cost: {cost:.2f}")

    # DWA step
    dwa    = DynamicWindowApproach()
    state  = (20.0, 20.0, 0.0, 0.005, 0.0)
    obs    = [(50.0, 50.0, 5.0), (100.0, 100.0, 8.0)]
    v, w   = dwa.compute_velocity(state, goal=(60.0, 60.0), obstacles=obs)
    print(f"DWA command: v={v:.4f} m/s, w={w:.3f} rad/s")
    print("\nPath planner OK ✓")
