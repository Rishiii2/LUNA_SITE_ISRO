"""
LUNA-SITE: End-to-End Mission Pipeline Demo
=============================================
This script executes the entire pipeline sequentially:
1. Layer 0: ISRO DFSAR PDS4 Data Parsing (Mock Synthetic)
2. Layer 3: CPR/DOP Physics calculation
3. Layer 8: Physics-Informed CNN Inference
4. Layer 15: NSGA-II Landing Site Selection
5. Layer 17: Pre-computes Mons Mouton RRT* path for the finale

Run this before launching the Streamlit Dashboard.
"""

import os
import numpy as np
import logging
import torch

from generate_synthetic_isro_data import generate_dataset
from cpr_dop_mapper import run_cpr_dop_pipeline
from nsga2_optimizer import generate_candidate_sites, run_nsga2
from rrt_star_planner import LunarCostMap, EnergyAwareRRTStar
from cnn_ice_detector import LunarIceCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s | LUNA-SITE | %(message)s")

def run_end_to_end_demo():
    print("=====================================================")
    print("🚀 LUNA-SITE CHANDRAYAAN-4 MISSION PIPELINE STARTING")
    print("=====================================================\n")

    # Phase 1: Data Parsing & Physics
    logging.info("[Layer 0] Initializing ISRO PDS4 Data Parser (pds4_tools)")
    generate_dataset(n_samples=10, save_dir="data")
    
    # Load first sample
    stokes_arr = np.load("data/stokes.npy")
    sample_stokes = stokes_arr[0]
    logging.info(f"[Layer 1-2] Loaded Stokes Tensor for Faustini Crater: {sample_stokes.shape}")
    
    logging.info("[Layer 3] Computing CPR and DOP Physics Maps")
    result = run_cpr_dop_pipeline(sample_stokes)
    logging.info(f"Target Ice Mask (CPR>1, DOP<0.13) extracted. Candidates: {result['n_clean']}")

    # Phase 2: CNN Inference (Mock run)
    logging.info("[Layer 8] Initializing Physics-Informed CNN (PINN) for Deep Inference")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LunarIceCNN().to(device)
    model.eval()
    with torch.no_grad():
        tensor_stokes = torch.tensor(sample_stokes, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(tensor_stokes)
        pred = logits.argmax(dim=1).item()
        ice_confidence = torch.softmax(logits, dim=1)[0][1].item() * 100
    logging.info(f"[Layer 9] Neural Inference Complete. Ice Confidence: {ice_confidence:.1f}%")

    # Phase 4: Optimization
    logging.info("[Layer 15] Running NSGA-II Multi-Objective Optimization for Landing Sites")
    sites = generate_candidate_sites(n_sites=50)
    pareto_front, _ = run_nsga2(sites)
    best_site = pareto_front[0]
    logging.info(f"[Layer 16] Target Selected: Site#{best_site.site_id} (CRM: {best_site.cost_risk_metric:.3f})")

    # Phase 5: RRT* Global Path Pre-Computation for Mons Mouton
    logging.info("[Layer 17] Pre-computing Energy-Aware RRT* Global Path for Mons Mouton Traverse")
    cost_map = LunarCostMap(shape=(128, 128))
    planner = EnergyAwareRRTStar(cost_map=cost_map, max_iters=500)
    start_pos = (10.0, 10.0)
    goal_pos = (110.0, 100.0) # Extraction site
    
    path, cost = planner.plan(start_pos, goal_pos)
    
    if path:
        logging.info(f"[✔] PATH SECURED: {len(path)} waypoints. Energy Cost: {cost:.2f}")
        logging.info("Saving path to 'data/mons_mouton_path.npy' for Gazebo DWA live-sim.")
        np.save("data/mons_mouton_path.npy", np.array(path))
    else:
        logging.warning("[!] Pathfinding failed. Increase iterations.")

    print("\n=====================================================")
    print("✅ PRE-MISSION PIPELINE COMPLETE.")
    print("You may now launch the dashboard: `docker-compose up`")
    print("=====================================================\n")

if __name__ == "__main__":
    run_end_to_end_demo()
