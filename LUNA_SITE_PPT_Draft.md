# LUNA-SITE: Final Presentation Draft (v3 SOTA Edition)

This document contains the slide-by-slide content for the ultimate Idea Submission PPT. It covers all 21 layers, updated with the new State-of-the-Art (SOTA) physics upgrades (m-chi, Bekker, AHP-TOPSIS).

---

## Slide 1: Title Slide
* **Title:** LUNA-SITE: Autonomous Landing Intelligence
* **Subtitle:** An End-to-End Multi-Objective Site Optimizer & Digital Twin
* **Key Focus:** SOTA Radar Physics, Multi-Modal Neural Networks, and AHP-TOPSIS Site Selection.

## Slide 2: Mission Context & ISRO Heritage
* **Content:**
  * **Learnings from the Past:** Chandrayaan-2's DFSAR gave us polarimetric L-band/S-band data.
  * **The Chandrayaan-4 Challenge:** For sample return, we must find water ice for ISRU, land safely, navigate a rover, and excavate.
  * **Our Baseline:** LUNA-SITE autonomously replicates, validates, and extends ISRO's selection process using advanced AI and orbital physics.

## Slide 3: The Scientific Problem
* **Content:**
  * **The Target:** Permanently Shadowed Regions (PSRs).
  * **The Difficulty:** We must rely on radar (DFSAR). However, rocky, rough terrain often mimics volumetric ice. 
  * **The LUNA-SITE Solution:** A 21-layer pipeline that fuses $m-\chi$ polarimetric physics with deep learning to find ice, ranks optimal landing sites via AHP-TOPSIS, and computes energy-safe rover paths using Bekker terramechanics.

## Slide 4: The 21-Layer LUNA-SITE Architecture
* **Content:** 
  * **Data:** PDS4 Parsing, PSR Mapping, LOLA DEM.
  * **Physics:** $m-\chi$ Polarimetric Decomposition, Roughness Rejection, Hungarian Orbit Tracking.
  * **AI/ML:** 5-ch CNN, MC Dropout, Grad-CAM XAI.
  * **Mechanics:** Depth Probability, Ice Volume, Dual-Zone Hazards, Traversability, Solar SPICE.
  * **Optimization:** Excavation Priority, ISRU Score, NSGA-II Pareto, AHP-TOPSIS Ranking.
  * **Execution:** Bekker RRT* Path, DWA Local Avoidance, Streamlit Dashboard.

## Slide 5: Phase 1 - Radar Physics Upgrade ($m-\chi$ Decomposition)
* **Algorithms & Justification:**
  * **Previous Flaw:** Basic Circular Polarization Ratio (CPR) heuristics suffer from surface roughness false-positives.
  * **Our Improvement:** Implemented the industry-standard **$m-\chi$ Polarimetric Decomposition** used on Chandrayaan-2 DFSAR data to mathematically isolate Volume Scattering (V > 0.4) from Surface and Double-Bounce scattering.
  * **Hungarian Tracking:** We use Kalman filters and the Hungarian algorithm to track polarimetric anomalies across multiple orbital passes.

## Slide 6: Phase 1 - Physics-Regularized Multi-Modal CNN
* **Algorithms & Justification:**
  * **Architecture:** A 5-channel CNN processing raw Stokes Vectors fused with DEM Roughness maps.
  * **Our Improvement:** The Loss function explicitly penalizes the network if it predicts ice where physics dictates it shouldn't exist.
  * **Explainability:** Monte Carlo (MC) Dropout for epistemic uncertainty and Grad-CAM for visual validation (avoiding the "Black Box" AI trap).

## Slide 7: Phase 2 - Hazard Mapping (The Optical Paradox)
* **Algorithms & Justification:**
  * **The Optical Paradox:** YOLOv8 cannot see inside a pitch-black PSR! 
  * **Dual-Zone Mapping:** We split hazards: **Sunlit Approach** (YOLOv8 on OHRC detects boulders) and **Dark PSR Target** (LOLA DEM roughness and Radar only). 

## Slide 8: Phase 3 - Multi-Objective Site Ranking (AHP-TOPSIS)
* **Algorithms & Justification:**
  * **Process:** Dense Grid Scanning evaluates thousands of candidates. Fast Non-Dominated Sorting (NSGA-II) isolates the optimal Pareto Front.
  * **Our Improvement:** Integrated the **AHP-TOPSIS** decision matrix to rank the final targets. It mathematically weighs competing constraints (Ice Volume vs. Solar Hours vs. Hazard) to output the absolute #1 best target, mirroring real ISRO mission planning.

## Slide 9: Phase 4 - Energy-Aware Rover Traversal (Bekker Terramechanics)
* **Algorithms & Justification:**
  * **Process:** RRT* computes optimal global routes.
  * **Our Improvement:** Rewrote the cost function from generic heuristics to exact Joules using **Bekker Terramechanics**. It models mechanical work against lunar gravity ($1.62 m/s^2$) and regolith rolling resistance, ensuring the rover reaches the extraction site without depleting its 100Wh battery.

## Slide 10: Phase 5 - The Mission Control Dashboard
* **Content:**
  * Live demonstration of the Streamlit dashboard rendering the $m-\chi$ polarimetry, AHP-TOPSIS rankings, and the live Bekker RRT* Battery Gauge.

## Slide 11: Team Roles
* **[Team Member 1]:** Aerospace Physics & Optimization (m-chi, AHP-TOPSIS, Bekker)
* **[Team Member 2]:** Artificial Intelligence (5-Channel CNN, Grad-CAM, MC Dropout)
* **[Team Member 3]:** Robotics & UI (RRT*, ROS2 Integration, Streamlit Dashboard)

## Slide 12: Thank You / Mission Metrics
* **5-Channel Physics Fusion**
* **Exact Joule Traversal Modeling**
* **100% SOTA Aerospace Compliance**
* Thank you to the ISRO Hackathon Judges!
