# LUNA-SITE: Comprehensive Engineering & Scientific Documentation
**Project:** Bharatiya Antariksh Hackathon 2026 | ISRO Challenge 8
**Topic:** Detection and Characterization of Subsurface Ice in Lunar South Polar Regions Using Chandrayaan-2 Radar and Imagery Data for Landing Site and Rover Traverse Planning.

---

## 1. Project Evolution: From Concept to ISRO Phase-A Architecture
Initially, LUNA-SITE was conceptualized as a standard hackathon AI project—a simple neural network that takes radar images and outputs ice locations. However, standard machine learning pipelines fail when applied to real-world space missions because they lack physical constraints and ignore the severe limitations of lunar hardware.

We evolved the project into a **21-Layer, 5-Phase Mission Planning Architecture**. Instead of treating this as a student data science project, we structured it exactly like ISRO's Space Applications Centre (SAC) would design a pre-mission planner for Chandrayaan-4. 

We integrated strict hardware constraints (Chandrayaan-4's <10° slope tolerance and <0.32m boulder clearance) and engineered a full robotics pipeline (RRT* + DWA) derived from the team's validated Tactical Aerial Combat Simulator codebase.

---

## 2. Initial Flaws & The Engineering Fixes

During the design phase, we identified four massive architectural logic traps that typical hackathon teams fall into. We redesigned our system to explicitly mitigate them.

### Flaw 1: The Optical Paradox
*   **The Trap:** Many teams use optical object detection (like YOLO) to map hazards. However, Challenge 8 explicitly targets Doubly Shadowed Craters (PSRs). Optical cameras are useless in pitch-black craters (temperatures at ~25K with zero photons).
*   **The LUNA-SITE Fix:** **Dual-Zone Hazard Mapping (Layer 10).** We explicitly split the pipeline. YOLOv8 is *only* used during the sunlit approach phase (analyzing OHRC images). Once the rover enters the PSR, the system automatically bypasses optical data and relies entirely on LOLA DEM and DFSAR surface roughness metrics.

### Flaw 2: The "Black Box" AI Trap
*   **The Trap:** Deep learning models (CNNs) are notorious for overfitting. A standard CNN might just memorize the visual shape of a crater rather than actually learning the physics of radar scattering. ISRO cannot trust unexplainable AI.
*   **The LUNA-SITE Fix:** **Explainable AI via Grad-CAM & MC Dropout (Layer 7).** We implemented Gradient-weighted Class Activation Mapping (Grad-CAM) to generate heatmaps, proving visually that the CNN is activating on the high-CPR radar physics, not crater geometry. Additionally, we use Monte Carlo (MC) Dropout during inference to output explicit "Uncertainty Confidence Bounds" rather than blind predictions.

### Flaw 3: Depth Estimation Overconfidence
*   **The Trap:** Claiming an AI can predict the *exact* depth of subsurface ice from orbital radar alone is scientifically inaccurate. Exact depth requires knowing the exact loss tangent ($\tan \delta$) of the local lunar regolith, which is unknown.
*   **The LUNA-SITE Fix:** **Physics-Informed Neural Networks (PINN).** Instead of outputting a single false depth integer, our PINN encodes the radar penetration equation to output a **Depth Probability Bound** (e.g., "75% probability that ice exists within the 2-meter drill limit").

### Flaw 4: The 30-Hour Compute Trap
*   **The Trap:** Trying to train deep learning models on terabytes of PRADAN datasets and running heavy global RRT* pathfinding algorithms live during the 30-hour Grand Finale will lead to hardware crashes and a failed demo.
*   **The LUNA-SITE Fix:** **Inference-First & Pre-Computation.** We designed the codebase to load pre-trained `.pt` weights. The global RRT* path is pre-computed, while only the lightweight DWA (Dynamic Window Approach) runs in real-time in the ROS2 Gazebo Digital Twin.

---

## 3. The Mathematics & Physics in Depth

### A. Polarimetric Radar Physics (CPR & DOP)
The Chandrayaan-2 Dual Frequency Synthetic Aperture Radar (DFSAR) captures backscattered energy in a 4-element Stokes Vector matrix ($S_1, S_2, S_3, S_4$). 

To detect ice, we look for **Volumetric Scattering** (caused by internal reflections inside ice blocks) rather than surface scattering (caused by flat dirt). 

1.  **Circular Polarization Ratio (CPR):**
    Ice reverses the polarization of the radar wave, causing the Same-Sense polarization to exceed the Opposite-Sense polarization.
    $$CPR = \frac{S_1 - S_4}{S_1 + S_4}$$
    **Threshold:** $CPR > 1.0$ strongly indicates volumetric ice or very rough blocky rocks.

2.  **Degree of Polarization (DOP):**
    Ice causes high depolarization (chaotic scattering). 
    $$DOP = \frac{\sqrt{S_2^2 + S_3^2 + S_4^2}}{S_1}$$
    **Threshold:** $DOP < 0.13$ helps filter out rough rocks, isolating pure ice.

### B. Physics-Informed Neural Network (PINN) Loss Equation
To estimate the depth of the ice ($L$), we use the electromagnetic penetration depth equation:
$$L = \frac{\lambda}{4\pi \sqrt{\epsilon'} \tan \delta}$$
Where:
*   $\lambda$ = Radar Wavelength (L-band = ~24cm, S-band = ~10cm)
*   $\epsilon'$ = Real part of the dielectric constant of lunar regolith.
*   $\tan \delta$ = Loss tangent (attenuation factor).

Because $\tan \delta$ is highly variable across the Moon, our PINN does not predict $L$ directly. It predicts $\epsilon'$ and $\tan \delta$ as distributions, creating a bounded probability function. The Custom PyTorch Loss function is:
$$Loss = BCE(Y_{pred}, Y_{true}) + \alpha \cdot \max(0, \frac{\lambda}{4\pi \tan \delta} - L_{max})$$
*(This penalizes the network if it predicts penetration depths that violate known physics).*

### C. Multi-Objective Site Ranking (NSGA-II)
We cannot simply maximize ice. We must optimize a 3D trade-off space: Ice Volume vs. Terrain Safety vs. Solar Illumination.
We use the **Non-dominated Sorting Genetic Algorithm II (NSGA-II)** to find the "Pareto Front" of optimal landing sites. 

---

## 4. Codebase Architecture & Implementation

The repository (located in `ISRO/code/`) contains a fully functional software prototype mirroring this architecture.

### 1. Data Mastery & Weak Labeling
*   `generate_synthetic_isro_data.py`: Solves the problem of local data limits by simulating 4-channel DFSAR Stokes arrays and LOLA DEM patches.
*   `lunar_dataset.py`: A PyTorch `DataLoader` that implements **Weakly-Supervised Labeling**. Because nobody has a ground-truth "answer key" for lunar ice, this script dynamically labels pixels as "Ice" during training if they mathematically satisfy the $CPR > 1$ and $DOP < 0.13$ physics equations.

### 2. Deep Learning Pipeline (Phase 2)
*   `cnn_ice_detector.py`: Implements the 2D Convolutional Neural Network. It explicitly forces PyTorch's `nn.Dropout(p=0.3)` to remain active during `model.train()` evaluation to simulate Monte Carlo approximations for Uncertainty bounds.
*   `train_cnn_pinn.py`: The actual PyTorch execution loop utilizing AdamW optimization to minimize the Binary Cross Entropy of the proxy physics labels. 

### 3. Hazard Mapping (Phase 3)
*   `yolo_hazard_mapper.py`: Programmatically implements the Dual-Zone mitigation for the Optical Paradox. It uses geolocation checks to bypass optical libraries (`ultralytics`) inside PSRs, keeping the rover safe from blind spots.
*   `train_yolo_hazards.py`: Automates the `yaml` configuration required to fine-tune pre-trained YOLO weights on lunar surface imagery.

### 4. The Digital Twin (Phase 5)
*   `luna_site_dashboard.py`: A comprehensive Streamlit Mission Control interface. It provides a highly polished UI to demonstrate the Pareto optimization charts, the simulated topographical maps, and the real-time ROS2 terminal feed for rover telemetry.

---
**Final Status:** Ready for July 1st Idea Submission. The documentation, the mathematics, and the Python prototype are flawlessly aligned.
