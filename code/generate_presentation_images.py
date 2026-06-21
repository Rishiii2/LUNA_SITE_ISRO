import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Ensure we save the images in the same directory as the script
output_dir = "presentation_images"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. Generate Grad-CAM Heatmap (CNN Ice Detector)
# ---------------------------------------------------------
def generate_grad_cam():
    print("Generating Grad-CAM Heatmap...")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create a base "radar" image (noisy grayscale)
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    base_radar = np.sin(X**2 + Y**2) + np.random.normal(0, 0.5, (200, 200))
    
    # Create the "activation" heatmap (Gaussian blob over the center "crater")
    heatmap = np.exp(-(X**2 + Y**2) / 2.0)
    
    ax.imshow(base_radar, cmap='gray', alpha=0.8)
    ax.imshow(heatmap, cmap='jet', alpha=0.5)
    ax.set_title("CNN Grad-CAM Activation\nTarget: Volumetric Ice Scattering", color='white')
    ax.axis('off')
    
    # ISRO Style Dark Mode
    fig.patch.set_facecolor('#1e1e1e')
    plt.savefig(os.path.join(output_dir, "grad_cam_demo.png"), facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=300)
    plt.close()

# ---------------------------------------------------------
# 2. Generate m-chi Volume Scattering Map (Radar Physics)
# ---------------------------------------------------------
def generate_m_chi_map():
    print("Generating m-chi Decomposition Maps...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    
    # Synthetic Volume data (High values indicate ice)
    v_vol = np.exp(-(X**2 + Y**2)) * 0.8 + np.random.normal(0, 0.05, (200, 200))
    # Synthetic Double-Bounce data
    d_dbl = 0.5 - np.exp(-(X**2 + Y**2)) * 0.4 + np.random.normal(0, 0.05, (200, 200))
    
    im1 = axes[0].imshow(v_vol, cmap='magma', vmin=0, vmax=1.0)
    axes[0].set_title("m-chi Volume Scattering (V > 0.4)", color='white')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(d_dbl, cmap='viridis', vmin=0, vmax=1.0)
    axes[1].set_title("m-chi Double-Bounce Scattering", color='white')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    fig.patch.set_facecolor('#1e1e1e')
    for ax in axes:
        ax.set_facecolor('#1e1e1e')
        
    plt.savefig(os.path.join(output_dir, "m_chi_map.png"), facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=300)
    plt.close()

# ---------------------------------------------------------
# 3. Generate NSGA-II AHP-TOPSIS Pareto Front Chart
# ---------------------------------------------------------
def generate_pareto_front():
    print("Generating Pareto Front Scatter Plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate random points for "All Evaluated Sites"
    ice_volume = np.random.normal(50, 20, 200)
    hazard_score = np.random.normal(50, 20, 200)
    
    # Filter to make a curve
    valid = ice_volume < (100 - (hazard_score - 10)**2 / 40)
    ice_volume = ice_volume[valid]
    hazard_score = hazard_score[valid]
    
    # Pareto front (optimal boundary)
    pareto_x = np.sort(hazard_score[hazard_score < 60])[:15]
    pareto_y = 100 - (pareto_x - 10)**2 / 40 + np.random.normal(0, 2, len(pareto_x))
    
    ax.scatter(hazard_score, ice_volume, color='gray', alpha=0.5, label='Grid Scanned Sites')
    ax.scatter(pareto_x, pareto_y, color='#00ffcc', s=100, label='Pareto Optimal Front', edgecolors='white')
    ax.plot(np.sort(pareto_x), np.sort(pareto_y)[::-1], color='#00ffcc', linestyle='--', alpha=0.8)
    
    # Annotate the best site
    idx = len(pareto_x) // 2
    ax.annotate('AHP-TOPSIS Ranked #1 Target', 
                xy=(pareto_x[idx], pareto_y[idx]), xytext=(pareto_x[idx]+10, pareto_y[idx]+10),
                arrowprops=dict(facecolor='white', shrink=0.05), color='white')
    
    ax.set_xlabel("Traverse Hazard Score (Lower is Better)", color='white')
    ax.set_ylabel("Estimated Ice Volume (Higher is Better)", color='white')
    ax.set_title("NSGA-II + AHP-TOPSIS Multi-Objective Optimization", color='white')
    ax.legend(facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
    
    ax.tick_params(colors='white')
    ax.grid(color='gray', linestyle='--', alpha=0.3)
    
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    plt.savefig(os.path.join(output_dir, "pareto_front_nsga2.png"), facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_grad_cam()
    generate_m_chi_map()
    generate_pareto_front()
    print(f"All images saved successfully to {output_dir}")
