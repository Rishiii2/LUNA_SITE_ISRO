import os
import numpy as np

def create_synthetic_dataset(base_dir, num_samples=100):
    """
    Generates synthetic ISRO PRADAN (DFSAR) and LOLA DEM datasets.
    This mimics 4-channel Stokes vectors (S1, S2, S3, S4) and DEM patches.
    """
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"[Layer 0] Generating {num_samples} synthetic ISRO DFSAR & LOLA DEM tiles...")
    
    for i in range(num_samples):
        # 1. Generate 4-channel Stokes vector (DFSAR) - 64x64 tiles
        # S1: Total intensity (0 to 1)
        s1 = np.random.uniform(0.1, 1.0, (64, 64))
        # S2, S3, S4: Polarization states
        s2 = np.random.uniform(-0.5, 0.5, (64, 64))
        s3 = np.random.uniform(-0.5, 0.5, (64, 64))
        s4 = np.random.uniform(-0.5, 0.5, (64, 64))
        
        stokes_tensor = np.stack([s1, s2, s3, s4], axis=0)
        
        # 2. Generate LOLA DEM (Elevation & Illumination Mask)
        # Elevation between -2000m and 100m
        dem = np.random.uniform(-2000, 100, (64, 64))
        # PSR Mask (1 = Dark, 0 = Sunlit)
        psr_mask = np.random.choice([0.0, 1.0], size=(64, 64), p=[0.8, 0.2])
        
        lola_tensor = np.stack([dem, psr_mask], axis=0)
        
        # Save to disk as .npy to mimic parsed .img GeoTIFFs
        np.save(os.path.join(data_dir, f'dfsar_tile_{i}.npy'), stokes_tensor)
        np.save(os.path.join(data_dir, f'lola_tile_{i}.npy'), lola_tensor)

    print(f"[✔] Successfully generated synthetic datasets in {data_dir}")

if __name__ == "__main__":
    create_synthetic_dataset("data")
