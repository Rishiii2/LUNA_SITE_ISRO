import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LunarIceDataset(Dataset):
    def __init__(self, data_dir):
        """
        Custom PyTorch Dataset for loading PRADAN DFSAR and LOLA DEM data.
        """
        self.data_dir = data_dir
        self.dfsar_files = sorted(glob.glob(os.path.join(data_dir, 'dfsar_tile_*.npy')))
        self.lola_files = sorted(glob.glob(os.path.join(data_dir, 'lola_tile_*.npy')))
        
        assert len(self.dfsar_files) == len(self.lola_files), "Mismatch between DFSAR and LOLA tiles!"

    def __len__(self):
        return len(self.dfsar_files)

    def _generate_proxy_label(self, stokes_tensor, lola_tensor):
        """
        Weakly-Supervised Label Generation.
        Since we have no ground truth ice labels, we generate a proxy label 
        based on the Physics of radar:
        CPR > 1.0 AND DOP < 0.13 AND inside PSR.
        """
        s1, s2, s3, s4 = stokes_tensor[0], stokes_tensor[1], stokes_tensor[2], stokes_tensor[3]
        
        # Avoid division by zero
        s1_safe = np.where(s1 == 0, 1e-6, s1)
        s1_plus_s4 = np.where((s1 + s4) == 0, 1e-6, s1 + s4)
        
        cpr = (s1 - s4) / s1_plus_s4
        dop = np.sqrt(s2**2 + s3**2 + s4**2) / s1_safe
        
        psr_mask = lola_tensor[1] # 1 is Dark, 0 is Sunlit
        
        # Generate 64x64 binary label mask
        ice_mask = (cpr > 1.0) & (dop < 0.13) & (psr_mask == 1.0)
        
        # Return a single value for the whole tile (1 if significant ice is found, else 0)
        tile_label = 1.0 if np.sum(ice_mask) > (0.1 * 64 * 64) else 0.0
        return np.array([tile_label], dtype=np.float32)

    def __getitem__(self, idx):
        # Load arrays
        stokes_tensor = np.load(self.dfsar_files[idx])
        lola_tensor = np.load(self.lola_files[idx])
        
        # Generate weak labels based on CPR/DOP
        label = self._generate_proxy_label(stokes_tensor, lola_tensor)
        
        # Extract features for CNN (Just Stokes vector for now)
        features = torch.tensor(stokes_tensor, dtype=torch.float32)
        target = torch.tensor(label, dtype=torch.float32)
        
        return features, target

def get_dataloader(data_dir, batch_size=16):
    dataset = LunarIceDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    print("[Layer 0] Testing LunarIceDataset PyTorch DataLoader...")
    loader = get_dataloader(r"C:\Users\rishi\Downloads\ISRO\code\data", batch_size=4)
    features, targets = next(iter(loader))
    print(f"Features Batch Shape: {features.shape} (Batch, Channels, H, W)")
    print(f"Targets Batch Shape: {targets.shape} (Batch, Label)")
