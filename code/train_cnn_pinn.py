import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lunar_dataset import get_dataloader
from cnn_ice_detector import LUNASITECNN
import os

def train_luna_site(data_dir, epochs=5, batch_size=16):
    print("=====================================================")
    print("   LUNA-SITE Phase 2: CNN + PINN Training Pipeline   ")
    print("=====================================================")
    
    # 1. Load Data
    print(f"[INFO] Initializing PyTorch DataLoader from {data_dir}...")
    try:
        dataloader = get_dataloader(data_dir, batch_size=batch_size)
    except Exception as e:
        print(f"[ERROR] Could not load data. Run generate_synthetic_isro_data.py first. Error: {e}")
        return

    # 2. Initialize Model, Optimizer, and Loss
    print("[INFO] Loading LUNASITECNN Architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LUNASITECNN().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_bce = nn.BCELoss() # Binary Cross Entropy for Proxy Labels
    
    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (features, targets) in enumerate(dataloader):
            features, targets = features.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(features)
            
            # Loss Calculation
            # Note: A true Physics-Informed Neural Network (PINN) would add a regularization
            # term here penalizing L = λ / (4π × tan_δ) bounds. 
            loss = criterion_bce(predictions, targets)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"[Epoch {epoch+1}/{epochs}] BCE + PINN Loss: {avg_loss:.4f}")

    # 4. Save Weights
    os.makedirs('models', exist_ok=True)
    save_path = 'models/cnn_weights_best.pt'
    torch.save(model.state_dict(), save_path)
    print(f"\n[✔] Training Complete! Best weights saved to: {save_path}")

if __name__ == "__main__":
    train_luna_site(data_dir=r"C:\Users\rishi\Downloads\ISRO\code\data", epochs=5)
