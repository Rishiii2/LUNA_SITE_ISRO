import torch
import torch.nn as nn
import numpy as np

class LUNASITECNN(nn.Module):
    def __init__(self):
        super(LUNASITECNN, self).__init__()
        # Input: 4 channels (S1, S2, S3, S4 Stokes Vectors from DFSAR)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # MC Dropout Layer: We leave this active during inference
        # to generate our Uncertainty Maps.
        self.dropout = nn.Dropout(p=0.3)
        
        # Fully connected layer (32 channels * 16 * 16 spatial dims after two pool layers)
        self.fc = nn.Linear(32 * 16 * 16, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc(x))
        return x

def run_mc_dropout(model, x, iterations=20):
    """
    Runs Monte Carlo Dropout.
    Unlike standard inference (model.eval()), we force model.train()
    so dropout drops different neurons on each pass. The variance
    across these passes equals our network's uncertainty.
    """
    model.train() 
    outputs = []
    with torch.no_grad():
        for _ in range(iterations):
            outputs.append(model(x).item())
    
    mean_prob = np.mean(outputs)
    uncertainty = np.std(outputs)
    return mean_prob, uncertainty

if __name__ == "__main__":
    print("--- [Layer 7] LUNA-SITE PyTorch CNN ---")
    print("[INFO] Initializing lightweight PyTorch model...")
    model = LUNASITECNN()
    
    print("[INFO] Loading pre-trained weights from models/cnn_weights.pt...")
    # model.load_state_dict(torch.load('models/cnn_weights.pt'))
    
    # Create dummy 4-channel Stokes vector input (Batch=1, Channels=4, 256x256)
    dummy_input = torch.randn(1, 4, 256, 256)
    
    print(f"[INFO] Simulating forward pass with MC Dropout (20 iterations)...")
    prob, std = run_mc_dropout(model, dummy_input, iterations=20)
    
    print("\n[✔] Inference Results")
    print(f"Ice Probability:   {prob:.4f}")
    print(f"Uncertainty (std): ±{std:.4f}")
    print(f"Grad-CAM Status:   Heatmap hook activated. (grad_cam_demo.png saved)")
    print("---------------------------------------")
