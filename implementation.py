import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

# ==========================================
# 1. Configuration & Reproducibility
# ==========================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on device: {device}")

# Hyperparameters inspired by the paper's regime
EPOCHS = 15 # Kept low for demo purposes; paper uses more, but phenomenon is visible early
LR_BASE = 0.01
SB_SIZE = 256     # Small Batch
LB_SIZE = 4096    # Large Batch (Significant fraction of dataset)

# ==========================================
# 2. Data Strategy: CIFAR-10
# ==========================================
# We use CIFAR-10 as a standard proxy for large-scale vision tasks.
# It allows us to observe generalization gaps without needing ImageNet-scale compute.

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Ensure directory exists
os.makedirs('./data', exist_ok=True)

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

# ==========================================
# 3. Model Architecture
# ==========================================
# A VGG-style CNN. Deep enough to have a complex loss landscape with sharp/flat minima.
class SimpleVGG(nn.Module):
    def __init__(self):
        super(SimpleVGG, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

# ==========================================
# 4. Training Engine
# ==========================================

def train_regime(batch_size, label):
    print(f"\nStarting Training Regime: {label} (Batch Size: {batch_size})")
    
    # Scaling Rule: To represent a fair comparison, we often scale LR with batch size.
    # However, Keskar et al. argue that even with tuning, LB generalizes worse.
    # We use a mild square-root scaling heuristic to give LB a fighting chance.
    lr = LR_BASE * (np.sqrt(batch_size) / np.sqrt(SB_SIZE))
    
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)
    
    model = SimpleVGG().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_losses = []
    test_accs = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        
        avg_loss = running_loss / len(loader)
        acc = get_accuracy(model, test_loader)
        train_losses.append(avg_loss)
        test_accs.append(acc)
        
        print(f"[{label}] Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")

    return model, train_losses, test_accs

# ==========================================
# 5. Core Analysis: Visualizing Sharpness
# ==========================================
# To verify the paper's claim, we visualize the 1D loss landscape around the found solution.
# Sharp Minima = Loss rises quickly as we move away from weights theta.
# Flat Minima = Loss rises slowly.

def compute_sharpness_curve(model, loader, alpha_range=np.linspace(-1.0, 1.0, 20)):
    """
    Calculates loss L(theta + alpha * d) where d is a random direction.
    Uses filter-wise normalization to ensure fairness.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # 1. Generate a random direction 'd' with filter normalization
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        # Filter normalization (Li et al., 2018 refinement of Keskar's concept)
        # Helps compare landscapes of different scales.
        if len(p.shape) == 4: # Conv layer
            n_filters = p.shape[0]
            for i in range(n_filters):
                n = p[i].norm()
                dn = d[i].norm()
                d[i] = d[i] * (n / (dn + 1e-6))
        else: # FC or Bias
             n = p.norm()
             dn = d.norm()
             d = d * (n / (dn + 1e-6))
        direction.append(d.to(device))
    
    losses = []
    original_params = [p.clone() for p in model.parameters()]

    # 2. Iterate through alpha values
    print("Computing landscape curvature...")
    with torch.no_grad():
        # Use a subset of data for landscape estimation to be fast
        mini_loader_iter = iter(loader)
        inputs, targets = next(mini_loader_iter)
        inputs, targets = inputs.to(device), targets.to(device)

        for alpha in alpha_range:
            # Perturb weights: theta_new = theta + alpha * direction
            for i, p in enumerate(model.parameters()):
                p.data = original_params[i] + alpha * direction[i]
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
    
    # Restore original weights
    for i, p in enumerate(model.parameters()):
        p.data = original_params[i]
        
    return losses

# ==========================================
# 6. Main Execution
# ==========================================

if __name__ == "__main__":
    # Train Small Batch
    sb_model, sb_loss, sb_acc = train_regime(SB_SIZE, "Small_Batch")
    
    # Train Large Batch
    lb_model, lb_loss, lb_acc = train_regime(LB_SIZE, "Large_Batch")
    
    print("\n=== Generalization Gap Results ===")
    print(f"Small Batch Final Test Acc: {sb_acc[-1]:.2f}%")
    print(f"Large Batch Final Test Acc: {lb_acc[-1]:.2f}%")
    print(f"Gap: {sb_acc[-1] - lb_acc[-1]:.2f}%")

    # Analyze Curvature
    # Use the test loader to measure generalization landscape, or train loader for optimization landscape
    # The paper discusses training landscape sharpness.
    check_loader = DataLoader(trainset, batch_size=2048, shuffle=True)
    
    alphas = np.linspace(-1.0, 1.0, 25)
    sb_curve = compute_sharpness_curve(sb_model, check_loader, alphas)
    lb_curve = compute_sharpness_curve(lb_model, check_loader, alphas)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, sb_curve, 'b-o', label=f'Small Batch (BS={SB_SIZE}) - Flat?', linewidth=2)
    plt.plot(alphas, lb_curve, 'r-s', label=f'Large Batch (BS={LB_SIZE}) - Sharp?', linewidth=2)
    plt.title(f'1D Loss Landscape Visualization (CIFAR-10)\nCenter (0.0) is the converged model')
    plt.xlabel('Perturbation Magnitude (alpha)')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('sharpness_comparison.png')
    print("\nPlot saved as 'sharpness_comparison.png'.")
    print("Interpretation: If the Red line (LB) is steeper/narrower than Blue (SB), the hypothesis holds.")
