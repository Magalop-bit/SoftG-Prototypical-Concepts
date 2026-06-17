import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.linear = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.linear(out)
        return out
    
if __name__ == "__main__":
    model = LeNet(num_classes=9).to(device)

    # Verify model on CPU first
    model_cpu = LeNet(num_classes=9).cpu()
    dummy = torch.randn(2, 1, 28, 28)
    out = model_cpu(dummy)
    print(f"Model output shape: {out.shape}")  # must be [2, 9]
    assert out.shape == (2, 9), f"Bad output shape: {out.shape}"

    model = model_cpu.to(device)