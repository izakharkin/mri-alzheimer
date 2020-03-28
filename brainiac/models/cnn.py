import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        
        self.conv = nn.Conv3d(1, 64, kernel_size=7, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=7, stride=7)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(183872, 1024)  # 98304 
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, out):
        out = self.conv(out)
        out = self.pool(out)
        out = self.fl(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

