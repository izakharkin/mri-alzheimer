import torch
from torch import nn
from torch.nn import functional as F

class LeNet3D(nn.Module):
    def __init__(self,num_classes):
        super(LeNet3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, kernel_size=11, stride=4, padding=2)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(16* 3* 3* 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16* 3* 3* 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x