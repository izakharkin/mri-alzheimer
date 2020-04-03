import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):

    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=7, stride=7)
#         self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(183872, 1024)  # 98304 
        self.fc2 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(0.2)


    def forward(self, out):
        out = self.dropout(self.pool1(F.relu(self.conv1(out))))
#         out = F.relu(self.conv2(out))
#         out = self.dropout(self.pool2(F.relu(self.conv3(out))))
        out = self.fl(out)
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out)
        return out

