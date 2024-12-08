import torch
import torch.nn as nn
import torch.nn.functional as F

class BBN(nn.Module):
    def __init__(self, num_classes):
        super(BBN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(64 * 32 * 32, num_classes)  # 假设输入图像尺寸为 64x64

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        logits = self.fc(features)
        return logits