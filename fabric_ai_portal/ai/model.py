"""MODEL CONFIGURATION (max 40 p)
Featurizer block (stem + residual blocks with batch normalization) and classification block
(dropout + fully connected layers).  Loss, optimizer, LR scheduler are configured in train.py.
"""
import torch
import torch.nn as nn

import config


class ResidualBlock(nn.Module):
    """conv-BN-ReLU-conv-BN  +  skip connection (1x1 conv when the shape changes)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class Featurizer(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128)):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.stem = nn.Sequential(nn.Conv2d(3, c0, 3, 2, 1, bias=False), nn.BatchNorm2d(c0), nn.ReLU(inplace=True))
        self.layer1 = ResidualBlock(c0, c1, stride=2)
        self.layer2 = ResidualBlock(c1, c2, stride=2)
        self.layer3 = ResidualBlock(c2, c3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_features = c3

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(self.stem(x))))
        return self.pool(x).flatten(1)


class FabricNet(nn.Module):
    def __init__(self, n_classes=len(config.CLASSES), dropout=config.DROPOUT):
        super().__init__()
        self.featurizer = Featurizer()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.featurizer.out_features, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.featurizer(x))


def build_model(device="cpu"):
    return FabricNet().to(device)


if __name__ == "__main__":
    m = build_model()
    print(m)
    print("parameters:", sum(p.numel() for p in m.parameters()))
    print(m(torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE)).shape)
