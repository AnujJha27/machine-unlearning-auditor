"""Neural checkpoints used by the image-unlearning experiment."""

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 14 * 14, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, classes))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))
