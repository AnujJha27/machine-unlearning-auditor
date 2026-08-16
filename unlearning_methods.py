"""Approximate deletion methods compared against the retraining oracle."""

from itertools import cycle

import torch
from torch import nn


def retain_finetune(model: nn.Module, retain_loader, device: torch.device, epochs: int = 1) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.train()
    for _ in range(epochs):
        for x, y in retain_loader:
            optimizer.zero_grad(); nn.functional.cross_entropy(model(x.to(device)), y.to(device)).backward(); optimizer.step()
    return model


def scrub(model: nn.Module, retain_loader, forget_loader, device: torch.device, epochs: int = 1, forget_weight: float = 0.5) -> nn.Module:
    """Retain learning while ascending deleted-example loss; tune weight on utility/leakage frontier."""
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.train()
    for _ in range(epochs):
        for (retain_x, retain_y), (forget_x, forget_y) in zip(retain_loader, cycle(forget_loader)):
            optimizer.zero_grad()
            objective = nn.functional.cross_entropy(model(retain_x.to(device)), retain_y.to(device))
            objective -= forget_weight * nn.functional.cross_entropy(model(forget_x.to(device)), forget_y.to(device))
            objective.backward(); optimizer.step()
    return model
