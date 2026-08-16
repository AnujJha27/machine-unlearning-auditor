"""Audits that deliberately test different notions of forgetting."""

import torch
from torch import nn


@torch.no_grad()
def collect(model: nn.Module, batches, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval(); probabilities = []; representations = []; losses = []
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probabilities.append(logits.softmax(-1).cpu())
        representations.append(model.encode(x).cpu())
        losses.append(nn.functional.cross_entropy(logits, y, reduction="none").cpu())
    return torch.cat(probabilities), torch.cat(representations), torch.cat(losses)


def cka(left: torch.Tensor, right: torch.Tensor) -> float:
    """Linear CKA: representation similarity invariant to orthogonal re-basing."""
    left, right = left - left.mean(0), right - right.mean(0)
    cross = (left.T @ right).square().sum()
    normalizer = ((left.T @ left).square().sum() * (right.T @ right).square().sum()).sqrt()
    return float(cross / normalizer.clamp_min(1e-12))


def membership_auc(member_loss: torch.Tensor, nonmember_loss: torch.Tensor) -> float:
    """Exact rank AUC for a loss-threshold membership attack; no sklearn dependency."""
    scores = torch.cat((-member_loss, -nonmember_loss))
    labels = torch.cat((torch.ones(len(member_loss)), torch.zeros(len(nonmember_loss))))
    ranks = scores.argsort().argsort().float() + 1
    positive_rank_sum = ranks[labels.bool()].sum()
    return float((positive_rank_sum - len(member_loss) * (len(member_loss) + 1) / 2) / (len(member_loss) * len(nonmember_loss)))
