"""Real neural unlearning experiment on a deterministic MNIST split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from audit_metrics import cka, collect, membership_auc
from models import SmallCNN
from unlearning_methods import retain_finetune, scrub


def loader(indices: list[int], train: bool, batch_size: int) -> DataLoader:
    data = datasets.MNIST("data", train=train, download=True, transform=transforms.ToTensor())
    return DataLoader(Subset(data, indices), batch_size=batch_size, shuffle=train, num_workers=0)


def fit(model: nn.Module, batches: DataLoader, device: torch.device, epochs: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for x, y in batches:
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x.to(device)), y.to(device)).backward()
            optimizer.step()


@torch.no_grad()
def evaluate(model: nn.Module, batches: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval(); correct = total = 0; confidence = 0.0
    for x, y in batches:
        probabilities = model(x.to(device)).softmax(-1)
        correct += int((probabilities.argmax(-1).cpu() == y).sum()); total += len(y)
        confidence += float(probabilities.max(-1).values.sum())
    return correct / total, confidence / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--forget", type=int, default=500)
    parser.add_argument("--output", default="results/mnist_audit.json")
    args = parser.parse_args()
    assert 0 < args.forget < args.limit
    torch.manual_seed(7); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = list(range(args.limit)); forget, retain = train[:args.forget], train[args.forget:]
    test = list(range(2_000))
    original = SmallCNN().to(device); fit(original, loader(train, True, 128), device, args.epochs)
    oracle = SmallCNN().to(device); fit(oracle, loader(retain, True, 128), device, args.epochs)
    fine_tuned = SmallCNN().to(device); fine_tuned.load_state_dict(original.state_dict())
    fine_tuned = retain_finetune(fine_tuned, loader(retain, True, 128), device)
    scrubbed = SmallCNN().to(device); scrubbed.load_state_dict(original.state_dict())
    scrubbed = scrub(scrubbed, loader(retain, True, 128), loader(forget, True, 128), device)
    records = {}
    oracle_probabilities, oracle_representation, _ = collect(oracle, loader(test, False, 256), device)
    _, _, nonmember_loss = collect(original, loader(test, False, 256), device)
    for name, model in {"original": original, "retrain_oracle": oracle, "retain_finetune": fine_tuned, "scrub": scrubbed}.items():
        test_accuracy, _ = evaluate(model, loader(test, False, 256), device)
        forget_accuracy, forget_confidence = evaluate(model, loader(forget, True, 256), device)
        probabilities, representation, forget_loss = collect(model, loader(forget, True, 256), device)
        test_probabilities, test_representation, _ = collect(model, loader(test, False, 256), device)
        records[name] = {
            "test_accuracy": test_accuracy, "forgotten_accuracy": forget_accuracy, "forgotten_confidence": forget_confidence,
            "membership_auc": membership_auc(forget_loss, nonmember_loss),
            "oracle_output_l1": float((test_probabilities - oracle_probabilities).abs().mean()),
            "oracle_representation_cka": cka(test_representation, oracle_representation),
            "oracle_parameter_distance": float(sum((a - b).norm().item() for a, b in zip(model.parameters(), oracle.parameters()))),
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
