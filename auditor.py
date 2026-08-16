"""Compare approximate deletion with the retrain-from-scratch oracle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticModel:
    weights: np.ndarray
    bias: float

    def probability(self, x: np.ndarray) -> np.ndarray:
        logits = np.clip(x @ self.weights + self.bias, -30, 30)
        return 1 / (1 + np.exp(-logits))


def train(x: np.ndarray, y: np.ndarray, start: LogisticModel | None = None, steps: int = 1_200) -> LogisticModel:
    model = LogisticModel(np.zeros(x.shape[1]), 0.0) if start is None else LogisticModel(start.weights.copy(), start.bias)
    for _ in range(steps):
        error = model.probability(x) - y
        model.weights -= 0.15 * (x.T @ error / len(x) + 1e-3 * model.weights)
        model.bias -= 0.15 * error.mean()
    return model


def loss(model: LogisticModel, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(model.probability(x), 1e-8, 1 - 1e-8)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def accuracy(model: LogisticModel, x: np.ndarray, y: np.ndarray) -> float:
    return float(((model.probability(x) >= 0.5) == y).mean())


def make_data(seed: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(500, 2))
    y = (x[:, 0] + 0.7 * x[:, 1] > 0).astype(float)
    deleted = np.arange(24)
    # Canary examples are deliberately easy to recognize; they make residual influence visible.
    x[deleted] = rng.normal(loc=(2.5, 2.5), scale=0.08, size=(len(deleted), 2))
    y[deleted] = 1
    test_x = rng.normal(size=(500, 2))
    test_y = (test_x[:, 0] + 0.7 * test_x[:, 1] > 0).astype(float)
    return x, y, deleted, test_x, test_y


def audit(candidate: LogisticModel, oracle: LogisticModel, x: np.ndarray, y: np.ndarray, deleted: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, float]:
    retained = np.ones(len(x), dtype=bool)
    retained[deleted] = False
    held_out_loss = loss(candidate, test_x, test_y).mean()
    member_signal = float(held_out_loss - loss(candidate, x[deleted], y[deleted]).mean())
    return {
        "oracle_divergence": float(np.abs(candidate.probability(test_x) - oracle.probability(test_x)).mean()),
        "parameter_distance": float(np.linalg.norm(candidate.weights - oracle.weights)),
        "membership_signal": member_signal,
        "canary_confidence": float(candidate.probability(x[deleted]).mean()),
        "retain_accuracy": accuracy(candidate, x[retained], y[retained]),
    }


def demo() -> None:
    x, y, deleted, test_x, test_y = make_data()
    retained = np.ones(len(x), dtype=bool)
    retained[deleted] = False
    original = train(x, y)
    oracle = train(x[retained], y[retained])
    finetuned = train(x[retained], y[retained], start=original, steps=50)
    no_op, repaired = (audit(model, oracle, x, y, deleted, test_x, test_y) for model in (original, finetuned))
    assert repaired["oracle_divergence"] < no_op["oracle_divergence"]
    print("method     oracle-divergence  parameter-distance  member-signal  canary-confidence  retain-accuracy")
    for name, result in (("no-op", no_op), ("fine-tune", repaired)):
        print(f"{name:<10} {result['oracle_divergence']:.4f}             {result['parameter_distance']:.4f}              {result['membership_signal']:.4f}         {result['canary_confidence']:.3f}              {result['retain_accuracy']:.1%}")


if __name__ == "__main__":
    demo()
