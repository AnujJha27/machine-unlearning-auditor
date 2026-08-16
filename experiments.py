"""Deletion-method benchmark with an oracle for each generated dataset."""

import json
from pathlib import Path

import numpy as np

from auditor import audit, make_data, train


def benchmark(seeds: range = range(10)) -> dict[str, dict[str, float]]:
    collected = {"no_op": [], "fine_tune": []}
    for seed in seeds:
        x, y, deleted, test_x, test_y = make_data(seed)
        retained = np.ones(len(x), dtype=bool)
        retained[deleted] = False
        original = train(x, y)
        oracle = train(x[retained], y[retained])
        for name, model in (("no_op", original), ("fine_tune", train(x[retained], y[retained], original, 50))):
            collected[name].append(audit(model, oracle, x, y, deleted, test_x, test_y))
    return {name: {metric: float(np.mean([row[metric] for row in rows])) for metric in rows[0]} for name, rows in collected.items()}


def save(results: dict[str, dict[str, float]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, indent=2) + "\n")
