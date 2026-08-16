"""Aggregate independent auditor runs into reportable mean ± standard deviation metrics."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("inputs", nargs="+"); parser.add_argument("--output", default="results/pilot_summary.json")
    args = parser.parse_args(); runs = [json.loads(Path(path).read_text()) for path in args.inputs]
    methods = ("original", "retrain_oracle", "retain_finetune", "scrub")
    metrics = ("test_accuracy", "canary_trigger_confidence", "membership_auc", "oracle_output_l1", "oracle_representation_cka")
    summary = {method: {metric: {"mean": mean([run[method][metric] for run in runs]), "std": stdev([run[method][metric] for run in runs])} for metric in metrics} for method in methods}
    Path(args.output).write_text(json.dumps({"n_runs": len(runs), "summary": summary}, indent=2) + "\n")
    print(json.dumps({"n_runs": len(runs), "summary": summary}, indent=2))


if __name__ == "__main__": main()
