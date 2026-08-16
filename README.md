# Machine Unlearning Auditor

Did a model actually forget the training data it was asked to delete?

## Research question

Accuracy after deletion is not evidence of forgetting. This auditor compares candidate deletion methods against a retrain-from-scratch oracle, then looks for residual deleted-data influence with independent probes.

## Current protocol

For each seeded dataset, the experiment trains an original binary classifier, removes a marked canary subset, trains a retain-set oracle, and evaluates two candidates: no deletion and retain-set fine-tuning. The audit reports:

- mean absolute prediction divergence from the retrained oracle;
- parameter distance from the oracle;
- a loss-based membership signal on deleted examples;
- deleted-canary confidence;
- retained-set utility.

The crucial result is diagnostic rather than celebratory: fine-tuning can reduce oracle divergence while leaving canary confidence high. Passing a utility or output-similarity check alone is therefore insufficient.

## Reproduce

```bash
python3 run.py
python3 -m unittest tests/test_audit.py
```

The multi-seed summary is saved to `results/audit.json`.

## Layout

```text
auditor.py      data generation, logistic model, deletion methods, probes
experiments.py  multi-seed oracle comparison and metric aggregation
run.py          reproducible experiment entry point
tests/          oracle-convergence regression check
```

## Next research increment

Replace the linear model with a real image or text classifier; add exact retraining, SCRUB-style distillation, influence-based deletion, and shadow-model membership attacks. Keep the existing oracle, canary, membership, and utility reports as non-negotiable acceptance criteria.
