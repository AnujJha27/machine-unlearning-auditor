# Machine Unlearning Auditor

**Question:** Did a deletion method actually remove the deleted samples' influence?

`auditor.py` trains a binary classifier, deletes a marked subset, and audits no-op and retain-set fine-tuning against a retrained oracle. The report includes prediction divergence, deleted-set membership signal, canary confidence, and retained-task accuracy.

```bash
python3 auditor.py
```
