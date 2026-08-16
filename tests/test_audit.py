import sys
from pathlib import Path
import unittest
import importlib.util

root = Path(__file__).parents[1]
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("unlearning_experiments", root / "experiments.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
benchmark = module.benchmark


class UnlearningTest(unittest.TestCase):
    def test_fine_tuning_moves_toward_oracle(self) -> None:
        results = benchmark(range(2))
        self.assertLess(results["fine_tune"]["oracle_divergence"], results["no_op"]["oracle_divergence"])
