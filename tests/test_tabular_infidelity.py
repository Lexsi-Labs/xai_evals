# tests/test_tabular_infidelity.py
import unittest

import numpy as np
import pandas as pd

from xai_evals.metrics import ExplanationMetricsTabular


class TestInfidelity(unittest.TestCase):
    def setUp(self):
        self.coefficients = np.array([1.0, 2.0, 3.0])
        self.metrics = ExplanationMetricsTabular.__new__(ExplanationMetricsTabular)
        self.metrics.features = ['f0', 'f1', 'f2']
        self.metrics.X_test = np.array([[1.0, 2.0, 3.0]])
        self.metrics.start_idx = 0
        self.metrics.end_idx = 1
        self.metrics._predict = lambda X: X @ self.coefficients
        self.metrics._generate_noise = lambda shape, epsilon: np.array([[0.1, 0.2, 0.3]])

    def _infidelity(self, attributions):
        frame = pd.DataFrame({'Feature': self.metrics.features, 'Attribution': attributions})
        return self.metrics._infidelity([frame])

    def test_exact_linear_attributions_score_zero(self):
        self.assertAlmostEqual(self._infidelity([1.0, 2.0, 3.0]), 0.0)

    def test_wrong_attributions_on_later_features_are_penalised(self):
        self.assertAlmostEqual(self._infidelity([1.0, 2.0, 0.0]), 0.81)


if __name__ == "__main__":
    unittest.main()
