# tests/test_tabular_metric_feature_order.py
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from xai_evals.metrics import ExplanationMetricsTabular


def ranked(features, attributions):
    """Same shape as SHAPExplainer.explain: rows sorted by absolute attribution."""
    frame = pd.DataFrame({'Feature': features, 'Attribution': attributions})
    return frame.sort_values(by='Attribution', key=abs, ascending=False)


class TestTabularMetricFeatureOrder(unittest.TestCase):
    def setUp(self):
        self.features = ['f0', 'f1', 'f2', 'f3']
        self.coefficients = np.array([0.0, 0.0, 1.0, 4.0])
        self.metrics = ExplanationMetricsTabular.__new__(ExplanationMetricsTabular)
        self.metrics.features = self.features
        self.metrics.X_test = np.ones((1, 4))
        self.metrics.start_idx = 0
        self.metrics.end_idx = 1
        self.metrics._predict = lambda X: X @ self.coefficients
        self.attributions = [ranked(self.features, self.coefficients)]

    def test_comprehensiveness_masks_the_top_features(self):
        self.assertAlmostEqual(self.metrics._comprehensiveness(self.attributions, k=1), 4.0)

    def test_sufficiency_keeps_the_top_features(self):
        self.assertAlmostEqual(self.metrics._sufficiency(self.attributions, k=1), -1.0)

    def test_faithfulness_pairs_each_feature_with_its_own_attribution(self):
        self.assertAlmostEqual(self.metrics._faithfulness_correlation(self.attributions), 1.0)

    def test_sensitivity_compares_each_feature_with_itself(self):
        explainer = MagicMock()
        explainer.explain.return_value = ranked(self.features, [0.0, 0.0, 4.0, 1.0])
        self.metrics.explainer = explainer
        self.assertAlmostEqual(self.metrics._sensitivity(self.attributions), np.sqrt(18))

    def test_feature_names_with_spaces_match_renamed_rows(self):
        self.metrics.features = ['f 0', 'f 1', 'f 2', 'f 3']
        renamed = ranked(['f_0', 'f_1', 'f_2', 'f_3'], self.coefficients)
        self.assertEqual(
            self.metrics._feature_attributions(renamed).tolist(), [0.0, 0.0, 1.0, 4.0]
        )

    def test_features_missing_from_the_explanation_count_as_zero(self):
        partial = pd.DataFrame({'Feature': ['f3', 'f2'], 'Attribution': [4.0, 1.0]})
        self.assertEqual(
            self.metrics._feature_attributions(partial).tolist(), [0.0, 0.0, 1.0, 4.0]
        )


if __name__ == "__main__":
    unittest.main()
