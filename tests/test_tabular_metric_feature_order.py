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

    def test_features_missing_from_the_explanation_are_logged(self):
        partial = pd.DataFrame({'Feature': ['f3', 'f2'], 'Attribution': [4.0, 1.0]})
        with self.assertLogs('xai_evals.metrics', level='WARNING') as logs:
            self.metrics._feature_attributions(partial)
        self.assertIn('2 of 4 features are missing', logs.output[0])

    def test_explanation_with_no_matching_feature_raises(self):
        other = pd.DataFrame({'Feature': ['p', 'q'], 'Attribution': [1.0, 2.0]})
        self.metrics.features = ['x', 'y']
        with self.assertRaises(ValueError) as caught:
            self.metrics._feature_attributions(other)
        message = str(caught.exception)
        self.assertIn("'p'", message)
        self.assertIn("'x'", message)

    def test_metrics_fail_on_an_unmatched_explanation_instead_of_scoring_zeros(self):
        other = pd.DataFrame({'Feature': ['p', 'q', 'r', 's'], 'Attribution': self.coefficients})
        with self.assertRaises(ValueError):
            self.metrics._comprehensiveness([other], k=1)

    def test_nan_attribution_is_not_turned_into_zero(self):
        explanation = pd.DataFrame(
            {'Feature': self.features, 'Attribution': [0.0, np.nan, 1.0, 4.0]}
        )
        values = self.metrics._feature_attributions(explanation)
        self.assertTrue(np.isnan(values[1]))
        self.assertEqual(values[[0, 2, 3]].tolist(), [0.0, 1.0, 4.0])

    def test_names_that_collide_after_renaming_raise(self):
        self.metrics.features = ['a b', 'a_b']
        explanation = pd.DataFrame({'Feature': ['a b', 'a_b'], 'Attribution': [1.0, 2.0]})
        with self.assertRaisesRegex(ValueError, 'must be unique'):
            self.metrics._feature_attributions(explanation)

    def test_duplicate_feature_names_raise_a_clear_error(self):
        self.metrics.features = ['b', 'a', 'c']
        explanation = pd.DataFrame({'Feature': ['a', 'b', 'a'], 'Attribution': [1.0, 2.0, 3.0]})
        with self.assertRaisesRegex(ValueError, 'must be unique'):
            self.metrics._feature_attributions(explanation)

    def test_duplicate_column_names_raise_a_clear_error(self):
        self.metrics.features = ['a', 'b', 'a']
        explanation = pd.DataFrame({'Feature': ['a', 'b', 'a'], 'Attribution': [1.0, 2.0, 3.0]})
        with self.assertRaisesRegex(ValueError, 'must be unique'):
            self.metrics._feature_attributions(explanation)


class TestNoiseSeed(unittest.TestCase):
    def _metrics(self, random_state):
        metrics = ExplanationMetricsTabular.__new__(ExplanationMetricsTabular)
        metrics.random_state = random_state
        return metrics

    def test_same_seed_gives_the_same_noise(self):
        first = self._metrics(7)._generate_noise((1, 5))
        second = self._metrics(7)._generate_noise((1, 5))
        np.testing.assert_array_equal(first, second)

    def test_successive_draws_differ(self):
        metrics = self._metrics(7)
        self.assertFalse(np.array_equal(metrics._generate_noise((1, 5)), metrics._generate_noise((1, 5))))

    def test_without_a_seed_runs_differ(self):
        first = self._metrics(None)._generate_noise((1, 5))
        second = self._metrics(None)._generate_noise((1, 5))
        self.assertFalse(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
