# tests/test_image_metric_aggregation.py
import os
import pickle
import tempfile
import unittest

from xai_evals.metrics import ExplanationMetricsImage


class TestAggregateCachedResults(unittest.TestCase):
    def setUp(self):
        self.metrics = ExplanationMetricsImage.__new__(ExplanationMetricsImage)
        self.cache_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.cache_dir.cleanup)

    def _aggregate(self, batches):
        cache_files = []
        for idx, batch in enumerate(batches):
            cache_file = os.path.join(self.cache_dir.name, f"batch_{idx}.pkl")
            with open(cache_file, "wb") as f:
                pickle.dump(batch, f)
            cache_files.append(cache_file)
        return self.metrics._aggregate_cached_results(cache_files)

    def test_sub_scores_keep_the_first_batch(self):
        batches = [{"MPRT": {"layer": 0.6}, "Faithfulness": 0.6}] * 3
        results = self._aggregate(batches)
        self.assertAlmostEqual(results["MPRT_layer"], 0.6)
        self.assertAlmostEqual(results["Faithfulness"], 0.6)
        self.assertEqual(results["num_batches"], 3)

    def test_nan_in_one_metric_does_not_change_another(self):
        results = self._aggregate([
            {"Sparseness": 0.6, "Faithfulness": float("nan")},
            {"Sparseness": 0.6, "Faithfulness": 0.2},
            {"Sparseness": 0.6, "Faithfulness": 0.4},
        ])
        self.assertAlmostEqual(results["Sparseness"], 0.6)
        self.assertAlmostEqual(results["Faithfulness"], 0.3)
        self.assertEqual(results["num_batches"], 3)

    def test_nan_scores_across_metrics_do_not_divide_by_zero(self):
        results = self._aggregate([
            {"Sparseness": float("nan"), "Faithfulness": float("nan")},
            {"Sparseness": float("nan"), "Faithfulness": 0.6},
            {"Sparseness": 0.6, "Faithfulness": 0.6},
        ])
        self.assertAlmostEqual(results["Sparseness"], 0.6)
        self.assertAlmostEqual(results["Faithfulness"], 0.6)
        self.assertEqual(results["num_batches"], 3)

    def test_nan_in_one_sub_score_does_not_change_another(self):
        results = self._aggregate([
            {"MPRT": {"layer1": float("nan"), "layer2": 0.5}},
            {"MPRT": {"layer1": 0.2, "layer2": 0.5}},
            {"MPRT": {"layer1": 0.4, "layer2": 0.5}},
        ])
        self.assertAlmostEqual(results["MPRT_layer1"], 0.3)
        self.assertAlmostEqual(results["MPRT_layer2"], 0.5)
        self.assertEqual(results["num_batches"], 3)

    def test_metric_with_no_finite_score_is_left_out(self):
        results = self._aggregate([{"Sparseness": float("nan"), "Faithfulness": 0.6}])
        self.assertNotIn("Sparseness", results)
        self.assertEqual(results["num_batches"], 1)

    def test_cache_files_are_deleted(self):
        self._aggregate([{"Faithfulness": 0.6}])
        self.assertEqual(os.listdir(self.cache_dir.name), [])


if __name__ == "__main__":
    unittest.main()
