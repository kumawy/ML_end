"""Regression checks on real Kolesa listings, never generated project data."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from unsupervised import run_unsupervised


class UnsupervisedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "kolesa_cars_2025.csv"
        raw = pd.read_csv(source, encoding="utf-8-sig")
        frame = raw.loc[raw["condition"].eq("used")].head(140).copy()
        frame["ad_id"] = frame["link"].str.extract(r"/show/(\d+)")[0]
        frame = frame.drop_duplicates("ad_id")
        text_engine = pd.to_numeric(
            frame["desc"].str.extract(r"(\d+(?:\.\d+)?)\s*л(?:,|\s)")[0],
            errors="coerce",
        )
        frame["engine_liters_clean"] = frame["engine_liters"].combine_first(text_engine)
        frame["log_km"] = np.log1p(frame["mileage_km"])
        cls.real_frame = frame

    def test_prices_cannot_change_clusters_or_pca(self):
        # Adding price to either input matrix would change these outputs.
        changed = self.real_frame.copy()
        changed["price"] = changed["price"].iloc[::-1].to_numpy() * 100
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            report = run_unsupervised(self.real_frame, Path(first))
            altered_report = run_unsupervised(changed, Path(second))
            original = pd.read_csv(Path(first) / "cluster_assignments.csv")
            altered = pd.read_csv(Path(second) / "cluster_assignments.csv")
            np.testing.assert_array_equal(original["cluster"], altered["cluster"])
            np.testing.assert_allclose(original[["PC1", "PC2", "PC3"]], altered[["PC1", "PC2", "PC3"]])
            self.assertEqual(report["selected_k"], altered_report["selected_k"])
            self.assertEqual(report["pca"], altered_report["pca"])
            self.assertNotEqual(report["cluster_profiles"], altered_report["cluster_profiles"])
            self.assertEqual(sum(row["count"] for row in report["cluster_profiles"]), len(self.real_frame))
            json.dumps(report, allow_nan=False)

    def test_missing_numeric_values_produce_finite_outputs(self):
        # Omitting imputation or accidentally fitting on complete cases would fail.
        frame = self.real_frame.copy()
        for feature, offset in [("year", 0), ("log_km", 1), ("engine_liters_clean", 2)]:
            frame.loc[frame.index[offset::7], feature] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            report = run_unsupervised(frame, Path(directory))
            assignments = pd.read_csv(Path(directory) / "cluster_assignments.csv")
            centroids = pd.read_csv(Path(directory) / "cluster_centroids_scaled.csv")
            self.assertEqual(len(assignments), len(frame))
            self.assertTrue(np.isfinite(assignments[["PC1", "PC2", "PC3"]]).all().all())
            self.assertTrue(np.isfinite(centroids.drop(columns="cluster")).all().all())
            self.assertEqual([r["k"] for r in report["selection_scores"]], list(range(2, 9)))
            self.assertAlmostEqual(sum(report["pca"]["explained_variance_ratio"]), 1.0)
            self.assertEqual(report["selected_k"], max(report["selection_scores"], key=lambda row: row["silhouette"])["k"])
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
