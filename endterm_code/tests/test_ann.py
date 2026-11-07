"""Behavior checks for KZT prediction conversion and validation-only checkpointing."""
from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


class ANNTests(unittest.TestCase):
    def load_ann(self):
        self.assertIsNotNone(
            importlib.util.find_spec("ann"),
            "Implement the ANN module before its prediction contract can pass.",
        )
        return importlib.import_module("ann")

    def test_predict_converts_standardized_log_target_to_finite_nonnegative_kzt(self):
        # Omitting either inverse scaling or expm1 must break the literal 999/9999 checks.
        ann = self.load_ann()
        x = np.zeros((2, 1))
        scaler = StandardScaler().fit(np.log1p([99.0, 9999.0]).reshape(-1, 1))
        for standardized, expected in [(0.0, 999.0), (1.0, 9999.0), (-100.0, 0.0)]:
            estimator = DummyRegressor(strategy="constant", constant=standardized).fit(x, [0, 0])
            model = ann.ANNRegressor(estimator, scaler)
            np.testing.assert_allclose(model.predict(x), [expected, expected], rtol=1e-10)
        extreme = DummyRegressor(strategy="constant", constant=1e8).fit(x, [0, 0])
        predictions = ann.ANNRegressor(extreme, scaler).predict(x)
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())

    def test_returned_model_restores_best_validation_mae_checkpoint(self):
        # Returning the last epoch, selecting by train loss, or fitting the target
        # scaler on validation labels must fail these observable contract checks.
        ann = self.load_ann()
        x_train = np.column_stack([np.tile([-1.0, 1.0], 48), np.zeros(96)])
        y_train = np.tile([99.0, 9999.0], 48)
        x_val = np.column_stack([np.tile([-1.0, 1.0], 8), np.zeros(16)])
        y_val = np.tile([9999.0, 99.0], 8)
        with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=1):
            model, report = ann.train_ann(x_train, y_train, x_val, y_val, temporary)
            json.dumps(report, allow_nan=False)
            history = pd.read_csv(Path(temporary) / "ann_history.csv")
            measured_mae = float(np.mean(np.abs(y_val - model.predict(x_val))))
            self.assertAlmostEqual(measured_mae, float(history["val_mae_kzt"].min()), places=7)
            self.assertAlmostEqual(measured_mae, report["best_validation_mae_kzt"], places=7)
            self.assertAlmostEqual(report["target_scaler"]["mean_log1p"], np.log(1000.0), places=10)
            self.assertAlmostEqual(report["target_scaler"]["scale_log1p"], np.log(10.0), places=10)
            self.assertEqual(report["target_scaler"]["n_samples_seen"], len(y_train))
            self.assertEqual(len(report["candidates"]), 2)
            self.assertTrue(any(row["best_epoch"] < row["epochs_run"] for row in report["candidates"]))
            self.assertEqual((Path(temporary) / "ann_history.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
