"""Native-category preprocessing and original-KZT prediction contracts."""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]


class CatBoostTests(unittest.TestCase):
    def load_module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("catboost_model"),
            "Implement native-categorical CatBoost before these contracts can pass.",
        )
        return importlib.import_module("catboost_model")

    def test_predict_inverts_log_target_and_returns_finite_nonnegative_prices(self):
        # Missing expm1, or skipping the overflow/negative guard, breaks this contract.
        module = self.load_module()
        raw = pd.DataFrame({"year": [2000.0, 2020.0], "brand": ["a", None]})
        for transformed, constant, expected in [
            (True, np.log(1000.0), 999.0),
            (False, 999.0, 999.0),
            (True, -100.0, 0.0),
            (False, -100.0, 0.0),
        ]:
            estimator = DummyRegressor(strategy="constant", constant=constant).fit(raw, [1, 2])
            wrapped = module.CatBoostPriceRegressor(
                estimator, list(raw), ["year"], ["brand"], transformed
            )
            np.testing.assert_allclose(wrapped.predict(raw), [expected, expected], rtol=1e-12)
        for transformed in (True, False):
            for extreme in (np.inf, -np.inf, np.nan, 1e100):
                estimator = DummyRegressor(strategy="constant", constant=0.0).fit(raw, [1, 2])
                # Exercise the prediction boundary, including estimator overflow;
                # sklearn deliberately rejects nonfinite constructor constants.
                estimator.constant_[:] = extreme
                wrapped = module.CatBoostPriceRegressor(
                    estimator, list(raw), ["year"], ["brand"], transformed
                )
                predictions = wrapped.predict(raw)
                self.assertTrue(np.isfinite(predictions).all())
                self.assertTrue((predictions >= 0).all())

    def test_native_categories_preserve_missing_numeric_values_without_one_hot_encoding(self):
        # Replacing categorical missingness by float NaN, or imputing numeric
        # missingness, violates the estimator's native-feature interface.
        module = self.load_module()
        import data
        frame, _ = data.load_data(ROOT)
        raw = frame.loc[:, data.FEATURES].copy()
        # These are existing source rows; deliberately exercise the supported
        # missing-value representations rather than relying on current cleaning.
        raw.loc[raw.index[:3], data.CATEGORICAL[0]] = [None, pd.NA, np.nan]
        original = raw.copy(deep=True)
        prepared = module.prepare_features(raw, list(raw), data.NUMERIC, data.CATEGORICAL)
        self.assertEqual(list(prepared), list(raw))
        self.assertEqual(prepared.shape, raw.shape)
        for column in data.CATEGORICAL:
            self.assertTrue(prepared[column].map(lambda value: isinstance(value, str)).all())
            self.assertTrue(prepared.loc[raw[column].isna(), column].eq("__missing__").all())
        for column in data.NUMERIC:
            np.testing.assert_array_equal(prepared[column].isna(), raw[column].isna())
        self.assertGreater(int(prepared[data.NUMERIC].isna().sum().sum()), 0)
        self.assertGreater(int(raw[data.CATEGORICAL].isna().sum().sum()), 0)
        pd.testing.assert_frame_equal(raw, original)

    def test_raw_frame_predictions_follow_training_column_order(self):
        # A wrapper forwarding caller column order would change or reject predictions.
        module = self.load_module()
        from catboost import CatBoostRegressor
        import data
        frame, _ = data.load_data(ROOT)
        raw = frame.loc[:, data.FEATURES].iloc[:96]
        prepared = module.prepare_features(raw, list(raw), data.NUMERIC, data.CATEGORICAL)
        estimator = CatBoostRegressor(
            iterations=12, depth=3, loss_function="MAE", thread_count=2,
            verbose=False, allow_writing_files=False, random_seed=42,
        ).fit(prepared, frame.price.iloc[:96], cat_features=data.CATEGORICAL)
        wrapped = module.CatBoostPriceRegressor(
            estimator, list(raw), list(data.NUMERIC), list(data.CATEGORICAL), False
        )
        np.testing.assert_allclose(
            wrapped.predict(raw.loc[:, list(reversed(raw.columns))]),
            estimator.predict(prepared), rtol=1e-12,
        )

    def test_log_target_checkpoint_metric_measures_absolute_errors_in_kzt(self):
        # Measuring errors in log space would not produce this hand-computed MAE.
        module = self.load_module()
        metric = module.LogTargetMAE()
        error, weight = metric.evaluate(
            [np.log1p([200.0, 900.0])], np.log1p([100.0, 1000.0]), None
        )
        self.assertAlmostEqual(metric.get_final_error(error, weight), 100.0, places=9)
        self.assertFalse(metric.is_max_optimal())


if __name__ == "__main__":
    unittest.main()
