"""Real boundary checks for the fixed ANN pools and dual-metric selector."""
import importlib
import importlib.util
from inspect import signature
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def tiny_frames():
    """Independent numeric/category fixture, never appended to the project CSV."""
    levels = np.tile([-1., 1.], 24)
    train = pd.DataFrame({
        'year': 2000. + levels, 'log_km': np.tile([1., np.nan, 3., 4.], 12),
        'mileage_missing': np.tile([0., 1., 0., 0.], 12),
        'engine_liters_clean': 2. + levels / 2,
        'engine_missing': 0., 'engine_not_applicable': 0., 'engine_source_conflict': 0.,
        'brand': np.where(levels < 0, 'A', 'B'),
        'model': np.where(levels < 0, 'small', 'large'), 'body': 'sedan',
        'fuel': 'petrol', 'transmission': 'automatic',
        'desc': np.where(levels < 0, 'левый руль, КПП автомат', 'правый руль, КПП автомат'),
        'price': np.tile([99., 9999.], 24), 'ad_id': np.arange(48).astype(str),
    })
    validation = train.iloc[:8].copy()
    validation['year'] = 2010.
    validation['brand'] = 'validation_only_brand'
    validation['log_km'] = 9.
    return train, validation


class RobustANNTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('robust_ann'),
                             'The fixed robust ANN training and selection module is required')
        return importlib.import_module('robust_ann')

    def test_selector_rejects_each_one_sided_gain_and_retains_baseline(self):
        # Removing either strict improvement gate would select one of these.
        module = self.module()
        target = np.full(5, 100.)
        baseline = np.array([125., 125., 125., 100., 100.])
        result = module.select_dual_blend({
            'baseline': baseline, 'mae_only': [121., 121., 121., 100., 100.],
            'hit_only': [200., 100., 100., 100., 100.], 'forest': baseline,
        }, target, 'baseline', ('mae_only', 'hit_only'), 'forest')
        self.assertEqual(result['selected_name'], 'baseline')
        self.assertEqual(result['selected_components'], ['baseline'])
        self.assertEqual(result['weights'], [1.])
        self.assertFalse(any(row['eligible'] for row in result['candidates']))
        self.assertEqual(len(result['candidates']), 9)

    def test_selector_ranks_eligible_candidates_by_normalized_mae_and_miss(self):
        # MAE-only ranking prefers mae_pool (8.4); the prescribed dual score
        # prefers hit_pool (11.4, all five prices within 20 percent).
        module = self.module()
        baseline = np.array([125., 125., 125., 100., 100.])
        result = module.select_dual_blend({
            'baseline': baseline, 'mae_pool': [121., 121., 100., 100., 100.],
            'hit_pool': [119., 119., 119., 100., 100.], 'forest': baseline,
        }, np.full(5, 100.), 'baseline', ('mae_pool', 'hit_pool'), 'forest')
        self.assertEqual(result['selected_components'], ['hit_pool'])
        self.assertEqual(result['weights'], [1.])
        self.assertAlmostEqual(result['validation_mae_kzt'], 11.4)
        self.assertEqual(result['validation_within_20pct'], 1.)
        self.assertAlmostEqual(result['dual_score'], .38)
        self.assertFalse(result['test_data_used'])
        json.dumps(result, allow_nan=False)

    def test_selector_rejects_invalid_targets_predictions_and_blend_weights(self):
        module = self.module()
        predictions = {'base': [125., 100.], 'a': [121., 100.], 'b': [119., 100.], 'rf': [125., 100.]}
        for targets in ([0., 100.], [100., np.nan], []):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                module.select_dual_blend(predictions, targets, 'base', ('a', 'b'), 'rf')
        with self.assertRaises(ValueError):
            module.select_dual_blend({**predictions, 'a': [1.]}, [100., 100.], 'base', ('a', 'b'), 'rf')
        with self.assertRaises(ValueError):
            module.select_dual_blend(predictions, [100., 100.], 'base', ('a', 'b'), 'rf', weights=(0.,))

    def test_optional_baseline_blends_can_win_the_same_strict_dual_selection(self):
        # The best expanded candidate is unavailable in the original grid;
        # reference normalization must still use the original baseline.
        module = self.module()
        self.assertIn('include_baseline_blends', signature(module.select_dual_blend).parameters)
        predictions = {
            'baseline': [140., 70., 115., 115.],
            'pool_a': [115., 150., 55., 85.],
            'pool_b': [105., 140., 95., 150.],
            'forest': [70., 45., 55., 75.],
        }
        arguments = (predictions, [100.] * 4, 'baseline', ('pool_a', 'pool_b'), 'forest')
        original = module.select_dual_blend(*arguments)
        expanded = module.select_dual_blend(*arguments, include_baseline_blends=True)
        self.assertEqual(len(original['candidates']), 9)
        self.assertEqual(original['selected_components'], ['pool_b', 'forest'])
        self.assertEqual(original['weights'], [.5, .5])
        self.assertAlmostEqual(original['dual_score'], .5375)
        self.assertEqual(len(expanded['candidates']), 15)
        self.assertEqual(expanded['selected_components'], ['pool_a', 'baseline'])
        self.assertEqual(expanded['weights'], [.25, .75])
        self.assertAlmostEqual(expanded['validation_mae_kzt'], 12.8125)
        self.assertEqual(expanded['validation_within_20pct'], .75)
        self.assertAlmostEqual(expanded['dual_score'], .50625)
        self.assertEqual(expanded['baseline_validation_metrics']['mae_kzt'], 25.)
        self.assertEqual(expanded['baseline_validation_metrics']['within_20pct'], .5)

    def test_usage_preprocessing_combines_declared_inputs_with_training_only_statistics(self):
        # Using validation values to impute/scale, ignoring an opted-in input,
        # or silently including undeclared inputs breaks these observations.
        module = self.module()
        self.assertIn('additional_numeric', signature(module.make_robust_preprocessor).parameters)
        train, validation = tiny_frames()
        train['log_vehicle_age'] = np.tile([1., 3., np.nan, 5.], 12)
        train['log_annual_mileage_proxy'] = np.tile([10., 20., 30., np.nan], 12)
        validation['log_vehicle_age'] = 1000.
        validation['log_annual_mileage_proxy'] = 2000.
        training = module.add_structured_features(train)
        query = module.add_structured_features(validation)
        originals = (module.ROBUST_NUMERIC, module.ROBUST_CATEGORICAL, module.ROBUST_FEATURES)
        prep = module.make_robust_preprocessor(
            additional_numeric=('log_vehicle_age', 'log_annual_mileage_proxy')).fit(training)
        imputer = prep.named_transformers_['num'].named_steps['impute']
        scaler = prep.named_transformers_['num'].named_steps['scale']
        np.testing.assert_array_equal(imputer.statistics_[-2:], [3., 20.])
        np.testing.assert_allclose(scaler.mean_[-2:], [3., 20.])
        np.testing.assert_allclose(scaler.scale_[-2:], [np.sqrt(2.), np.sqrt(50.)])
        encoded = prep.transform(query)
        changed = query.copy()
        changed['log_vehicle_age'] = 1001.
        changed['log_annual_mileage_proxy'] = 2002.
        self.assertFalse(np.array_equal(encoded, prep.transform(changed)))
        np.testing.assert_array_equal(imputer.statistics_[-2:], [3., 20.])
        default = module.make_robust_preprocessor().fit(training)
        np.testing.assert_array_equal(default.transform(query), default.transform(changed))
        self.assertEqual(encoded.shape[1], default.transform(query).shape[1] + 2)
        self.assertEqual(originals,
                         (module.ROBUST_NUMERIC, module.ROBUST_CATEGORICAL, module.ROBUST_FEATURES))

    def test_usage_input_allowlist_rejects_targets_unknowns_and_duplicates_before_fitting(self):
        module = self.module()
        self.assertIn('additional_numeric', signature(module.make_robust_preprocessor).parameters)
        self.assertIn('additional_numeric', signature(module.fit_robust_pool).parameters)
        train, validation = tiny_frames()
        for extras in [('price',), ('ad_id',), ('year',), ('unknown_feature',),
                       ('log_vehicle_age', 'log_vehicle_age'), None, 'log_vehicle_age']:
            with self.subTest(extras=extras):
                with self.assertRaises(ValueError):
                    module.make_robust_preprocessor(additional_numeric=extras)
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaises(ValueError):
                        module.fit_robust_pool(
                            train, train.price, validation, validation.price,
                            np.full(8, 2000.), Path(temporary) / 'must_not_fit',
                            additional_numeric=extras)
                    self.assertFalse((Path(temporary) / 'must_not_fit').exists())

    def test_usage_pool_requires_declared_columns_and_records_its_actual_numeric_contract(self):
        module = self.module()
        self.assertIn('additional_numeric', signature(module.fit_robust_pool).parameters)
        train, validation = tiny_frames()
        train['log_vehicle_age'] = np.tile([1., 3., np.nan, 5.], 12)
        train['log_annual_mileage_proxy'] = np.tile([10., 20., 30., np.nan], 12)
        validation['log_vehicle_age'] = 1000.
        validation['log_annual_mileage_proxy'] = 2000.
        extras = ('log_vehicle_age', 'log_annual_mileage_proxy')
        config = module._PoolConfig(seeds=(11,), hidden_layer_sizes=(4,),
                                    batch_size=16, max_epochs=3, patience=2)
        with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=1):
            for missing_from_training in [True, False]:
                incomplete_train = train.drop(columns=extras[0]) if missing_from_training else train
                incomplete_validation = validation if missing_from_training else validation.drop(columns=extras[1])
                with self.assertRaisesRegex(ValueError, 'Missing robust ANN predictors'):
                    module.fit_robust_pool(
                        incomplete_train, train.price, incomplete_validation, validation.price,
                        np.full(8, 2000.), Path(temporary) / 'incomplete',
                        additional_numeric=extras, _config=config)
            models, report = module.fit_robust_pool(
                train, train.price, validation, np.tile([9999., 99.], 4),
                np.full(8, 2000.), temporary, additional_numeric=extras, _config=config)
            self.assertEqual(set(models), {'ANN pool MAE', 'ANN pool dual'})
            self.assertEqual(report['additional_numeric_inputs'], list(extras))
            self.assertEqual(report['numeric_inputs'], [*module.ROBUST_NUMERIC, *extras])
            self.assertEqual(report['feature_set'], 'corrected_steering_plus_age_usage')
            history = pd.read_csv(Path(temporary) / report['artifacts']['history_csv'])
            self.assertEqual(set(history.feature_set), {report['feature_set']})
            enhanced = module.add_structured_features(validation)
            for name, pool in models.items():
                prediction = pool.predict({'enhanced': enhanced})
                self.assertTrue(np.isfinite(prediction).all())
                saved = joblib.load(Path(temporary) / report['artifacts']['models'][name])
                np.testing.assert_array_equal(prediction, saved.predict({'enhanced': enhanced}))
                prep = pool.components[0].preprocessor
                np.testing.assert_array_equal(
                    prep.named_transformers_['num'].named_steps['impute'].statistics_[-2:], [3., 20.])

    def test_training_preserves_train_only_transforms_checkpoints_and_serialization(self):
        # Validation-only numeric extremes/categories must not affect medians,
        # scaling or learned vocabulary; returning the final network rather than
        # a best checkpoint must disagree with recorded member validation MAE.
        module = self.module()
        train, validation = tiny_frames()
        before_train, before_validation = train.copy(deep=True), validation.copy(deep=True)
        y_train = train.price.to_numpy()
        y_val = np.tile([9999., 99.], 4)
        config = module._PoolConfig(seeds=(11, 23), hidden_layer_sizes=(8, 4),
                                    batch_size=16, max_epochs=12, patience=3)
        with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=1):
            models, report = module.fit_robust_pool(train, y_train, validation, y_val,
                                                   np.full(8, 2000.), temporary, _config=config)
            self.assertEqual(set(models), {'ANN pool MAE', 'ANN pool dual'})
            pd.testing.assert_frame_equal(train, before_train)
            pd.testing.assert_frame_equal(validation, before_validation)
            json.dumps(report, allow_nan=False)
            self.assertEqual(report['target_scaler']['n_samples_seen'], 48)
            self.assertAlmostEqual(report['target_scaler']['mean_log1p'], np.log(1000.), places=10)
            self.assertAlmostEqual(report['target_scaler']['scale_log1p'], np.log(10.), places=10)
            self.assertFalse(report['test_data_used'])
            history = pd.read_csv(Path(temporary) / report['artifacts']['history_csv'])
            required = {'epoch', 'seed', 'train_scaled_log_mse', 'val_scaled_log_mse',
                        'train_mae_kzt', 'val_mae_kzt', 'val_within_20pct', 'dual_score',
                        'best_mae', 'best_dual'}
            self.assertTrue(required.issubset(history.columns))
            self.assertTrue(np.isfinite(history[list(required - {'best_mae', 'best_dual'})]).all().all())
            self.assertEqual((Path(temporary) / report['artifacts']['history_plot']).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            enhanced = module.add_structured_features(validation)
            for name, pool in models.items():
                self.assertEqual(pool.weights, [.5, .5])
                prediction = pool.predict({'enhanced': enhanced})
                self.assertTrue(np.isfinite(prediction).all())
                self.assertTrue((prediction >= 0.).all())
                saved = Path(temporary) / (name.replace(' ', '_') + '.joblib')
                joblib.dump(pool, saved)
                np.testing.assert_array_equal(prediction, joblib.load(saved).predict({'enhanced': enhanced}))
                for component in pool.components:
                    prep = component.preprocessor
                    imputer = prep.named_transformers_['num'].named_steps['impute']
                    self.assertAlmostEqual(imputer.statistics_[0], 2000.)
                    self.assertAlmostEqual(imputer.statistics_[1], 3.)
                    encoder = prep.named_transformers_['cat'].named_steps['encode']
                    self.assertNotIn('validation_only_brand', encoder.categories_[0])
                    self.assertEqual(component.representation, 'enhanced')
                    measured = np.mean(np.abs(y_val - component.predict({'enhanced': enhanced})))
                    member = next(row for row in report['members'] if row['seed'] == component.estimator.estimator.random_state)
                    metric = 'mae_kzt' if name == 'ANN pool MAE' else 'dual_score'
                    checkpoint = member['checkpoints'][metric]
                    self.assertAlmostEqual(measured, checkpoint['validation_mae_kzt'], places=7)
                    rows = history[history.seed.eq(member['seed'])]
                    column = 'val_mae_kzt' if metric == 'mae_kzt' else 'dual_score'
                    best_epoch = int(rows.loc[rows[column].idxmin(), 'epoch'])
                    self.assertEqual(checkpoint['best_epoch'], best_epoch)
            with self.assertRaises((KeyError, ValueError)):
                models['ANN pool dual'].predict({'enhanced': validation})


if __name__ == '__main__':
    unittest.main()
