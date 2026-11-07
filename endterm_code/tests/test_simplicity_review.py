"""Boundaries and actual tiny-data training for the frozen simplicity review."""
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd


def tiny_frame():
    n = 80
    frame = pd.DataFrame({
        'year': np.tile([2000., 2010., 2018., 2024.], 20),
        'log_km': np.tile([1., np.nan, 3., 4.], 20),
        'mileage_missing': np.tile([0., 1., 0., 0.], 20),
        'engine_liters_clean': 2., 'engine_missing': 0.,
        'engine_not_applicable': 0., 'engine_source_conflict': 0.,
        'brand': np.tile(['A', 'B'], 40), 'model': 'small', 'body': 'sedan',
        'fuel': 'petrol', 'transmission': 'automatic',
        'desc': 'левый руль, КПП автомат', 'mileage_km': 100000.,
        'ad_id': [str(i) for i in range(n)],
        'price': np.tile([1e6, 5e6, 8e6, 15e6], 20),
    }, index=np.arange(100, 180))
    for column in ['dual_fuel_claim', 'fuel_text_conflict', 'restyling_claim',
                   'leather_claim', 'panorama_claim', 'warranty_claim',
                   'power_kw', 'power_missing']:
        frame[column] = 0.
    frame['brand_model'], frame['trim'] = 'A | small', 'not_stated'
    return frame


def decision_fixture():
    baseline = {'n': 100, 'mae_kzt': 100., 'within_20pct_count': 70,
                'affordable': {'n': 50, 'median_ape': .2},
                'bands': {'Q1': {'n': 25, 'within_20pct_count': 12},
                          'Q2': {'n': 25, 'within_20pct_count': 15}}}
    result = {name: deepcopy(baseline) for name in
              ('single_mae', 'single_log', 'single_forest', 'historical_recipe')}
    result['single_mae']['mae_kzt'] = 103.
    result['single_mae']['bands']['Q1']['within_20pct_count'] = 13
    result['single_log']['mae_kzt'] = 102.
    result['single_log']['bands']['Q2']['within_20pct_count'] = 16
    result['single_log']['affordable']['median_ape'] = .19
    result['single_forest']['bands']['Q1']['within_20pct_count'] = 14
    result['single_forest']['affordable']['median_ape'] = .1
    return result


class SimplicityReviewTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('simplicity_review'),
                             'The frozen simplicity experiment runner is required')
        return importlib.import_module('simplicity_review')

    def test_summary_uses_inclusive_band_edges_hit_counts_and_affordable_median(self):
        module = self.module()
        result = module.summarize_predictions(
            [1e6, 3.6e6, 4e6, 6.7e6, 8e6, 11.5e6, 12e6],
            [1.2e6, 4.5e6, 4e6, 6.7e6, 6e6, 11.5e6, 15e6])
        self.assertEqual(result['within_20pct_count'], 4)
        self.assertEqual([result['bands'][key]['n'] for key in ('Q1', 'Q2', 'Q3', 'Q4')],
                         [2, 2, 2, 1])
        self.assertAlmostEqual(result['affordable']['median_ape'], .1)
        empty = module.summarize_predictions([12e6], [12e6])
        self.assertIsNone(empty['affordable']['median_ape'])
        json.dumps(empty, allow_nan=False)
        for y, pred in [([], []), ([0.], [1.]), ([1.], [np.nan]), ([1.], [1., 2.])]:
            with self.subTest(y=y, pred=pred), self.assertRaises(ValueError):
                module.summarize_predictions(y, pred)

    def test_selection_enforces_every_joint_gate_and_prefers_fewer_estimators(self):
        module = self.module()
        self.assertEqual(module.choose_candidate(decision_fixture())['selected_name'], 'single_log')
        mutations = (
            lambda a: a.update(mae_kzt=103.00001),
            lambda a: a.update(within_20pct_count=69),
            lambda a: a['bands']['Q2'].update(within_20pct_count=14),
            lambda a: a['bands']['Q1'].update(within_20pct_count=12),
            lambda a: a['affordable'].update(median_ape=.200001),
        )
        for mutate in mutations:
            rows = decision_fixture()
            rows['single_log'] = deepcopy(rows['historical_recipe'])
            rows['single_forest'] = deepcopy(rows['historical_recipe'])
            mutate(rows['single_mae'])
            with self.subTest(mutate=mutate):
                selection = module.choose_candidate(rows)
                self.assertEqual(selection['selected_name'], 'historical_recipe')
                self.assertEqual(selection['eligible_names'], [])

    def test_fold_plan_covers_each_row_once_without_inner_outer_leakage(self):
        module = self.module()
        frame = tiny_frame()
        plan = module.plan_folds(frame)
        assessed = []
        for fold in plan:
            train, checkpoint, outer = (set(fold[key]) for key in
                ('inner_train_source_rows', 'inner_checkpoint_source_rows', 'outer_assessment_source_rows'))
            self.assertFalse(train & checkpoint or train & outer or checkpoint & outer)
            self.assertEqual(train | checkpoint | outer, set(frame.index))
            assessed.extend(outer)
        self.assertEqual(sorted(assessed), list(frame.index))
        self.assertEqual(plan, module.plan_folds(frame))

    def test_real_shared_trajectory_restores_both_checkpoints_and_serializes(self):
        module = self.module()
        from dataset_features import add_dataset_features
        from structured_features import add_structured_features
        frame = add_structured_features(add_dataset_features(tiny_frame()))
        training, checkpoint = frame.iloc[:64], frame.iloc[64:].copy()
        checkpoint['brand'] = 'checkpoint_only'
        checkpoint['year'] = 2025.
        config = module._ReviewConfig(single_epochs=4, single_patience=2,
                                      reference_epochs=3, reference_patience=2,
                                      pool_epochs=2, pool_patience=1)
        with tempfile.TemporaryDirectory() as temporary:
            models, report = module.fit_single_trajectory(
                training, training.price.to_numpy(), checkpoint,
                checkpoint.price.to_numpy(), Path(temporary), _config=config)
            history = pd.read_csv(Path(temporary) / 'single_history.csv')
            for name, criterion in [('single_mae', 'mae_kzt'), ('single_log', 'mean_absolute_log1p_error')]:
                predictions = models[name].predict({'enhanced': checkpoint})
                actual = (np.mean(np.abs(predictions - checkpoint.price.to_numpy()))
                          if name == 'single_mae' else
                          np.mean(np.abs(np.log1p(predictions) - np.log1p(checkpoint.price.to_numpy()))))
                self.assertAlmostEqual(actual, history[criterion].min(), places=7)
                saved = Path(temporary) / (name + '.joblib')
                joblib.dump(models[name], saved)
                np.testing.assert_array_equal(joblib.load(saved).predict({'enhanced': checkpoint}), predictions)
            scaler = models['single_mae'].components[0].estimator.target_scaler
            self.assertEqual(scaler.n_samples_seen_, 64)
            prep = models['single_mae'].components[0].preprocessor
            self.assertAlmostEqual(prep.named_transformers_['num'].named_steps['scale'].mean_[0], 2013.)
            categories = prep.named_transformers_['cat'].named_steps['encode'].categories_[0]
            self.assertNotIn('checkpoint_only', categories)
            self.assertEqual(report['training_rows'], 64)
            json.dumps(report, allow_nan=False)

    def test_existing_output_is_rejected_without_touching_prior_evidence(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / 'prior_evidence.txt'
            marker.write_text('retain me', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                module.run_review('/a/nonexistent/source', temporary)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'retain me')

    def test_real_fold_rebuilds_historical_recipe_without_retaining_bulk_checkpoints(self):
        module = self.module()
        from dataset_features import add_dataset_features
        from structured_features import add_structured_features
        raw = tiny_frame()
        features = add_structured_features(add_dataset_features(raw.drop(columns='price')))
        fold = {'fold': 1, 'inner_train_source_rows': list(range(100, 164)),
                'inner_checkpoint_source_rows': list(range(164, 172)),
                'outer_assessment_source_rows': list(range(172, 180))}
        config = module._ReviewConfig(single_epochs=2, single_patience=1,
                                      reference_epochs=2, reference_patience=1,
                                      pool_epochs=1, pool_patience=1)
        with tempfile.TemporaryDirectory() as temporary:
            predictions, report = module._fit_fold(features, raw.price, fold, Path(temporary), config)
            self.assertEqual(set(predictions), {'single_mae', 'single_log', 'single_forest', 'historical_recipe'})
            self.assertTrue(all(len(values) == 8 and np.isfinite(values).all() for values in predictions.values()))
            self.assertEqual(report['component_counts']['historical_recipe'], 11)
            self.assertEqual(report['historical_flat_weights'], [.05] * 5 + [.1125] * 5 + [.1875])
            self.assertIn('artifact_bytes', report)
            self.assertTrue(all(value > 0 for value in report['artifact_bytes'].values()))
            for pool in ('corrected_pool', 'usage_pool'):
                destination = Path(temporary) / pool
                self.assertTrue((destination / 'robust_ann_history.csv').exists())
                self.assertFalse((destination / 'robust_ann_checkpoints').exists())
            json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
