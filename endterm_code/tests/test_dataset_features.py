"""Contracts for lossless, target-independent fixed-snapshot usage proxies."""
import math
import unittest

import numpy as np
import pandas as pd

from dataset_features import EXTRA_NUMERIC, add_dataset_features


AGE_COLUMNS = ['vehicle_age_proxy', 'log_vehicle_age', 'log_annual_mileage_proxy']


class DatasetFeatureTests(unittest.TestCase):
    def test_only_three_proxies_are_added_and_source_cells_are_independent(self):
        # Extra parsing, source mutation, sorting or target access breaks this boundary.
        frame = pd.DataFrame({'year': [1956, 2025], 'mileage_km': [555., np.nan]},
                             index=pd.Index([91, 91], name='original_row'))
        untouched = frame.copy(deep=True)
        enriched = add_dataset_features(frame)
        self.assertEqual(enriched.columns.tolist(), [*frame.columns, *AGE_COLUMNS])
        pd.testing.assert_frame_equal(frame, untouched)
        pd.testing.assert_frame_equal(enriched[frame.columns], untouched)
        with_target = frame.assign(price=[-1., 999999999.], price_raw=['invalid', 'text'],
                                   ad_id=['b', 'a'], link=['url-b', 'url-a'])
        pd.testing.assert_frame_equal(enriched[AGE_COLUMNS],
                                      add_dataset_features(with_target)[AGE_COLUMNS])
        enriched.iloc[0, enriched.columns.get_loc('year')] = 0
        pd.testing.assert_frame_equal(frame, untouched)
        self.assertEqual(EXTRA_NUMERIC, ['log_vehicle_age', 'log_annual_mileage_proxy'])

    def test_snapshot_age_and_annual_mileage_use_max_age_one(self):
        frame = pd.DataFrame({'year': [2025, 2024, 2015, 1948, 2005],
                              'mileage_km': [1000., 1000., 1000., 0., np.nan]})
        enriched = add_dataset_features(frame)
        self.assertEqual(enriched.vehicle_age_proxy.tolist(), [0, 1, 10, 77, 20])
        np.testing.assert_allclose(enriched.log_vehicle_age,
                                   [0, math.log(2), math.log(11), math.log(78), math.log(21)])
        np.testing.assert_allclose(enriched.log_annual_mileage_proxy.iloc[:4],
                                   [math.log(1001), math.log(1001), math.log(101), 0])
        self.assertTrue(pd.isna(enriched.log_annual_mileage_proxy.iloc[4]))
        override = add_dataset_features(frame.iloc[[0]], snapshot_year=2035)
        self.assertEqual(override.vehicle_age_proxy.iloc[0], 10)
        self.assertAlmostEqual(override.log_vehicle_age.iloc[0], math.log(11))
        self.assertAlmostEqual(override.log_annual_mileage_proxy.iloc[0], math.log(101))

    def test_invalid_source_values_stay_unchanged_and_proxies_stay_unknown(self):
        frame = pd.DataFrame({'year': [None, 'unreadable', 2026, 0, -1, 2015.5,
                                       np.inf, 2015, 2015, 2015, 2015],
                              'mileage_km': [1000, 1000, 1000, 1000, 1000, 1000,
                                             1000, -10, np.inf, 'unknown', None]})
        enriched = add_dataset_features(frame)
        self.assertTrue(enriched.vehicle_age_proxy.iloc[:7].isna().all())
        self.assertTrue(enriched.log_vehicle_age.iloc[:7].isna().all())
        self.assertTrue(enriched.log_annual_mileage_proxy.isna().all())
        pd.testing.assert_frame_equal(enriched[frame.columns], frame)
        for snapshot in [None, 0, -1, 2025.5, np.nan, True, '2025']:
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                add_dataset_features(frame, snapshot_year=snapshot)

    def test_required_columns_and_generated_collisions_fail_before_edits(self):
        frame = pd.DataFrame({'year': [2020], 'mileage_km': [123.]})
        for column in frame:
            with self.subTest(missing=column), self.assertRaisesRegex(ValueError, column):
                add_dataset_features(frame.drop(columns=column))
        for column in AGE_COLUMNS:
            collision = frame.assign(**{column: 'original cell'})
            untouched = collision.copy(deep=True)
            with self.subTest(collision=column), self.assertRaisesRegex(ValueError, column):
                add_dataset_features(collision)
            pd.testing.assert_frame_equal(collision, untouched)

    def test_empty_input_preserves_source_schema_and_adds_float_proxies(self):
        frame = pd.DataFrame({'year': pd.Series(dtype='float64'),
                              'mileage_km': pd.Series(dtype='float64')})
        enriched = add_dataset_features(frame)
        self.assertEqual(enriched.columns.tolist(), [*frame.columns, *AGE_COLUMNS])
        pd.testing.assert_frame_equal(enriched[frame.columns], frame)
        self.assertTrue(all(str(enriched[column].dtype) == 'float64' for column in AGE_COLUMNS))


if __name__ == '__main__':
    unittest.main()
