"""Source-cell and split contracts for the focused 38-column audit export."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import data
from dataset_export import export_dataset


ROOT = Path(__file__).resolve().parents[1]
AGE_COLUMNS = ['vehicle_age_proxy', 'log_vehicle_age', 'log_annual_mileage_proxy']


class DatasetExportTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            'ad_id': ['a1', 'a2', 'a3'], 'price': [1000., 2000., 3000.],
            'price_raw': ['1 000 ₸', '2 000 ₸', '3 000 ₸'],
            'brand': ['BMW', 'Toyota', 'Lexus'], 'model': ['525', 'Camry', 'RX 350'],
            'year': [2020., 2010., 2015.], 'mileage_km': [50000., np.nan, 888.],
            'drive': [None, 'awd', None],
            'desc': ['Б/у седан, 2 л, бензин, левый руль, КПП автомат',
                     'Б/у седан, 2 л, бензин, КПП автомат, черный',
                     'Б/у кроссовер, 3.5 л, бензин, правый руль, КПП автомат'],
        }, index=[50, 10, 30])

    def split(self):
        return {'train': [10], 'validation': [30], 'test': [50]}

    def test_original_dataset_export_has_38_columns_and_preserves_all_cells_and_splits(self):
        # Over-exporting unused families or changing any retained cell/split fails here.
        source = ROOT / 'kolesa_cars_2025.csv'
        source_bytes = source.read_bytes()
        frame, _ = data.load_data(ROOT)
        untouched = frame.copy(deep=True)
        splits = data.split_data(frame)
        self.assertEqual((len(frame), len(frame.columns)), (2553, 32))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = export_dataset(frame, splits, output, source)
            saved = pd.read_csv(output / 'used_cars_enriched.csv', dtype={'ad_id': str})
            self.assertEqual(saved.shape, (2553, 38))
            self.assertEqual(saved.columns.tolist(), [*frame.columns, 'steering',
                                                       *AGE_COLUMNS, 'source_row_index', 'split'])
            self.assertEqual(saved.source_row_index.tolist(), frame.index.tolist())
            restored = saved.set_index('source_row_index')
            restored.index.name = frame.index.name
            pd.testing.assert_frame_equal(restored[frame.columns], frame,
                                          check_dtype=False, check_exact=True)
            for label, indices in splits.items():
                self.assertEqual(set(saved.loc[saved['split'].eq(label), 'source_row_index']),
                                 set(indices))
            self.assertEqual(report['split_counts'], {'train': 1787, 'validation': 383, 'test': 383})
            self.assertEqual(report['fit_input_columns'], [*data.FEATURES, 'steering',
                                                           'log_vehicle_age', 'log_annual_mileage_proxy'])
            self.assertEqual(report['source_raw_sha256'], hashlib.sha256(source_bytes).hexdigest())
            self.assertTrue(report['preserves_ids_targets'])
            self.assertTrue(report['source_row_order_preserved'])
            self.assertEqual({path.name for path in output.iterdir()},
                             {'used_cars_enriched.csv', 'data_dictionary.csv', 'quality_report.json', 'README.md'})
        pd.testing.assert_frame_equal(frame, untouched)
        self.assertEqual(source.read_bytes(), source_bytes)

    def test_dictionary_and_report_separate_fit_inputs_from_target_audit_and_metadata(self):
        frame = self.frame()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = export_dataset(frame, self.split(), output)
            inputs = set(report['fit_input_columns'])
            self.assertTrue({'year', 'steering', 'log_vehicle_age', 'log_annual_mileage_proxy'} <= inputs)
            self.assertFalse({'price', 'price_raw', 'ad_id', 'desc', 'drive', 'source_row_index',
                              'split', 'vehicle_age_proxy'} & inputs)
            self.assertEqual(report['missing_counts']['mileage_km'], 1)
            self.assertEqual(report['snapshot_year_for_proxies'], 2025)
            self.assertEqual(json.loads((output / 'quality_report.json').read_text()), report)
            dictionary = pd.read_csv(output / 'data_dictionary.csv').set_index('column')
            self.assertEqual(dictionary.index.tolist(), [*frame.columns, 'steering', *AGE_COLUMNS,
                                                         'source_row_index', 'split'])
            self.assertEqual(dictionary.loc['price', 'role'], 'target')
            self.assertEqual(dictionary.loc['split', 'role'], 'split_metadata')
            self.assertEqual(dictionary.loc['vehicle_age_proxy', 'role'], 'derived_audit')
            self.assertFalse(dictionary.loc['vehicle_age_proxy', 'fit_input'])
            saved = pd.read_csv(output / 'used_cars_enriched.csv')
            self.assertEqual(saved.source_row_index.tolist(), [50, 10, 30])
            self.assertEqual(saved['split'].tolist(), ['test', 'train', 'validation'])
            self.assertEqual(saved.steering.tolist(), ['left_explicit', 'not_stated', 'right_explicit'])
        pd.testing.assert_frame_equal(frame, self.frame())

    def test_invalid_splits_cannot_silently_change_evaluation(self):
        invalid = [
            {'train': [10], 'validation': [30], 'test': [30, 50]},
            {'train': [10], 'validation': [30], 'test': []},
            {'train': [10], 'validation': [30], 'test': [70]},
            {'train': [10], 'validation': [30], 'test': [10]},
            {'train': [10], 'test': [30, 50]},
            {'train': [10, 10], 'validation': [30], 'test': [50]},
        ]
        for split in invalid:
            with self.subTest(split=split), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, 'split|Split'):
                    export_dataset(self.frame(), split, Path(directory))
        duplicate_index = self.frame().set_axis([10, 10, 30])
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            export_dataset(duplicate_index, self.split(), Path(directory))

    def test_invalid_id_target_or_reserved_columns_are_rejected_without_repair(self):
        variants = [self.frame().assign(ad_id=['a', 'a', 'b']),
                    self.frame().assign(ad_id=['a', None, 'b']),
                    self.frame().assign(price=[-1., 2000., 3000.]),
                    self.frame().assign(price=[np.inf, 2000., 3000.]),
                    self.frame().assign(price=[np.nan, 2000., 3000.]),
                    self.frame().assign(source_row_index=[1, 2, 3]),
                    self.frame().assign(split=['test', 'train', 'validation']),
                    self.frame().assign(steering=['source'] * 3)]
        for frame in variants:
            untouched = frame.copy(deep=True)
            with self.subTest(columns=frame.columns.tolist()), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    export_dataset(frame, self.split(), Path(directory))
            pd.testing.assert_frame_equal(frame, untouched)


if __name__ == '__main__':
    unittest.main()
