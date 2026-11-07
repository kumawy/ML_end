"""Behavioral contracts for a lossless, target-independent enriched copy."""
import importlib
import importlib.util
import math
import unittest

import numpy as np
import pandas as pd


class DatasetFeatureTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('dataset_features'),
                             'The original-dataset enrichment module is required')
        return importlib.import_module('dataset_features')

    def frame(self, **columns):
        count = len(next(iter(columns.values()))) if columns else 1
        values = dict(desc=['КПП автомат'] * count, brand=['BMW'] * count,
                      model=['525'] * count, year=[2015] * count,
                      mileage_km=[100000.0] * count)
        values.update(columns)
        return pd.DataFrame(values)

    def test_enrichment_preserves_every_original_cell_and_duplicate_row_index(self):
        # In-place edits, sorting, filtering, or target normalization would fail.
        original = self.frame(desc=['КПП автомат, белый, люк', None],
                              year=[1956, 2025], mileage_km=[555.0, np.nan])
        original['price'] = [1.0, 999999999.0]
        original['price_raw'] = ['1 ₸', '999 999 999 ₸']
        original['ad_id'] = ['source-a', 'source-b']
        original['link'] = ['https://a.invalid', 'https://b.invalid']
        original['extra_source_field'] = pd.Categorical(['raw', 'retained'])
        original.index = pd.Index([91, 91], name='original_row')
        untouched = original.copy(deep=True)
        enriched = self.module().add_dataset_features(original)
        self.assertIsNot(enriched, original)
        pd.testing.assert_frame_equal(original, untouched)
        pd.testing.assert_frame_equal(enriched[original.columns], untouched)
        enriched.iloc[0, enriched.columns.get_loc('price')] = -1
        self.assertEqual(original.price.iloc[0], 1.0)

    def test_price_identity_and_link_have_no_effect_and_are_not_required(self):
        module = self.module()
        minimal = self.frame(desc=['КПП автомат, белый, литые диски, кожа'])
        extra = minimal.assign(price=-100, price_raw='КПП механика, черный, люк',
                               ad_id='Mercedes-Benz E 200', link='КПП автомат, люк')
        columns = module.EXTRA_NUMERIC + module.EXTRA_CATEGORICAL + module.QUALITY_COLUMNS
        pd.testing.assert_frame_equal(module.add_dataset_features(minimal)[columns],
                                      module.add_dataset_features(extra)[columns])

    def test_age_and_mileage_proxies_use_snapshot_2025_and_keep_missingness(self):
        enriched = self.module().add_dataset_features(
            self.frame(year=[2025, 2024, 2015, 1948, 2005],
                       mileage_km=[1000, 1000, 1000, 0, np.nan]))
        self.assertEqual(enriched.vehicle_age_proxy.tolist(), [0, 1, 10, 77, 20])
        np.testing.assert_allclose(enriched.log_vehicle_age,
                                   [0, math.log(2), math.log(11), math.log(78), math.log(21)])
        np.testing.assert_allclose(enriched.log_annual_mileage_proxy.iloc[:4],
                                   [math.log(1001), math.log(1001), math.log(101), 0])
        self.assertTrue(pd.isna(enriched.log_annual_mileage_proxy.iloc[4]))
        self.assertTrue(pd.isna(enriched.mileage_km.iloc[4]))

    def test_explicit_snapshot_override_is_used_in_both_proxies(self):
        enriched = self.module().add_dataset_features(
            self.frame(year=[2025], mileage_km=[1000]), snapshot_year=2035)
        self.assertEqual(enriched.vehicle_age_proxy.iloc[0], 10)
        self.assertAlmostEqual(enriched.log_annual_mileage_proxy.iloc[0], math.log(101))

    def test_invalid_years_and_negative_mileage_only_make_derived_values_missing(self):
        original = self.frame(year=[None, 'unreadable', 2026, 0, 2015.5, 2015],
                              mileage_km=[1000, 1000, 1000, 1000, 1000, -10])
        enriched = self.module().add_dataset_features(original)
        self.assertTrue(enriched.vehicle_age_proxy.iloc[:5].isna().all())
        self.assertTrue(enriched.log_vehicle_age.iloc[:5].isna().all())
        self.assertTrue(enriched.log_annual_mileage_proxy.isna().all())
        pd.testing.assert_frame_equal(enriched[original.columns], original)
        for invalid_snapshot in [None, 0, -1, 2025.5, float('nan'), True, '2025']:
            with self.subTest(snapshot=invalid_snapshot), self.assertRaises(ValueError):
                self.module().add_dataset_features(original, snapshot_year=invalid_snapshot)

    def test_color_uses_the_exterior_slot_after_either_explicit_mileage_layout(self):
        cases = [
            ('КПП автомат, с пробегом 100 000 км, черный, литые диски', 'черный'),
            ('КПП механика, 100\u00a0000 км, белый, велюр', 'белый'),
            (' кпп   ВАРИАТОР , 100\u202f000 км,  ХАМЕЛЕОН , металлик', 'хамелеон'),
            ('КПП робот, вишня', 'вишня'),
            ('КПП автомат, бронза', 'бронза'),
            ('КПП автомат, жёлтый', 'жёлтый'),
            ('КПП автомат, зеленый', 'зеленый'),
            ('КПП автомат, голубой', 'голубой'),
        ]
        enriched = self.module().add_dataset_features(self.frame(desc=[row[0] for row in cases]))
        self.assertEqual(enriched.color_literal.tolist(), [row[1] for row in cases])

    def test_interior_narrative_negated_unknown_and_truncated_color_are_unknown(self):
        descriptions = [
            'КПП автомат, кожа, кондиционер, Европеец кожа салон черный',
            'КПП автомат, Продам машину, черный, люк',
            'КПП автомат, не черный, литые диски',
            'КПП автомат, Голубой металлик, кожа',
            'КПП автомат, кожа, черный',
            'КПП автомат, черн...',
            'КПП автомат, ультрафиолетовый, люк',
            'КПП автомат, желтый',  # No silent ё/е color-label normalization.
            'КПП не указана, белый, кожа',
            'КПП автомат, белый, КПП механика, кожа',
            'черный, литые диски, люк', None, float('nan'), 42,
        ]
        enriched = self.module().add_dataset_features(self.frame(desc=descriptions))
        self.assertEqual(enriched.color_literal.tolist(), ['not_stated'] * len(descriptions))

    def test_multiple_color_literals_remain_explicit_and_do_not_guess_a_hue(self):
        enriched = self.module().add_dataset_features(self.frame(desc=[
            'КПП автомат, черный, металлик, белый, кожа',
            'КПП автомат, черный, металлик, черный, кожа',
            'КПП автомат, белый, Продам, черный',
        ]))
        self.assertEqual(enriched.color_literal.tolist(), ['multiple_literals', 'черный', 'белый'])
        self.assertEqual(enriched.upholstery_literal.tolist(), ['кожа', 'кожа', 'not_stated'])

    def test_equipment_claims_require_complete_positive_contiguous_source_clauses(self):
        cases = [
            ('КПП автомат, белый, металлик, литые диски, люк, кожа', 1, 1, 'кожа'),
            ('КПП автомат, люк, велюр', None, 1, 'велюр'),
            ('КПП автомат, белый, панорамная крыша, комбинированный', None, None, 'комбинированный'),
            ('КПП автомат, белый, abs, bluetooth, алькантара', None, None, 'алькантара'),
            ('КПП автомат, белый, без люка, кожа', None, None, 'not_stated'),
            ('КПП автомат, белый, люка нет, литые диски', None, None, 'not_stated'),
            ('КПП автомат, белый, без литых дисков, люк', None, None, 'not_stated'),
            ('КПП автомат, белый, литые дис...', None, None, 'not_stated'),
            ('КПП автомат, белый, лю...', None, None, 'not_stated'),
            ('КПП автомат, белый, ко...', None, None, 'not_stated'),
            ('КПП автомат, белый, Продам с люком, литые диски, кожа', None, None, 'not_stated'),
            ('КПП автомат, белый, без кожи, кожа, люк', None, None, 'not_stated'),
            ('КПП автомат, белый, литые диски', 1, None, 'not_stated'),
        ]
        enriched = self.module().add_dataset_features(self.frame(desc=[row[0] for row in cases]))
        for row, (_, wheels, roof, upholstery) in zip(enriched.itertuples(), cases):
            with self.subTest(description=row.desc):
                self.assertEqual(row.upholstery_literal, upholstery)
                for actual, expected in [(row.alloy_wheels_claim, wheels), (row.sunroof_claim, roof)]:
                    if expected is None:
                        self.assertTrue(pd.isna(actual))
                    else:
                        self.assertEqual(actual, expected)

    def test_multiple_upholstery_clauses_are_not_promoted_to_combined_material(self):
        enriched = self.module().add_dataset_features(self.frame(desc=[
            'КПП автомат, белый, велюр, кожа',
            'КПП автомат, белый, кожа, кожа',
            'КПП автомат, белый, комбинированный, алькантара',
        ]))
        self.assertEqual(enriched.upholstery_literal.tolist(), ['multiple_literals', 'кожа', 'multiple_literals'])

    def test_lexical_families_are_brand_scoped_and_unmatched_models_stay_distinct(self):
        pairs = [('BMW', '525'), ('BMW', '530'), ('BMW', 'X5'), ('BMW', 'X5 M'),
                 ('BMW', 'M5'), ('Toyota', '525'), ('Mercedes-Benz', 'E 200'),
                 ('Lexus', 'E 200'), ('Lexus', 'RX 350'), ('Lexus', 'RX'),
                 ('Toyota', 'Land Cruiser'), ('Toyota', 'Land Cruiser Prado'),
                 ('BMW', '5 Series'), (None, '525'), ('BMW', None), ('', '525')]
        enriched = self.module().add_dataset_features(
            self.frame(brand=[pair[0] for pair in pairs], model=[pair[1] for pair in pairs]))
        self.assertEqual(enriched.model_family_lexical.tolist(), [
            'BMW | numeric_prefix_5', 'BMW | numeric_prefix_5',
            'BMW | original_model:X5', 'BMW | original_model:X5 M',
            'BMW | original_model:M5', 'Toyota | original_model:525',
            'Mercedes-Benz | literal_prefix_E', 'Lexus | literal_prefix_E',
            'Lexus | literal_prefix_RX', 'Lexus | original_model:RX',
            'Toyota | original_model:Land Cruiser', 'Toyota | original_model:Land Cruiser Prado',
            'BMW | original_model:5 Series', 'unknown', 'unknown', 'unknown',
        ])

    def test_quality_flags_are_row_local_review_signals_without_value_corrections(self):
        original = self.frame(year=[2015, 2016, 2000, 2000, 2000, 2000, 2000, 2000],
                              mileage_km=[1000, 999, 555, 0, 1111.0, 111.1, 11, np.nan],
                              drive=['awd', None, 'unknown', '', np.nan, 'fwd', None, None])
        enriched = self.module().add_dataset_features(original)
        self.assertEqual(enriched.review_low_mileage_old_vehicle.tolist(),
                         [True, False, True, True, False, True, True, False])
        self.assertEqual(enriched.review_repeated_digit_mileage.tolist(),
                         [False, True, True, False, True, False, False, False])
        self.assertEqual(enriched.original_drive_requires_review.tolist(),
                         [True, False, True, True, False, True, False, False])
        pd.testing.assert_frame_equal(enriched[original.columns], original)
        single = self.module().add_dataset_features(original.iloc[[2]])
        columns = enriched.columns.difference(original.columns)
        pd.testing.assert_frame_equal(single[columns], enriched.loc[[2], columns])

    def test_truncation_signal_requires_final_three_dots_at_the_documented_snippet_length(self):
        descriptions = ['x' * (length - 3) + '...' for length in [118, 119, 120, 121, 122, 123]]
        descriptions += ['Short...', 'x' * 117 + '…', 'x' * 114 + '...end',
                         'x' * 117 + '...  ', None]
        enriched = self.module().add_dataset_features(self.frame(desc=descriptions))
        self.assertEqual(enriched.description_truncation_signal.tolist(),
                         [False, True, True, True, True, False, False, False, False, True, False])

    def test_quality_and_raw_proxy_age_are_not_part_of_extra_predictor_contract(self):
        module = self.module()
        original = self.frame()
        enriched = module.add_dataset_features(original)
        predictors = module.EXTRA_NUMERIC + module.EXTRA_CATEGORICAL
        self.assertTrue(set(predictors).issubset(enriched.columns))
        self.assertFalse(set(module.QUALITY_COLUMNS) & set(predictors))
        self.assertNotIn('vehicle_age_proxy', predictors)
        self.assertFalse({'price', 'price_raw', 'ad_id', 'link', 'drive'} & set(predictors))

    def test_required_source_columns_and_generated_name_collisions_fail_before_edits(self):
        module = self.module()
        original = self.frame()
        for column in ['desc', 'brand', 'model', 'year', 'mileage_km']:
            with self.subTest(missing=column), self.assertRaisesRegex(ValueError, column):
                module.add_dataset_features(original.drop(columns=column))
        collision = original.assign(color_literal='source must survive')
        unchanged = collision.copy(deep=True)
        with self.assertRaisesRegex(ValueError, 'color_literal'):
            module.add_dataset_features(collision)
        pd.testing.assert_frame_equal(collision, unchanged)

    def test_empty_frame_retains_schema_without_reading_price_or_identity(self):
        original = self.frame().iloc[:0]
        module = self.module()
        enriched = module.add_dataset_features(original)
        self.assertEqual(len(enriched), 0)
        pd.testing.assert_frame_equal(enriched[original.columns], original)
        self.assertTrue(set(module.EXTRA_NUMERIC + module.EXTRA_CATEGORICAL +
                            module.QUALITY_COLUMNS).issubset(enriched))


if __name__ == '__main__':
    unittest.main()
