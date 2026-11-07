"""Regressions anchored to hand-inspected listings in the unchanged CSV."""
import hashlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import data

ROOT = Path(__file__).resolve().parents[1]


class VehicleFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.audit = data.load_data(ROOT)
        cls.by_id = cls.frame.set_index('ad_id', drop=False)

    def require_features(self, *names):
        self.assertTrue(set(names).issubset(self.frame), f'Missing vehicle features: {names}')

    def test_explicit_structured_dual_fuel_is_not_petrol(self):
        self.assertEqual(self.by_id.loc['206388093', 'fuel'], 'gas_petrol')
        self.assertEqual(self.by_id.loc['206820356', 'fuel'], 'gas_petrol')

    def test_narrative_dual_fuel_claim_does_not_erase_conflicting_specification(self):
        self.require_features('dual_fuel_claim', 'fuel_text_conflict')
        row = self.by_id.loc['206019469']
        self.assertEqual(row.fuel, 'petrol')
        self.assertEqual(row.dual_fuel_claim, 1)
        self.assertEqual(row.fuel_text_conflict, 1)
        # 113 ASCII-hyphen mentions plus the inspected «Газ — бензин» listing.
        self.assertEqual(int(self.frame.dual_fuel_claim.sum()), 114)
        self.assertEqual(self.by_id.loc['206196328', 'fuel_text_conflict'], 1)
        self.assertEqual(int(self.frame.fuel.eq('gas_petrol').sum()), 109)

    def test_electric_displacement_is_structural_rather_than_median_petrol(self):
        self.assertEqual(self.by_id.loc['169223607', 'fuel'], 'electric')
        self.require_features('engine_missing', 'engine_not_applicable', 'engine_source_conflict')
        row = self.by_id.loc['169223607']
        self.assertEqual(row.engine_liters_clean, 0)
        self.assertEqual(row.engine_not_applicable, 1)
        self.assertEqual(row.engine_missing, 0)
        self.assertEqual(row.engine_source_conflict, 0)
        self.assertEqual(int(self.frame.engine_not_applicable.sum()), 15)

    def test_contradictory_passat_retains_reported_engine_and_is_flagged(self):
        self.require_features('engine_missing', 'engine_not_applicable', 'engine_source_conflict')
        row = self.by_id.loc['206855374']
        self.assertEqual(row.fuel, 'electric')
        self.assertEqual(row.engine_liters_clean, 1.8)
        self.assertEqual(row.engine_source_conflict, 1)
        self.assertEqual(row.engine_not_applicable, 0)
        self.assertEqual(row.engine_missing, 0)

    def test_invalid_combustion_displacement_remains_missing(self):
        self.require_features('engine_missing', 'engine_not_applicable')
        for ad_id in ['204501352', '198281774']:
            row = self.by_id.loc[ad_id]
            self.assertTrue(np.isnan(row.engine_liters_clean))
            self.assertEqual(row.engine_missing, 1)
            self.assertEqual(row.engine_not_applicable, 0)

    def test_explicit_trim_and_unit_qualified_power_are_extracted(self):
        self.require_features('brand_model', 'trim', 'power_kw', 'power_missing')
        self.assertEqual(self.by_id.loc['206695408', 'brand_model'], 'Porsche | Panamera')
        self.assertEqual(self.by_id.loc['206695408', 'trim'], 'GTS')
        self.assertEqual(self.by_id.loc['206189905', 'trim'], 'AMG')
        self.assertEqual(self.by_id.loc['196211789', 'trim'], 'not_stated')
        self.assertAlmostEqual(self.by_id.loc['206189905', 'power_kw'], 470.7192, places=4)
        self.assertAlmostEqual(self.by_id.loc['205993064', 'power_kw'], 598.0513, places=4)
        self.assertAlmostEqual(self.by_id.loc['205198468', 'power_kw'], 389.8143, places=4)
        self.assertTrue(np.isnan(self.by_id.loc['206695408', 'power_kw']))

    def test_explicit_equipment_and_warranty_claims_are_detected(self):
        self.require_features('restyling_claim', 'leather_claim', 'panorama_claim', 'warranty_claim')
        for ad_id, feature in [('206665070', 'restyling_claim'),
                               ('206820356', 'leather_claim'),
                               ('206857041', 'panorama_claim'),
                               ('206752052', 'warranty_claim')]:
            self.assertEqual(self.by_id.loc[ad_id, feature], 1)
        # Truncated «панорамна...» does not prove the full equipment claim.
        self.assertEqual(self.by_id.loc['169223607', 'panorama_claim'], 0)

    def test_negated_claims_and_unqualified_numbers_do_not_become_features(self):
        self.assertTrue(callable(getattr(data, 'engineer_vehicle_features', None)))
        fixture = self.by_id.loc[['206695408']].copy()
        fixture['name'] = 'Porsche Panamera'
        fixture['desc'] = ('2024 г., Б/у лифтбек, 4 л, бензин, КПП робот, '
                           'не GTS, дорестайлинг, без кожи, панорамы нет, '
                           'гарантия закончилась, мощность 802, цена 64000000')
        row = data.engineer_vehicle_features(fixture).iloc[0]
        self.assertEqual(row.trim, 'not_stated')
        for feature in ['restyling_claim', 'leather_claim', 'panorama_claim', 'warranty_claim']:
            self.assertEqual(row[feature], 0)
        self.assertTrue(np.isnan(row.power_kw))

    def test_unknown_fuel_remains_explicit_and_is_not_guessed_from_model(self):
        self.assertTrue(callable(getattr(data, 'engineer_vehicle_features', None)))
        fixture = self.by_id.loc[['163019633']].copy()
        fixture['fuel'] = np.nan
        fixture['desc'] = '2019 г., Б/у седан, КПП автомат'
        row = data.engineer_vehicle_features(fixture).iloc[0]
        self.assertEqual(row.fuel, 'unknown')
        self.assertEqual(row.engine_not_applicable, 0)
        self.assertEqual(row.engine_missing, 1)

    def test_price_fields_and_identity_fields_cannot_change_predictors(self):
        self.assertTrue(callable(getattr(data, 'engineer_vehicle_features', None)))
        original = self.frame.copy()
        modified = original.copy()
        modified['price'] = modified.price * 7 + 802
        modified['price_raw'] = '802 hp 64 000 000 ₸'
        modified['ad_id'] = '000000000'
        modified['link'] = 'https://example.invalid/show/000000000'
        before = data.engineer_vehicle_features(original)[data.FEATURES]
        after = data.engineer_vehicle_features(modified)[data.FEATURES]
        pd.testing.assert_frame_equal(before, after)
        self.assertFalse({'price', 'price_raw', 'ad_id', 'link'} & set(data.FEATURES))

    def test_exact_historical_split_order_is_preserved(self):
        split = data.split_data(self.frame)
        expected = {
            'train': 'c396b47e0cc194404910dad769f9906775a2a824ae03ac66584b4048b48da832',
            'validation': 'dae1b3f312c98b86c35632cf05a0dabad93a014f1a367eb561fa8ce5cb83ffa7',
            'test': '3a4137109eef76420e5648c73d19727dfcfe7b4b08f53ce2db80b298b32312fb',
        }
        for name, indices in split.items():
            digest = hashlib.sha256('\n'.join(self.frame.loc[indices, 'ad_id']).encode()).hexdigest()
            self.assertEqual(digest, expected[name])
        self.assertEqual(self.audit['source_sha256'],
                         'b348551cd876bde7d25fce79c3fcdbd05920403c567a6aedc0e0a5e19e260468')

    def test_frequency_search_and_explicit_unknown_categories_are_supported(self):
        from inspect import signature
        self.assertIn('min_frequency', signature(data.make_preprocessor).parameters)
        train = self.frame.loc[data.split_data(self.frame)['train'], data.FEATURES]
        query = train.iloc[:1].copy()
        query['fuel'] = np.nan
        query['model'] = 'unseen-model'
        for threshold in [1, 2, 5]:
            prep = data.make_preprocessor(min_frequency=threshold).fit(train)
            transformed = prep.transform(query)
            self.assertTrue(np.isfinite(transformed).all())
            imputed = prep.named_transformers_['cat'].named_steps['impute'].transform(query[data.CATEGORICAL])
            self.assertEqual(imputed[0, data.CATEGORICAL.index('fuel')], 'unknown')

    def test_corrected_ablation_ignores_extra_claims_but_keeps_engine_state(self):
        from inspect import signature
        self.assertIn('feature_set', signature(data.make_preprocessor).parameters)
        train = self.frame.loc[data.split_data(self.frame)['train'], data.FEATURES]
        prep = data.make_preprocessor(min_frequency=2, feature_set='corrected').fit(train)
        query = self.by_id.loc[['169223607', '206855374'], data.FEATURES].copy()
        before = prep.transform(query)
        changed = query.copy()
        for feature in ['dual_fuel_claim', 'fuel_text_conflict', 'restyling_claim',
                        'leather_claim', 'panorama_claim', 'warranty_claim',
                        'power_kw', 'power_missing']:
            changed[feature] = 999
        changed['brand_model'] = 'excluded brand/model claim'
        changed['trim'] = 'excluded trim claim'
        np.testing.assert_array_equal(prep.transform(changed), before)
        self.assertTrue(np.isfinite(before).all())
        for feature in ['engine_missing', 'engine_not_applicable', 'engine_source_conflict']:
            changed_engine = query.copy()
            changed_engine[feature] += 1
            self.assertFalse(np.array_equal(prep.transform(changed_engine), before), feature)

    def test_unknown_feature_set_is_rejected_before_fitting(self):
        from inspect import signature
        self.assertIn('feature_set', signature(data.make_preprocessor).parameters)
        with self.assertRaisesRegex(ValueError, 'feature_set'):
            data.make_preprocessor(feature_set='misspelled-ablation')


if __name__ == '__main__':
    unittest.main()
