"""Semantic boundaries for the one new literal steering predictor."""
import importlib
import importlib.util
import unittest

import pandas as pd


class StructuredSteeringTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('structured_features'),
                             'The literal steering extraction module is required')
        return importlib.import_module('structured_features')

    def test_only_complete_structured_clauses_before_kpp_count(self):
        # Substring matching, missing boundaries, negation, or narrative scanning
        # would turn one of the literal negative cases into a false assertion.
        module = self.module()
        cases = [
            ('Б/у седан, правый руль, КПП автомат', 'right_explicit'),
            ('Б/у седан,  ЛЕВЫЙ   РУЛЬ , кпп автомат', 'left_explicit'),
            ('правый руль, левый руль, КПП механика', 'conflict'),
            ('правый руль, правый руль, КПП механика', 'right_explicit'),
            ('Б/у седан, КПП автомат, правый руль', 'not_stated'),
            ('Б/у седан, не правый руль, КПП автомат', 'not_stated'),
            ('Б/у седан, переделан под левый руль, КПП автомат', 'not_stated'),
            ('Б/у седан, правый руль без ДТП, КПП автомат', 'not_stated'),
            ('Б/у седан, правый руль', 'not_stated'),
            ('Б/у седан, правый руль, КППавтомат', 'not_stated'),
            (None, 'not_stated'), (float('nan'), 'not_stated'), (42, 'not_stated'),
        ]
        for description, expected in cases:
            with self.subTest(description=description):
                self.assertEqual(module.extract_structured_steering(description), expected)

    def test_price_identity_and_brand_do_not_influence_steering_or_mutate_input(self):
        module = self.module()
        original = pd.DataFrame({
            'desc': ['Б/у седан, правый руль, КПП автомат', 'Б/у седан, КПП автомат'],
            'price': [1.0, 999999999.0], 'ad_id': ['left', 'right'],
            'brand': ['unrelated', 'Japan'], 'steering': ['old', 'old'],
        }, index=[11, 91])
        unchanged = original.copy(deep=True)
        result = module.add_structured_features(original)
        pd.testing.assert_frame_equal(original, unchanged)
        self.assertEqual(result.steering.tolist(), ['right_explicit', 'not_stated'])
        perturbed = original.assign(price=[999999999.0, 1.0], ad_id=['x', 'y'], brand=['B', 'A'])
        self.assertEqual(module.add_structured_features(perturbed).steering.tolist(), result.steering.tolist())
        self.assertEqual(result.index.tolist(), [11, 91])
        with self.assertRaises(ValueError):
            module.add_structured_features(original.drop(columns='desc'))


if __name__ == '__main__':
    unittest.main()
