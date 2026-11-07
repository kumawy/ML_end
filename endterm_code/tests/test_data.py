import hashlib
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class DataProtocolTests(unittest.TestCase):
    def require_module(self):
        self.assertTrue((ROOT / 'data.py').exists(), 'Data preparation module is required')
        import data
        return data

    def test_original_data_and_historical_split_are_preserved(self):
        module = self.require_module()
        frame, audit = module.load_data(ROOT)
        split = module.split_data(frame)
        self.assertEqual(len(frame), 2553)
        self.assertEqual([len(split[k]) for k in ('train', 'validation', 'test')], [1787, 383, 383])
        ids = [set(frame.loc[split[k], 'ad_id']) for k in ('train', 'validation', 'test')]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(audit['source_git_blob_sha'], '01e57295927f91ffd7996745ce6a867b6d41b0f7')

    def test_transforming_unseen_categories_and_missing_mileage_does_not_refit(self):
        module = self.require_module()
        frame, _ = module.load_data(ROOT)
        split = module.split_data(frame)
        train = frame.loc[split['train'], module.FEATURES]
        val = frame.loc[split['validation'], module.FEATURES].copy()
        prep = module.make_preprocessor()
        prep.fit(train)
        before = prep.named_transformers_['num'].named_steps['impute'].statistics_.copy()
        self.assertTrue(np.allclose(before, np.nanmedian(train[module.NUMERIC].to_numpy(), axis=0)))
        val.loc[val.index[0], 'brand'] = 'Unseen test brand'
        val.loc[val.index[0], 'log_km'] = np.nan
        transformed = prep.transform(val)
        after = prep.named_transformers_['num'].named_steps['impute'].statistics_
        self.assertTrue(np.array_equal(before, after))
        self.assertTrue(np.isfinite(transformed).all())


if __name__ == '__main__':
    unittest.main()
