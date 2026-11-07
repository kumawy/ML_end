import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd


class DatasetExportTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('dataset_export'),
                             'The enriched dataset needs an auditable export boundary')
        return importlib.import_module('dataset_export')

    def frame(self):
        return pd.DataFrame({
            'ad_id':['a1','a2','a3'], 'price':[1000.,2000.,3000.],
            'price_raw':['1 000 ₸','2 000 ₸','3 000 ₸'],
            'brand':['BMW','Toyota','Lexus'], 'model':['525','Camry','RX 350'],
            'year':[2020.,2010.,2015.], 'mileage_km':[50000.,float('nan'),888.],
            'drive':[None,'awd',None],
            'desc':['Б/у седан, 2 л, бензин, КПП автомат, 50 000 км, белый, люк',
                    'Б/у седан, 2 л, бензин, КПП автомат, черный, салон черный',
                    'Б/у кроссовер, 3.5 л, бензин, КПП автомат, 888 км, серый'],
        }, index=[10,30,50])

    def test_round_trip_preserves_every_identity_target_and_split(self):
        module=self.module();frame=self.frame()
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            report=module.export_dataset(frame,{'train':[10],'validation':[30],'test':[50]},output)
            saved=pd.read_csv(output/'used_cars_enriched.csv',dtype={'ad_id':str})
            self.assertEqual(saved.ad_id.tolist(),frame.ad_id.tolist())
            self.assertEqual(saved.price.tolist(),frame.price.tolist())
            self.assertEqual(saved.source_row_index.tolist(),frame.index.tolist())
            self.assertEqual(saved['split'].tolist(),['train','validation','test'])
            self.assertTrue(pd.isna(saved.loc[1,'mileage_km']))
            self.assertTrue(report['preserves_ids_targets'])
            pd.testing.assert_frame_equal(frame,self.frame())

    def test_export_separates_predictors_from_targets_quality_and_metadata(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as directory:
            report=module.export_dataset(self.frame(),{'train':[10],'validation':[30],'test':[50]},Path(directory))
            candidates=set(report['candidate_feature_columns'])
            self.assertTrue({'color_literal','model_family_lexical','log_vehicle_age'}<=candidates)
            self.assertFalse({'price','price_raw','ad_id','desc','source_row_index','split',
                              'review_repeated_digit_mileage','original_drive_requires_review'} & candidates)
            dictionary=pd.read_csv(Path(directory)/'data_dictionary.csv').set_index('column')
            self.assertEqual(dictionary.loc['price','role'],'target')
            self.assertEqual(dictionary.loc['split','role'],'split_metadata')
            self.assertEqual(dictionary.loc['review_repeated_digit_mileage','role'],'quality_flag')

    def test_incomplete_or_overlapping_splits_cannot_silently_change_evaluation(self):
        module=self.module()
        for split in [{'train':[10],'validation':[30],'test':[30,50]},
                      {'train':[10],'validation':[30],'test':[]},
                      {'train':[10],'validation':[30],'test':[70]}]:
            with self.subTest(split=split),tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError,'split|Split'):
                    module.export_dataset(self.frame(),split,Path(directory))


if __name__=='__main__':unittest.main()
