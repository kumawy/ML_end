import unittest
import importlib
import importlib.util
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression


class EnsembleTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('ensemble'),'Serializable validation-selected blend is required')
        return importlib.import_module('ensemble')

    def test_blend_search_minimizes_validation_mae_with_declared_weights(self):
        module=self.module()
        result=module.select_blend({'A':np.array([0.,200.]),'B':np.array([200.,0.])},np.array([100.,100.]),[('A','B')])
        self.assertEqual(result['weights'],[.5,.5])
        self.assertEqual(result['validation_mae_kzt'],0.)
        self.assertEqual(len(result['candidates']),3)

    def test_components_receive_their_correct_feature_representation(self):
        module=self.module()
        train=pd.DataFrame({'x':[0.,1.]})
        model=LinearRegression().fit(train,[0.,100.])
        components=[module.Component(model,'legacy'),module.Component(model,'enhanced')]
        predictor=module.PredictionEnsemble(components,[.25,.75])
        result=predictor.predict({'legacy':pd.DataFrame({'x':[1.]}),'enhanced':pd.DataFrame({'x':[3.]})})
        np.testing.assert_allclose(result,[250.])

    def test_invalid_weight_sum_is_rejected(self):
        module=self.module()
        model=DummyRegressor().fit([[0],[1]],[10,10])
        with self.assertRaises(ValueError):
            module.PredictionEnsemble([module.Component(model,'enhanced')],[.8])


if __name__=='__main__':
    unittest.main()
