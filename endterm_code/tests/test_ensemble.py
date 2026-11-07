import unittest
import importlib
import importlib.util
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression


class EnsembleTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('ensemble'),'Serializable validation-selected blend is required')
        return importlib.import_module('ensemble')

    def test_frozen_recipe_and_saved_inference_keep_all_ten_ann_members_and_forest(self):
        # Dropping a member, swapping the two pools, changing weights, or losing
        # fitted components on serialization changes the hand-computed 7500 KZT.
        import run_project
        module = self.module()
        self.assertTrue(hasattr(run_project, 'build_selected_ensemble'),
                        'Build the final frozen recipe without validation blend search')
        x = pd.DataFrame({'x': [0., 1.]})
        def component(value):
            estimator = DummyRegressor(strategy='constant', constant=value).fit(x, [0., 0.])
            return module.Component(estimator, 'enhanced')
        usage = module.PredictionEnsemble([component(value) for value in [1000., 2000., 3000., 4000., 5000.]], [.2] * 5)
        corrected = module.PredictionEnsemble([component(value) for value in [6000., 7000., 8000., 9000., 10000.]], [.2] * 5)
        forest = component(12000.)
        selected = run_project.build_selected_ensemble(corrected, usage, forest)
        np.testing.assert_allclose(selected.weights, [.05] * 5 + [.1125] * 5 + [.1875], rtol=0, atol=1e-16)
        np.testing.assert_allclose(selected.predict({'enhanced': x}), [7500., 7500.], rtol=0, atol=1e-10)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'selected_model.joblib'
            joblib.dump(selected, path)
            np.testing.assert_array_equal(joblib.load(path).predict({'enhanced': x}), selected.predict({'enhanced': x}))

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
