"""Small predeclared validation-only mixtures and their saved inference contract."""
from dataclasses import dataclass
from typing import Any
import numpy as np


def select_blend(validation_predictions, y_val, pairs, weights=(.25,.5,.75)):
    target=np.asarray(y_val,dtype=float).ravel()
    if not len(target) or not np.isfinite(target).all():
        raise ValueError('Validation targets must be finite and nonempty')
    candidates=[]
    for first,second in pairs:
        a=np.asarray(validation_predictions[first],dtype=float).ravel()
        b=np.asarray(validation_predictions[second],dtype=float).ravel()
        if a.shape != target.shape or b.shape != target.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Each finite prediction must align with validation targets')
        for weight in weights:
            if not 0 < weight < 1:
                raise ValueError('Blend weights must lie strictly between zero and one')
            prediction=weight*a+(1-weight)*b
            candidates.append({'components':[first,second],'weights':[float(weight),float(1-weight)],
                               'validation_mae_kzt':float(np.mean(np.abs(target-prediction)))})
    if not candidates:
        raise ValueError('At least one candidate blend is required')
    best=min(candidates,key=lambda row:row['validation_mae_kzt'])
    return {'selected_components':best['components'],'weights':best['weights'],
            'validation_mae_kzt':best['validation_mae_kzt'],'candidates':candidates,
            'selection':'Lowest validation MAE among predeclared pairs and weights; no fitting on test',
            'test_data_used':False}


@dataclass
class Component:
    estimator: Any
    representation: str
    preprocessor: Any = None

    def predict(self, inputs):
        frame=inputs[self.representation]
        if self.preprocessor is not None:
            frame=self.preprocessor.transform(frame)
        return np.asarray(self.estimator.predict(frame),dtype=float).ravel()


@dataclass
class PredictionEnsemble:
    components: list
    weights: list

    def __post_init__(self):
        values=np.asarray(self.weights,dtype=float)
        if len(values)!=len(self.components) or not len(values) or not np.isfinite(values).all() or (values<0).any() or not np.isclose(values.sum(),1.):
            raise ValueError('Nonnegative component weights must sum to one')

    def predict(self, inputs):
        predictions=[component.predict(inputs) for component in self.components]
        if any(values.shape!=predictions[0].shape or not np.isfinite(values).all() for values in predictions):
            raise ValueError('Component predictions must be finite and aligned')
        return np.maximum(np.sum(np.asarray(predictions)*np.asarray(self.weights)[:,None],axis=0),0.)
