"""Serializable components and fixed arithmetic prediction ensembles."""
from dataclasses import dataclass
from typing import Any
import numpy as np


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
