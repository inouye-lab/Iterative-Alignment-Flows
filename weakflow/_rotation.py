"""Rotation estimators including SIE and Random."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state


__all__ = ['RandomRotationEstimator']


class RandomRotationEstimator(BaseEstimator):
    def __init__(self, random_state=None):
        self.random_state = random_state
    
    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n_features = X.shape[1]
        Q, _ = np.linalg.qr(rng.randn(n_features, n_features))
        self.rotation_ = Q
        return self
