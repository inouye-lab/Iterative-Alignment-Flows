# MI via gradient boosted trees?
import copy  # For deepcopy function

import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels

from ddl.base import (CompositeDestructor, DestructorMixin,
                      create_inverse_transformer)
from ddl.independent import (IndependentDensity, IndependentDestructor,
                             IndependentInverseCdf)


__all__ = ['MutualInformationEstimator']


class MutualInformationEstimator():
    def __init__(self, classifier=None):
        self.classifier = classifier
        
    def fit(self, X, y=None, X_test=None, y_test=None):
        if self.classifier is None:
            classifier = GaussianProcessClassifier(1.0 * RBF(1.0))
        else:
            classifier = clone(self.classifier)
        normalizer = StandardScaler()
        X = normalizer.fit_transform(X)
        X_test = normalizer.transform(X_test)
        
        classifier.fit(X, y)
        self.fitted_classifier_ = classifier
        self.normalizer_ = normalizer
        if X_test is not None:
            assert y_test is not None, 'y_test also needs to be supplied'
            self.mutual_information_ = self._estimate_mutual_information(X_test, y_test)
            y_pred_test = classifier.predict(X_test)
            self.accuracy_ = accuracy_score(y_test, y_pred_test)
        return self
    
    def _estimate_mutual_information(self, X, y):
        # MI = JSD() \approx max_f E_{X_1}[log(f(x))] + E_{X_2}[log(1-f(x))]
        #TODO check is fitted
        X = self.normalizer_.transform(X)
        y_prob = self.fitted_classifier_.predict_proba(X)
        # NOTE: Must take mean since separate expectations (equivalent if n_0 = n_1)
        return -np.mean([
            log_loss(y[y==yy], y_prob[y==yy], labels=[0, 1])
            for yy in unique_labels(y)
        ])
