import numpy as np
from sklearn.utils import check_X_y

from ddl.base import (CompositeDestructor, DestructorMixin,
                      create_inverse_transformer)
from ddl.independent import (IndependentDensity, IndependentDestructor,
                             IndependentInverseCdf)


__all__ = ['ClassifierDestructor', 'ConditionalTransformer']


class ClassifierDestructor(CompositeDestructor):
    def __init__(self):
        pass
    
    def _fit_pre_transform(self, X, y=None):
        self.pre_transformer_ = None
        return None
    
    def _fit_conditional_transform(self, X, y=None):
        self.conditional_transformer_ = None
        return None
    
    def _fit_post_transform(self, X, y=None):
        self.post_transformer_ = None
        return None
    
    def fit_transform(self, X, y=None):
        X, y = check_X_y(X, y)
        fitted_transforms = [
            self._fit_pre_transform,
            self._fit_conditional_transform,
            self._fit_post_transform,
        ]
        self.fitted_destructors_ = []
        for t_fitter in fitted_transforms:
            t = t_fitter(X, y)
            if t is not None:
                X = t.transform(X, y)
                self.fitted_destructors_.append(t)
        self.n_features_ = X.shape[1]
        return X
    
    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self


class ConditionalTransformer():
    def fit(self, X, y=None):
        raise RuntimeError('This class cannot be fitted.')
        
    @classmethod
    def create_fitted(cls, fitted_class_transformers, classes, **kwargs):
        t = cls(**kwargs)
        t.fitted_class_transformers_ = fitted_class_transformers
        t.classes_ = classes
        return t
        
    def transform(self, X, y=None):
        X, y = check_X_y(X, y)
        Z = X.copy()
        for t, yy in zip(self.fitted_class_transformers_, self.classes_):
            sel = (y==yy)
            if np.sum(sel) == 0: continue
            Z[sel, :] = t.transform(Z[sel, :])
        return Z
    
    def inverse_transform(self, X, y=None):
        X, y = check_X_y(X, y)
        Z = X.copy()
        for t, yy in zip(self.fitted_class_transformers_, self.classes_):
            sel = (y==yy)
            if np.sum(sel) == 0: continue
            Z[sel, :] = t.inverse_transform(Z[sel, :])
        return Z
    
    def score_samples(self, X, y=None):
        X, y = check_X_y(X, y)
        Z = X.copy()
        scores = np.zeros(Z.shape[0])
        for i, yy in enumerate(self.classes_):
            sel = (y==yy)
            if np.sum(sel) == 0: continue
            scores[sel] += self.fitted_class_transformers_[i].score_samples(Z[sel, :])
            Z[sel, :] = self.fitted_class_transformers_[i].transform(Z[sel, :])
            # TODO: Inefficient (should implement transform_score)
        return scores
