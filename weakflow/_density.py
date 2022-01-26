import copy  # For deepcopy function

from sklearn.base import clone
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

from ddl.base import (CompositeDestructor, DestructorMixin,
                      create_inverse_transformer)
from ddl.independent import (IndependentDensity, IndependentDestructor,
                             IndependentInverseCdf)

from ._base import ClassifierDestructor, ConditionalTransformer


__all__ = ['DensityConditionalDestructor']


# Deep density destructor algorithm
class DensityConditionalDestructor(ClassifierDestructor):
    """Wrapper for density destructors looks like a classifier destructor."""
    def __init__(self, density_destructor=None):
        self.density_destructor = density_destructor
    
    def _fit_conditional_transform(self, X, y=None):
        X, y = check_X_y(X, y)
        classes = unique_labels(y)
        
        # Fit independent density destructors
        density_destructor = self.density_destructor
        if density_destructor is None:
            density_destructor = IndependentDestructor()
        fitted_class_transformers = [
            clone(density_destructor).fit(X[y==yy, :])
            for yy in classes
        ]
        
        # Put into a conditional transformer
        self.conditional_transformer_ = ConditionalTransformer.create_fitted(
            fitted_class_transformers, classes)
        return self.conditional_transformer_
