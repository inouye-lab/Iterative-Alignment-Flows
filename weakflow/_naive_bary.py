import copy  # For deepcopy function

import numpy as np
from sklearn.base import clone
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels

from ddl.base import (CompositeDestructor, DestructorMixin,
                      create_inverse_transformer)
from ddl.independent import (IndependentDensity, IndependentDestructor,
                             IndependentInverseCdf)
from ddl.linear import LinearProjector
from ddl.univariate import HistogramUnivariateDensity

from ._base import ClassifierDestructor, ConditionalTransformer


__all__ = ['NaiveBaryClassifierDestructor']


class NaiveBaryClassifierDestructor(ClassifierDestructor):
    def __init__(self, hist_bins='auto', n_grid=100, bound_eps='auto'):
        self.hist_bins = hist_bins
        self.n_grid = n_grid
        self.bound_eps = bound_eps
    
    def _fit_conditional_transform(self, X, y=None):
        X, y = check_X_y(X, y)
        classes = unique_labels(y)
        
        # Setup independent density destructor
        bins = self.hist_bins
        if bins == 'auto':
            bins = int(np.round(np.sqrt(X.shape[0])))
        ind = CompositeDestructor([
            IndependentDestructor(), # Gaussian
            IndependentDestructor(IndependentDensity(
                HistogramUnivariateDensity(bins=bins, bounds=[0,1])
            ))
        ])
        
        # Learn density destructor for each
        fitted_class_transformers = []
        for i, yy in enumerate(classes):
            d = clone(ind)
            d.fit(X[y==yy, :])
            fitted_class_transformers.append(d)
            
        # Save for post_transform
        self.conditional_transformer_ = ConditionalTransformer.create_fitted(fitted_class_transformers, classes)
        return self.conditional_transformer_
    
    def _fit_post_transform(self, X, y=None):
        assert len(self.conditional_transformer_.classes_) == 2, 'Only 2 classes is implemented'
        X = check_array(X)
        n_features = X.shape[1]
        X = None # Only need X for n_features
        d_bary_inv = _get_ind_bary_inv_transformer(
            self.conditional_transformer_, self.n_grid, 
            self.bound_eps, n_features, is_unit_bounded=False)
        self.post_transformer_ = d_bary_inv
        return self.post_transformer_
    

def _get_ind_bary_inv_transformer(conditional_transformer, n_grid, bound_eps, 
                                 n_features, is_unit_bounded=False):
    # -- Construct barycenter transform via McCann's interpolation
    # Start by finding quantiles all features
    # Quantiles bounded away from 0 and 1
    # Go in both directions via inverse transforms
    if bound_eps == 'auto':
        bound_eps = 1/n_grid
    u_bary = np.linspace(-0.5, 0.5, n_grid) * (1-bound_eps) + 0.5
    U_bary = np.outer(u_bary, np.ones(n_features))
    X_query_per_class = np.array([
        t.inverse_transform(U_bary)
        for t in conditional_transformer.fitted_class_transformers_
    ])
    X_bary = np.mean(X_query_per_class, axis=0) # 

    if not is_unit_bounded:
        # Use Gaussian pre-destructor to handle tails
        normal_destructor = IndependentDestructor().fit(X_bary)
        X_bary = normal_destructor.transform(X_bary)

    # Form pseudo-histogram for d_bary that will serve as interpolation
    fitted = []
    for xb, ub in zip(X_bary.T, U_bary.T):  # each feature independently
        # hist should be pseudo-counts/mass (i.e., not density values)
        #  HistogramUnivariateDensity.create_fitted accounts for bin_edges
        hist = np.concatenate(([bound_eps/2], np.diff(ub), [bound_eps/2]))  # Mass in each bin is like spacing between u_bary
        d_hist = HistogramUnivariateDensity.create_fitted(
            hist, bin_edges=np.concatenate(([0], xb, [1]))) 
        fitted.append(d_hist)

    # Create independent destructor and invert
    ind_fitted = IndependentDensity.create_fitted(fitted, n_features=n_features)
    d_hist_bary = IndependentDestructor.create_fitted(ind_fitted)
    if not is_unit_bounded:
        d_bary = CompositeDestructor.create_fitted([
            normal_destructor,
            d_hist_bary,
        ])
    else:
        d_bary = d_hist_bary
    d_bary.n_features_ = n_features
    d_bary_inv = create_inverse_transformer(d_bary)
    return d_bary_inv


