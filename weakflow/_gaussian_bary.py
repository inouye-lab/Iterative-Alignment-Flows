import numpy as np
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels

from ddl.base import (CompositeDestructor, DestructorMixin,
                      create_inverse_transformer)
from ddl.independent import (IndependentDensity, IndependentDestructor,
                             IndependentInverseCdf)
from ddl.linear import LinearProjector

from ._base import ClassifierDestructor, ConditionalTransformer


__all__ = ['GaussianBaryClassifierDestructor']


class GaussianBaryClassifierDestructor(ClassifierDestructor):
    def __init__(self):
        pass
    
    def _fit_conditional_transform(self, X, y=None):
        X, y = check_X_y(X, y)
        classes = unique_labels(y)
        n_features = X.shape[1]
        assert len(classes) == 2, 'Only allow 2 classes'
        
        # Fit mean and covariance
        means = []
        covariances = []
        for i, yy in enumerate(classes):
            Xy = X[y==yy, :]
            means.append(np.mean(Xy, axis=0))
            covariances.append(np.cov(Xy, rowvar=False))
        assert means[0].shape[0] == n_features
        assert covariances[0].shape[0] == n_features
        assert covariances[0].shape[1] == n_features
        
        # Compute barycenter mean and covariance
        mean_bary = np.mean(means, axis=0)
        def mp(A, p):
            w, V = np.linalg.eig(A)
            return (V * np.power(w, p)) @ V.T
        def compute_bary_cov(cov_1, cov_2):
            return (
                mp(cov_1, -0.5) 
                @ mp(0.5*(
                    cov_1 + mp(mp(cov_1, 0.5) @ cov_2 @ mp(cov_1, 0.5), 0.5)
                ), 2)
                @ mp(cov_1, -0.5)
            )
        cov_bary = compute_bary_cov(*covariances)
        assert np.allclose(cov_bary, compute_bary_cov(covariances[1], covariances[0]))
        
        cov_bary_neg_half = mp(cov_bary, -0.5)
        
        # Form Ay for each class
        fitted_class_transformers = []
        for i, yy in enumerate(classes):
            if i == 0:
                i_diff = 1
            else:
                i_diff = 0
            def compute_By(Cy, Ctilde):
                return (
                    mp(Cy, -0.5)
                    @ mp(mp(Cy, 0.5) @ Ctilde @ mp(Cy, 0.5), 0.5)
                    @ mp(Cy, -0.5)
                )
            By = compute_By(covariances[i], covariances[i_diff])
            Ay = cov_bary_neg_half @ (0.5 * (np.eye(n_features) + By))
            mean_y = means[i]
            standard_normal_cdf = create_inverse_transformer(IndependentInverseCdf.create_fitted(
                n_features=n_features
            ))
            d = CompositeDestructor.create_fitted([
                LinearProjector.create_fitted(A=Ay, b=-np.dot(Ay, mean_y)),
                standard_normal_cdf,
            ])
            d.n_features_ = n_features
            fitted_class_transformers.append(d)
            
        # Save for post_transform
        self.conditional_transformer_ = ConditionalTransformer.create_fitted(
            fitted_class_transformers, classes)
        self.cov_bary_ = cov_bary
        self.cov_bary_neg_half_ = cov_bary_neg_half
        self.mean_bary_ = mean_bary
        return self.conditional_transformer_
    
    def _fit_post_transform(self, X, y=None):
        X = check_array(X)
        n_features = X.shape[1]
        X = None # Only need X for n_features
        
        A = self.cov_bary_neg_half_
        b = np.dot(A, -self.mean_bary_)
        standard_normal_cdf = create_inverse_transformer(IndependentInverseCdf.create_fitted(
            n_features=n_features
        ))
        d_bary = CompositeDestructor.create_fitted([
            LinearProjector.create_fitted(A=A, b=b),
            standard_normal_cdf
        ])
        d_bary.n_features_ = n_features
        d_bary_inv = create_inverse_transformer(d_bary)
        return d_bary_inv
