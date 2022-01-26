import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.datasets import make_spd_matrix
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import unique_labels
import torch

from weakflow._base import ClassifierDestructor, ConditionalTransformer
from ddl.base import (CompositeDestructor, DestructorMixin, create_inverse_transformer, 
                      BoundaryWarning, DataConversionWarning, IdentityDestructor)
from ddl.independent import IndependentDensity, IndependentDestructor, IndependentInverseCdf
from ddl.univariate import HistogramUnivariateDensity
from ddl.linear import LinearProjector


class GaussianBaryMultiClassifierDestructor(ClassifierDestructor):

    def __init__(self, n_iters =20, weight = None, mode=1, verbosity = 0):

        super().__init__()
        self.n_iters = n_iters
        self.weight = weight
        self.mode = mode
        self.verbosity = verbosity
    
    def _fit_conditional_transform(self, X, y=None):
        X, y = check_X_y(X, y)
        classes = unique_labels(y)
        n_classes = len(classes)
        n_features = X.shape[1]
        weight = self.weight
        
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

        # set the weight to be uniform if there is no weight vector given
        if weight is None:
            weight = np.ones((1,n_classes)) / n_classes
        assert weight.shape[0] == 1
        
        # Compute barycenter mean and covariance
        # compute the mean
        mean_bary = np.squeeze(weight.dot(means))
        # function for matrix power
        def mp(A, p):
            w, V = np.linalg.eig(A)
            return (V * np.power(w, p)) @ V.T
        def compute_bary_cov(covariances):
            # algorithm based on Remark 9.5 from https://arxiv.org/pdf/1803.00567.pdf
            # initialize the covariance matrix with random positive semi-definitive matrix
            cov = make_spd_matrix(n_features)
            assert np.all(np.linalg.eigvals(cov) >= 0) 
            i_list = []
            err_list = []

            for i in range(self.n_iters):
                new_cov = np.zeros((n_features,n_features))
                if self.mode == 1:
                    for j in range(n_classes):
                        new_cov += weight[0,j] * mp(mp(cov,0.5) @ covariances[j] @ mp(cov,0.5), 0.5) 
                if self.mode == 2:
                    for j in range(n_classes):
                        new_cov += weight[0,j] * mp(mp(cov,0.5) @ covariances[j] @ mp(cov,0.5), 0.5) 
                    new_cov = mp(cov, -0.5) @ mp(new_cov, 2) @ mp(cov, -0.5)
                # test whether the algorithm converges
                if i>0:
                    i_list.append(i)
                    err_list.append(mean_squared_error(new_cov, cov))

                cov = np.copy(new_cov)
                if self.verbosity > 0:
                    if i % 10 == 0:
                        print(f'iteration {i} done!')
            if self.verbosity > 0:
                fig,ax = plt.subplots()
                ax.plot(i_list, err_list)
                ax.set_xlabel('iterations')
                ax.set_ylabel('MSE between covariances')
                ax.set_title('check convergence')
            return cov

        cov_bary = compute_bary_cov(covariances)
        #assert np.allclose(cov_bary, compute_bary_cov(covariances))
        
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

class NaiveBaryMultiClassifierDestructor(ClassifierDestructor):
    
    def __init__(self, hist_bins='auto', n_grid=100, bound_eps='auto', weight=None):
        self.hist_bins = hist_bins
        self.n_grid = n_grid
        self.bound_eps = bound_eps
        self.weight = weight
    
    def _fit_conditional_transform(self, X, y=None):
        # fit independent destructor for each class

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
        
        # Fit independent destructor for each class 
        # which can destroy the data to uniform distribution
        fitted_class_transformers = []
        for i, yy in enumerate(classes):
            d = clone(ind)
            d.fit(X[y==yy, :])
            fitted_class_transformers.append(d)
            
        # Save for post_transform
        self.conditional_transformer_ = ConditionalTransformer.create_fitted(fitted_class_transformers, classes)
        return self.conditional_transformer_
    
    def _fit_post_transform(self, X, y=None):
        #assert len(self.conditional_transformer_.classes_) == 2, 'Only 2 classes is implemented'
        classes = unique_labels(y)
        n_classes = len(classes)
        n_features = X.shape[1]
        weight = self.weight
        if weight is None:
            weight = np.ones((1,n_classes)) / n_classes
        assert weight.shape[0] == 1

        X = check_array(X)
        n_features = X.shape[1]
        X = None # Only need X for n_features
        d_bary_inv = _get_ind_bary_inv_transformer(
            self.conditional_transformer_, self.n_grid, 
            self.bound_eps, n_features, weight, is_unit_bounded=False)
        self.post_transformer_ = d_bary_inv
        return self.post_transformer_
    

def _get_ind_bary_inv_transformer(conditional_transformer, n_grid, bound_eps, 
                                 n_features, weight, is_unit_bounded=False,):
    # -- Construct barycenter transform via McCann's interpolation
    # Start by finding quantiles all features
    # Quantiles bounded away from 0 and 1
    # Go in both directions via inverse transforms
    if bound_eps == 'auto':
        bound_eps = 1/n_grid


    # transform the histogram using the inverse transform just fitted for 
    # each class and take the average
    # based on default, 0.005, 0.015,... 0.995 shape = (100,)
    u_bary = np.linspace(-0.5, 0.5, n_grid) * (1-bound_eps) + 0.5
    # based on result above, make a grid shape = (100,n_features)
    U_bary = np.outer(u_bary, np.ones(n_features))
    # shape = (n_classes, 100, n_features)
    X_query_per_class = np.array([
        t.inverse_transform(U_bary)
        for t in conditional_transformer.fitted_class_transformers_
    ])
    # average over classes
    #X_bary = np.mean(X_query_per_class, axis=0) # 
    X_bary = torch.zeros(X_query_per_class.shape[1], X_query_per_class.shape[2])
    for i,t in enumerate(X_query_per_class):
        X_bary += weight[0,i] * t

    if not is_unit_bounded:
        # Use Gaussian pre-destructor to handle tails
        normal_destructor = IndependentDestructor().fit(X_bary)
        X_bary = normal_destructor.transform(X_bary)

    #it the density of barycenter 
    # Form pseudo-histogram for d_bary that will serve as interpolation
    fitted = []
    # for each row
    for xb, ub in zip(X_bary.T, U_bary.T):  # each feature independently
        # hist should be pseudo-counts/mass (i.e., not density values)
        #  HistogramUnivariateDensity.create_fitted accounts for bin_edges
        # assign weight/mass for each bin
        hist = np.concatenate(([bound_eps/2], np.diff(ub), [bound_eps/2]))  # Mass in each bin is like spacing between u_bary
        d_hist = HistogramUnivariateDensity.create_fitted(
            hist, bin_edges=np.concatenate(([0], xb, [1]))) 
        fitted.append(d_hist)

    #fine the destructor based on the density just found 
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
