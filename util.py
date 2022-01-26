import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from sklearn.utils import check_random_state
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from ddl.base import (CompositeDestructor, DestructorMixin, create_inverse_transformer, 
                      BoundaryWarning, DataConversionWarning, IdentityDestructor)
from ddl.independent import IndependentDensity, IndependentDestructor, IndependentInverseCdf
from ddl.linear import LinearProjector
from ddl.univariate import HistogramUnivariateDensity, ScipyUnivariateDensity
from ddl.deep import DeepDestructor
from ddl.linear import LinearProjector, RandomOrthogonalEstimator, BestLinearReconstructionDestructor

from weakflow._density import DensityConditionalDestructor
from weakflow._tree import TreeDensity, DecisionTreeClassifier, TreeClassifierDestructor


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist_data = np.array(mnist.data)
    sorted_mnist_data = np.copy(mnist_data)
    sorted_mnist_data[:60000] = mnist_data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    sorted_mnist_data[60000:] = mnist_data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]
    return sorted_mnist_data,np.array(mnist.target)
    
def load_data(data_name, class_list = [0,1], test_size=1000):
    """
    data_name: fetch_openml dataset names
    MNIST: 'mnist_784'
    FMNIST: 'Fashion-MNIST'
    class_list: [0,1] as default
    """
    mnist = fetch_openml(data_name, version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
    #sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
        
    print(f'1. Attempting to load/fetch {data_name} data via sklearn.datasets.fetch_mldata.')
    #X_raw, y_raw = mnist.data, mnist.target
    X_raw, y_raw = sort_by_target(mnist)
    print('  Done! data shape = %s, max value = %g' % (str(X_raw.shape), np.max(X_raw)))

    # Add uniform noise to dequantize the images
    print('2. Normalizing values between 0 and 1.')
    random_state = 0
    rng = check_random_state(random_state)
    X_deq = (X_raw + rng.rand(*X_raw.shape)) / 256.0
    #X = (X_raw+0.5)/256.0
    print('  Done! After dequantization and normalization: min=%g, max=%g' % (np.min(X_deq), np.max(X_deq)))

    # Selecting only zeros and ones
    print(f'2a. Only selecting specific classes {class_list}')
    sel = []
    for i,t in enumerate(y_raw):
        if t in class_list:
            sel.append(i)
    X = X_deq[sel, :]
    y = y_raw[sel]

    # Create train and test splits of the data
    print('3. Setting up train and test sizes.')
    X_all = X
    y_all = y

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=rng)
    print(f'All: {X_all.shape}, Train: {X_train.shape}, Test: {X_test.shape})')
    return X_train, X_test, y_train, y_test


 
# Image plotting function
def plot_images(X, input_size=(1, 28, 28), fig_height=4, title=None, ylabel=None, vmin=-1, vmax=0):
    n_images = X.shape[0]
    fig, axes = plt.subplots(1, n_images, figsize=(fig_height * n_images, fig_height))
    if n_images == 1:
        axes = np.array([axes]) # Add dimension
    for x, ax in zip(X, axes):
        ax.imshow(-x.reshape(*input_size[1:]), cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None:
        fig.suptitle(title, fontsize=40)
    if ylabel is not None:
        axes[0].set_ylabel(ylabel, fontsize=40)    



# wrap cdf and inverse cdf of normal distribution
def wrap_transform(est):

    std_inverse = IndependentInverseCdf()
    std_normal = IndependentDestructor(IndependentDensity(ScipyUnivariateDensity(
        scipy_rv=scipy.stats.norm, scipy_fit_kwargs=dict(floc=0, fscale=1)
    )))
    return CompositeDestructor([std_inverse, est, std_normal])

# create deep density destructors
def create_deep_density_cd(n_canonical_destructors):
    copula_destructor = CompositeDestructor([
        IndependentDestructor(IndependentDensity(HistogramUnivariateDensity(
            bins=40, bounds=[0, 1], alpha=100))),
        IndependentInverseCdf(),
        BestLinearReconstructionDestructor(
            linear_estimator=PCA(), destructor=IndependentDestructor(),
            linear_projector_kwargs=dict(fit_bias=False))
    ])
    return DeepDestructor(
        init_destructor=DensityConditionalDestructor(
            density_destructor=CompositeDestructor([
                IndependentDestructor(IndependentDensity(HistogramUnivariateDensity(
                    bins=256, bounds=[0, 1], alpha=1)))
            ])),
        canonical_destructor=DensityConditionalDestructor(
            density_destructor=copula_destructor),
        n_canonical_destructors=n_canonical_destructors, 
        random_state=np.random.RandomState(0), # Random state set for all estimators
    )

# create deep tree destructors
def create_deep_tree_cd(n_canonical_destructors, max_leaf_nodes = 10, uniform_weight = 0.9, track_marginals = False):
    
    std_inverse = IndependentInverseCdf()
    std_normal = IndependentDestructor(IndependentDensity(ScipyUnivariateDensity(
        scipy_rv=scipy.stats.norm, scipy_fit_kwargs=dict(floc=0, fscale=1)
    )))

    # wrap tree destructors with cdf and inverse cdf of normal distribution
    copula_destructor = CompositeDestructor([
                        std_normal,
                        TreeClassifierDestructor(
                        tree_classifier=DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes),
                        tree_density=TreeDensity(uniform_weight=uniform_weight),
                        track_marginals=track_marginals,), 
                        std_inverse])
    return DeepDestructor(
        canonical_destructor=copula_destructor,
        n_canonical_destructors=n_canonical_destructors, # Random state set for all estimators
    )
