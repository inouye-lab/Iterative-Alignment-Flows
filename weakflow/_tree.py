import copy
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_X_y
from sklearn.utils.multiclass import unique_labels

from ddl.base import IdentityDestructor, CompositeDestructor, create_inverse_transformer
from ddl.univariate import HistogramUnivariateDensity
from ddl.independent import IndependentDestructor, IndependentDensity
from ddl.tree import TreeDensity

from . import ClassifierDestructor, ConditionalTransformer


class TreeClassifierDestructor(ClassifierDestructor):
    def __init__(self, tree_classifier=None, tree_density=None, class_weights=None,
                 track_marginals=True):
        self.tree_classifier = tree_classifier
        self.tree_density = tree_density
        self.class_weights = class_weights
        self.track_marginals = track_marginals
        
    def _fit_conditional_transform(self, X, y=None):
        X, y = check_X_y(X, y)
        if self.tree_classifier is not None:
            tree_classifier = clone(self.tree_classifier)
        else:
            tree_classifier = DecisionTreeClassifier()
        if self.tree_density is not None:
            tree_density = clone(self.tree_density)
        else:
            tree_density = TreeDensity()
        classes = unique_labels(y)
        if self.class_weights is None:
            class_weights = np.ones(len(classes))
            class_weights = class_weights / np.sum(class_weights)
        else:
            class_weights = self.class_weights
        assert np.allclose(np.sum(class_weights), 1)
        assert np.all(class_weights >= 0)
            
        # Fit tree structure
        tree_classifier.fit(X, y)
        
        # Fit tree densities for each class
        tree_densities = [
            clone(tree_density).fit(X[y == yy, :], fitted_tree_estimator=tree_classifier)
            for yy in classes
        ]
        
        class_transformers = _create_barycenter_transformers(
            tree_densities, class_weights, X.shape[1], self.track_marginals)
        
        # Save variables
        self.classes_ = classes
        self.class_weights_ = class_weights
        self.fitted_tree_densities_ = tree_densities
        self.fitted_tree_classifier_ = tree_classifier
        self.conditional_transformer_ = ConditionalTransformer.create_fitted(
            class_transformers, classes)
        return self.conditional_transformer_


def _create_barycenter_transformers(tree_densities, class_weights, n_features,
                                    track_marginals):
    """Create barycenter transforms."""
    # Sanity check that all densities have same structure and splits
    for i, node_arr in enumerate(zip(*[t.tree_ for t in tree_densities])):  # Must be depth-first and left-then-right traversal
        # Assert features and thresholds are the same (values can be different)
        is_leaf_arr = np.array([n.is_leaf() for n in node_arr])
        assert np.all(is_leaf_arr == is_leaf_arr[0])
        if np.sum(is_leaf_arr) > 0:
            features = np.array([n.feature for n in node_arr])
            assert np.all(features==features[0])
            thresholds = np.array([n.threshold for n in node_arr])
            assert np.all(thresholds==thresholds[0])
    
    # Original is depth-first and left-then-right traversal
    # Reversed allows bottom up traversal of nodes
    orig_trees = copy.deepcopy([t.tree_ for t in tree_densities])
    #print(tree_densities[0].tree_)
    for tr in orig_trees:
        # Needed because weighted barycenter
        _relative_to_absolute_probability(iter(tr), 1)
    forward_trees = copy.deepcopy(orig_trees)
    loop_vals = list(enumerate(
        zip(zip(*orig_trees), zip(*forward_trees))
    ))
    # Store the needed histogram array for each node 
    n_nodes = len(loop_vals)
    parallel_histogram_arr = np.array([dict() for _ in range(n_nodes)])
    #print([f'{i}:{node_arr[0].is_leaf()}' for i, (node_arr, forward_arr) in loop_vals])
    for i, (node_arr, forward_arr) in reversed(loop_vals):
        # Skip if leaf
        if node_arr[0].is_leaf():
            continue
        
        # Using absolute probability value of node and class weights to compute
        #  barycenter weights
        weight_vec = np.array([n.value for n in node_arr]) * class_weights
        weight_vec = weight_vec / np.sum(weight_vec)
        
        # Create histograms
        def _create_histogram(n):
            # Get left and right histograms 
            #  (create children histograms if needed, i.e., if they are leaves)
            def check_child(child):
                if child == 'left':
                    idx = n.left_child_index
                    default_bin_edges = [n.domain[n.feature, 0], n.threshold]
                else:
                    idx = n.right_child_index
                    default_bin_edges = [n.threshold, n.domain[n.feature, 1]]
                assert idx >= 0, 'Should be greater than zero since non-leaf node'
                if track_marginals:
                    hist_dict = parallel_histogram_arr[idx]
                else:
                    hist_dict = dict()
                if n.feature in hist_dict: # Only extract if available otherwise default
                    hist = hist_dict[n.feature]
                else:
                    # Uniform histogram for children that are leaves
                    hist = HistogramUnivariateDensity.create_fitted([1], default_bin_edges)
                return hist
            left_hist, right_hist = check_child('left'), check_child('right')
            # Merge disjoint histograms of children via left and right relative probabilities
            # Scaled bin probabilities (diff of cdf values)
            values = np.concatenate([
                prob_child * np.diff(hist.rv_._hcdf)
                for prob_child, hist in zip(n.destructor, [left_hist, right_hist])
            ])
            #   Check that bounds meet at threshold
            assert np.allclose(left_hist.rv_._hbins[-1], right_hist.rv_._hbins[0]), 'Edges should match'
            bin_edges = np.concatenate([
                left_hist.rv_._hbins[:-1], right_hist.rv_._hbins
            ])
            return HistogramUnivariateDensity.create_fitted(values, bin_edges)
        histograms = [_create_histogram(n) for n in node_arr]
        
        # Compute barycenter histogram
        bary_hist = _compute_barycenter(histograms, weight_vec)
        
        # Maintain marginal distributions
        #  Copy left child
        if track_marginals:
            n0 = node_arr[0]
            marginals_merged = parallel_histogram_arr[n0.left_child_index].copy()
            marginals_right = parallel_histogram_arr[n0.right_child_index]
            #  Merge with right child (skipping current feature)
            for ii in range(n_features):
                if ii == n0.feature:
                    continue
                if ii in marginals_merged and ii in marginals_right:
                    prob_left, prob_right = n0.destructor
                    prob_sum = prob_left + prob_right
                    marginals_merged[ii] = _mixture_hist(
                        marginals_merged[ii], marginals_right[ii],
                        prob_left/prob_sum, prob_right/prob_sum)
                elif ii in marginals_right:
                    marginals_merged[ii] = marginals_right[ii]
                #raise RuntimeError('Have not implemented merge/mixture yet')
            # Add barycenter to this node for this feature
            marginals_merged[n0.feature] = bary_hist
            parallel_histogram_arr[n0.node_i] = marginals_merged
        
        # Form transform with a simple 1D array as input (need to reshape stuff)
        def _create_forward(hist, bary_hist):
            return CompositeDestructor.create_fitted([
                IndependentDestructor.create_fitted(
                    IndependentDensity.create_fitted([hist])),
                create_inverse_transformer(
                    IndependentDestructor.create_fitted(
                        IndependentDensity.create_fitted([bary_hist]))
                ),
            ])
        # Loop over classes
        for yy, (hist, forward_n) in enumerate(zip(
                histograms, forward_arr)):
            forward_n.destructor = _create_forward(hist, bary_hist)
            forward_n.value = np.nan
            # Note: inverse threshold can be figured out automatically since top down
            # Note: Bounds are the same in both cases as well so no need to adjust
            
    barycenter_transforms = [
        _TreeValueTransformer.create_fitted(forward_tree)
        for forward_tree in forward_trees
    ]
    return barycenter_transforms
    

def _relative_to_absolute_probability(tree_depth_iter, prob):
    """Change from relative to relative probabilities at the leaves.
    Hacky way of using the "destructor" attribute since the value isn't checked.
    Modifies the tree in place.
    """
    node = next(tree_depth_iter)
    if node.is_leaf():
        # Return absolute probability of node and reset to nan
        node.value = prob
        node.destructor = (None, None)
    else:
        left_relative = prob * node.value
        right_relative = prob * (1 - node.value)
        _relative_to_absolute_probability(tree_depth_iter, left_relative)
        _relative_to_absolute_probability(tree_depth_iter, right_relative)
        
        node.value = prob
        node.destructor = (left_relative, right_relative)

    
class _TreeValueTransformer():
    # This will be used for each class and for inversion
    # 1D transforms along split dimension 
    #  (nothing at leaves themselves, though could have node transformers)
    # Use value for transform (inverse and forward)
    def fit(self, X, y=None):
        raise RuntimeError('This class cannot be fitted directly. '
                           'See `create_fitted`.')
        
    @classmethod
    def create_fitted(cls, forward_tree, **kwargs):
        t = cls(**kwargs)
        t.forward_tree_ = forward_tree
        return t

    def transform(self, X, y=None):
        return self._tree_transform(self.forward_tree_, X, y, inverse=False)
    
    def inverse_transform(self, X, y=None):
        return self._tree_transform(self.forward_tree_, X, y, inverse=True)
    
    def _tree_transform(self, tree, X, y=None, inverse=False):
        X = check_array(X, copy=True, dtype=np.float)
        n_samples, n_features = X.shape
        
        # Compute selections (i.e., which nodes fall into which part of the tree)
        # TODO: Figure out how to do this without saving the selections for every node
        def _update_stack(child):
            """Add left or right child to stack and update selection
            Note this is late binding for outer variables so it should only be
            called when sel, t have been defined (i.e.
            inside the node loop). Defined here so that the compiler doesn't keep
            creating new functions each time.
            """
            sel_new = sel.copy()
            if child == 'left':
                sel_new[sel] = X[sel, node.feature] < t
            else:
                sel_new[sel] = X[sel, node.feature] >= t
            
            # Push onto stack
            stack.append(sel_new)
            
        if not inverse:
            #FORWARD: Get selections by going in normal order
            #  Then reverse so we go bottom up
            sel = np.ones(n_samples, dtype=np.bool)
            stack = [sel]
            sel_arr = []
            for node in tree:
                sel = stack.pop()
                if node.is_leaf():
                    sel_arr.append(None)
                else:
                    sel_arr.append(sel)
                    # Add children to stack
                    t = node.threshold
                    _update_stack('right')
                    _update_stack('left')
            # Choose order based on inverse or forward
            node_iter = reversed(list(enumerate(zip(tree, sel_arr))))
            # Loop through nodes and apply transform to selected data
            for i, (node, sel) in node_iter:
                #print(f'Processing node:{node.node_i}:{node.is_leaf()}')
                if node.is_leaf(): continue
                if np.sum(sel) > 0:
                    destructor = node.destructor
                    X[sel, node.feature] = destructor.transform(
                        X[sel, node.feature].reshape(-1, 1)
                    ).ravel()
        else:
            #INVERSE: Go in normal order (top down), 
            #  but transform before updating selection
            sel = np.ones(n_samples, dtype=np.bool)
            stack = [sel]
            for node in tree:
                sel = stack.pop()
                if node.is_leaf(): continue
                if np.sum(sel) > 0:
                    destructor = node.destructor
                    X[sel, node.feature] = destructor.inverse_transform(
                        X[sel, node.feature].reshape(-1, 1)
                    ).ravel()
                
                t = node.threshold
                _update_stack('right')
                _update_stack('left')

        # Cleanup values since numerical errors can cause them to fall
        #  outside of the destructor range of [0, 1]
        X = np.maximum(np.minimum(X, 1), 0)
        return X


def _mixture_hist(hist_a, hist_b, prob_a, prob_b):
    # Create histogram mixture of the two histograms
    # Loop through and keep track of probabilities
    def extract(h): return h.rv_._hbins, h.rv_._hpdf # First and last pdf are 0
    bins_a, pdf_a = extract(hist_a)
    bins_b, pdf_b = extract(hist_b)
    assert np.allclose(prob_a + prob_b, 1), 'Should be relative probabilities'

    # Initialize
    assert np.allclose(bins_a[0], bins_b[0]), 'bin starts should be the same'
    bins_new = [bins_a[0]] # Initial should be left bin edge
    prob_new = []
    ia, ib = 1, 1
    def add_split(next_split):
        width = next_split - bins_new[-1]
        prob = width * (prob_a*pdf_a[ia] + prob_b*pdf_b[ib])
        bins_new.append(next_split)
        prob_new.append(prob)
    splits = np.unique(np.concatenate([bins_a, bins_b]))
    while True:
        if bins_a[ia] < bins_b[ib]:
            # Compute probability
            add_split(bins_a[ia])
            ia += 1
        else:
            add_split(bins_b[ib])
            if bins_a[ia] == bins_b[ib]:
                ia += 1
            ib += 1
        if ia == len(bins_a) or ib == len(bins_a):
            break
    assert len(bins_a) - ia + len(bins_b) - ib <= 1, 'Should only be one left'
    return HistogramUnivariateDensity.create_fitted(prob_new, bins_new)


def _check_histogram(hist):
    assert np.all(~np.isnan(hist.rv_._hpdf))
    prob = np.diff(hist.rv_._hcdf)
    

    
def _compute_barycenter(histograms, weight_vec):
    # Calculate barycenter by sorting all unique quantiles uq (evaluate cdf at split),
    #  then evaluate inverse_cdf for all these points and average to get xq
    # Simple way for single split t
    #u = np.unique(np.array([d.rv_.cdf(t) for d in densities]))
    # More general way by evaluating cdf at bin edges 
    #  (except for first and last whose CDF will always be 0 and 1)
    u = np.unique(np.concatenate([d.rv_.cdf(d.rv_._hbins) for d in histograms]))
    min_x = np.min([d.rv_._hbins[0] for d in histograms])
    max_x = np.max([d.rv_._hbins[-1] for d in histograms])
    x = np.dot(weight_vec, np.array([d.rv_.ppf(u) for d in histograms]))
    # Handle case where x overlaps (i.e., 2 or more densities have same exact value)
    x, idx = np.unique(x, return_index=True)
    u = u[idx]
    # Extract bin_edges and probabilities
    bin_edges = x
    prob = np.diff(u)
    bary = HistogramUnivariateDensity.create_fitted(prob, bin_edges)
    for h in histograms:
        _check_histogram(h)
    _check_histogram(bary)
    return bary