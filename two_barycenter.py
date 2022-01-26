import torch
import torch.nn as nn
import numpy as np
import scipy.stats
from sklearn.utils.multiclass import unique_labels

from metrics import transport_cost, SinkhornDistance, wd_sinkhorn
from base_gaussian import GaussianCDF, GaussianInverseCDF
from weakflow._gaussian_bary import GaussianBaryClassifierDestructor
from weakflow._naive_bary import NaiveBaryClassifierDestructor
from ddl.base import CompositeDestructor, IdentityDestructor
from ddl.independent import IndependentDensity, IndependentDestructor, IndependentInverseCdf
from ddl.univariate import ScipyUnivariateDensity
from weakflow._tree import TreeDensity, DecisionTreeClassifier, TreeClassifierDestructor
from SWD import maxSWDdirection


# max Sliced Wasserstein Distance Barycenter Classifier Destructor
class MSWDBaryClassifierDestructor(nn.Module):
    
    def __init__(self):

        super().__init__()
        self.layer = nn.ModuleList([])
        self.num_layer = 0

    def forward(self, X, y):
        for i in range(len(self.layer)):
            X = self.layer[i](X, y)
        return X

    def inverse(self, X, y):
        for i in reversed(range(len(self.layer))):
            X = self.layer[i].inverse(X, y)
        return X

    def partial_forward(self,X,y,last_layer,toy=True):
        for i in range(last_layer):
            X = self.layer[i](X, y)
        if not toy:            
            X = self.layer[-1](X,y)
        return X
        
    def partial_inverse(self,X,y,last_layer,toy=True):
        if not toy:            
            X = self.layer[-1].inverse(X, y)
        for i in reversed(range(last_layer)):
            X = self.layer[i].inverse(X, y)
        return X

    # Add destructor as needed 
    def add_layer(self,layer):
        self.layer.append(layer)
        self.num_layer += 1
        return self
    
    # For the transformation in the unit space
    def initialize(self, X, y):
        cdf = GaussianInverseCDF()
        Z = cdf.fit_transform(X, y)
        self.layer.append(cdf)
        self.num_layer += 1
        return Z

    # For the transformation in the unit space
    def end(self, X, y):
        cdf = GaussianCDF()
        Z = cdf.fit_transform(X, y)
        self.layer.append(cdf)
        self.num_layer += 1
        return Z

    # keep track of wd in the latent space in the whole iteration
    def eval_wd(self, X, y, sep=5):
        layer_list = []
        wd_list = []
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
        Z = X
        for i in range(self.num_layer):
            Z = self.layer[i](Z, y) 
            Z_temp = Z
            if i % sep == 0 or i == self.num_layer-1 or i==0:
                layer_list.append(i)
                wd_list.append(wd_sinkhorn(Z_temp, y).detach().numpy())
        return wd_list, layer_list

    # keep track of transporation cost in the whole iteration
    def eval_tcost(self, X, y, sep=5, gau=True):
        # the first layer does nothing
        layer_list = []
        tcost_list = []
        if gau:
            Z = self.layer[0](X, y) 
            for i in range(1, self.num_layer-1):
                Z = self.layer[i](Z, y) 
                Z_temp = GaussianCDF().fit_transform(Z, y)
                if i % sep == 0 or i == self.num_layer-2 or i==1:
                    layer_list.append(i)
                    tcost_list.append(transport_cost(X, Z_temp, y))
        elif gau is False:
            Z = X
            for i in range(self.num_layer):
                Z = self.layer[i](Z, y) 
                Z_temp = Z
                if i % sep == 0 or i == self.num_layer-1 or i==0:
                    layer_list.append(i)
                    tcost_list.append(transport_cost(X, Z_temp, y))

        return tcost_list, layer_list


class MSWDBary(nn.Module):

    # mSWD based destructor
    
    def __init__(self, nfeatures, ndim, bary_type):
        # nfeatures: dimensions in original space
        # ndim: dimensions after projection

        super().__init__()
        self.nfeatures = nfeatures
        self.ndim = ndim

        # initialize w
        wi = torch.randn(self.nfeatures, self.ndim)
        Q, R = torch.qr(wi)
        L = torch.sign(torch.diag(R))
        wT = (Q * L)
        self.wT = nn.Parameter(wT)

        self.bary_type = bary_type
        self.bary = IdentityDestructor() # initialize with identiy
        
    def fit_wT(self, X, y, ndim=16, MSWD_p=2, MSWD_max_iter=200):
        # find the projection matrix
        # modified from https://github.com/biweidai/SIG_GIS/blob/master/SIT.py

        classes = unique_labels(y)
        X_list = []
        for t in classes:
            X_list.append(X[np.nonzero(y == t)[0]])
        assert len(X_list) == 2
        X0 = X_list[0]
        X1 = X_list[1]
        assert X0.shape[1] == X1.shape[1]
        wT, SWD = maxSWDdirection(X0, X1, n_component=ndim, maxiter=MSWD_max_iter, p=MSWD_p)
        with torch.no_grad():
            SWD, indices = torch.sort(SWD, descending=True)
            wT = wT[:,indices]
            self.wT[:] = torch.qr(wT)[0] 
        
        return self
    
    def fit_wT_rand(self, X, y, ndim=16):
        # for random projection compared to mSWD

        wi = torch.randn(X.shape[1], ndim, device=X.device)
        self.wT[:] = torch.qr(wi)[0]
        return self

    def fit_bary(self, X, y, max_leaf_nodes = 10, uniform_weight = 0.1, track_marginals = False):
        # fit the specified destructor

        if self.bary_type == 'nb':
            cd = NaiveBaryClassifierDestructor()
        elif self.bary_type == 'gb':
            cd = GaussianBaryClassifierDestructor()
        elif self.bary_type == 'gbnb':
            cd = CompositeDestructor([
                GaussianBaryClassifierDestructor(),
                NaiveBaryClassifierDestructor(),
            ])
        elif self.bary_type == 'tree':
            # max_leaf_nodes = 10
            # uniform_weight = 0.1
            # track_marginals = False
            std_inverse = IndependentInverseCdf()
            std_normal = IndependentDestructor(IndependentDensity(ScipyUnivariateDensity(
                scipy_rv=scipy.stats.norm, scipy_fit_kwargs=dict(floc=0, fscale=1)
            )))

            cd = CompositeDestructor([
                    std_normal,
                    TreeClassifierDestructor(
                    tree_classifier=DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes),
                    tree_density=TreeDensity(uniform_weight=uniform_weight),
                    track_marginals=track_marginals,), 
                    std_inverse])
        
        # fit the destrutor after the projection
        Xm = X @ self.wT
        cd.fit(Xm.detach().numpy(), y)
        
        self.bary = cd

        return self

    def transform(self, X, y, mode='forward'):
        
        X = torch.Tensor(X)
        Xm = X @ self.wT
        remaining = X - Xm @ self.wT.T
        if mode == 'forward':
            z = torch.Tensor(self.bary.transform(Xm.detach().numpy(), y))
        elif mode == 'inverse':
            z = torch.Tensor(self.bary.inverse_transform(Xm.detach().numpy(), y))

        X = remaining + z @ self.wT.T

        return X

    def forward(self, X, y):
        return self.transform(X, y, mode='forward')        

    def inverse(self, X, y):
        return self.transform(X, y, mode='inverse')  


class OriginalBary(nn.Module):

    # classifier destructor without projection
    
    def __init__(self, bary_type):
        super().__init__()
        self.bary_type = bary_type
        if bary_type == 'gb':
            self.bary = GaussianBaryClassifierDestructor()
        elif bary_type == 'nb':
            self.bary = NaiveBaryClassifierDestructor()
        elif self.bary_type == 'gbnb':
            self.bary = CompositeDestructor([
                GaussianBaryClassifierDestructor(),
                NaiveBaryClassifierDestructor(),
            ])
        elif bary_type == 'tree':
            max_leaf_nodes = 10
            uniform_weight = 0.1
            track_marginals = False
            std_inverse = IndependentInverseCdf()
            std_normal = IndependentDestructor(IndependentDensity(ScipyUnivariateDensity(
                scipy_rv=scipy.stats.norm, scipy_fit_kwargs=dict(floc=0, fscale=1)
            )))

            self.bary = CompositeDestructor([
                    std_normal,
                    TreeClassifierDestructor(
                    tree_classifier=DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes),
                    tree_density=TreeDensity(uniform_weight=uniform_weight),
                    track_marginals=track_marginals,), 
                    std_inverse])

    def fit_transform(self, X, y):
        Z = self.bary.fit_transform(X, y)
        return Z

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self
    
    def forward(self, X, y):
        Z = self.bary.transform(X, y)
        return Z

    def inverse(self, X, y):
        try:
            Z =self.bary.inverse_transform(X.detach().numpy(), y)
        except:
            Z =self.bary.inverse_transform(X, y)
        return Z


def add_one_layer(model, X, y, bary_type, ndim, rand=False, max_leaf_nodes = 10, uniform_weight = 0.1, track_marginals = False):
    
    X = torch.Tensor(X)
    nfeatures = X.shape[1]

    if bary_type == 'nb':
        layer = MSWDBary(nfeatures, ndim, 'nb')
    elif bary_type == 'gb':
        layer = MSWDBary(nfeatures, ndim, 'gb')
    elif bary_type =='gbnb':
        layer = MSWDBary(nfeatures, ndim, 'gbnb')
    elif bary_type =='tree':
        layer = MSWDBary(nfeatures, ndim, 'tree')
    
    # Use half of the data to fit the direction and the other half of the data to fit the destructor
    tn_samples = X.shape[0]
    n_samples = int(tn_samples/2)
    assert n_samples == tn_samples/2
    n_parts = int(np.floor(n_samples/2))

    # use the same amount of data from each class
    # the input should have the structure as X=[X1,X2],y=[y1,y2]
    X0 = X[:n_samples]
    y0 = y[:n_samples]
    X0 = X0[torch.randperm(X0.shape[0])]
    X1 = X[n_samples:]
    y1 = y[n_samples:]
    X1 = X1[torch.randperm(X1.shape[0])]

    X_p1 = torch.cat((X0[:n_parts], X1[:n_parts]))
    y_p1 = np.concatenate((y0[:n_parts], y1[:n_parts]))
    X_p2 = torch.cat((X0[n_parts:], X1[n_parts:]))
    y_p2 = np.concatenate((y0[n_parts:], y1[n_parts:]))
    
    if rand:
        layer.fit_wT_rand(X_p1, y_p1, ndim=ndim)
    else:
        layer.fit_wT(X_p1, y_p1, ndim=ndim, MSWD_p=2, MSWD_max_iter=200)

    layer.fit_bary(X_p2, y_p2, max_leaf_nodes = max_leaf_nodes, uniform_weight = uniform_weight, track_marginals = track_marginals)

    Z = layer(X, y)

    model.add_layer(layer)

    return model, Z


def add_one_layer_ori(model, X, y, bary_type):

    X = torch.Tensor(X)

    if bary_type == 'nb':
        layer = OriginalBary('nb')
    elif bary_type == 'gb':
        layer = OriginalBary('gb')
    elif bary_type =='gbnb':
        layer = OriginalBary('gbnb')

    layer.fit(X, y)

    Z = layer(X, y)
    model.add_layer(layer)

    return model, torch.Tensor(Z)