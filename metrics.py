import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.multiclass import unique_labels
import itertools


def transport_cost(X, Z, y, weight=None):
    '''
    Evaluate Transporation Cost
    '''
    
    X = torch.Tensor(X)
    Z = torch.Tensor(Z)
    classes = unique_labels(y)
    n_classes = len(classes)

    # set uniform weight as default
    if weight is None:
        weight = np.ones(n_classes) / n_classes 
    
    cost = 0
    # empirical expectation
    for n,l in enumerate(classes):
        sel = np.nonzero(y==l)[0]
        X_temp = X[sel]
        Z_temp = Z[sel]

        cost_temp = 0
        # find the squared l2-norm for each class
        for i,j in zip(X_temp, Z_temp):
            cost_temp += torch.dist(i, j)**2
        cost_temp = cost_temp/ X_temp.shape[0]
        
        # apply weight
        cost += cost_temp * weight[n]
    return cost.item()


def wd_sinkhorn(Z, y, weight= None):
    '''
    Evaluate the 2-Wasserstein Distance in the latent space using Sinkhorn Iterations 
    Designed for computing WD when input Z are samples from shared distributions transformed from each class
    '''
    
    Z = torch.Tensor(Z)
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
    classes = unique_labels(y)
    n_classes = len(classes)

    if weight is None:
        weight = np.ones(n_classes) / n_classes 

    Z_list = dict()
    for t in classes:
        Z_list[t] = Z[np.nonzero(y == t)[0]]

    #  find all possible combinations of Z_i and Z_j
    idx = list(itertools.combinations(classes, 2))
    n_scale = len(idx)
    wd = 0
    for m,n in idx:
        wd = wd + sinkhorn(Z_list[int(m)], Z_list[int(n)])
    wd = wd/ n_scale
    return wd


def wd_average(X, Z, y, n_samples, weight = None): 
    """  
    Number of classes k=2  
    Evaluate the 2-Wasserstein Distance in the original space space using Sinkhorn Iterations 
    X -- original distribution
    Z -- flipped distribution (in the original space)
    """    

    X = torch.Tensor(X)
    Z = torch.Tensor(Z)
    classes = unique_labels(y)
    n_classes = len(classes)
    # prepare flipped label
    y_flip = 1-y

    if weight is None:
        weight = np.ones(n_classes) / n_classes 

    wd = 0
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
    # empirical expectation
    for n,l in enumerate(classes):
        sel = np.nonzero(y==l)[0]
        X_temp = X[sel]
        # find Z from another class
        sel = np.nonzero(y_flip==l)[0]
        Z_temp = Z[sel]
        wd += sinkhorn(X_temp, Z_temp).detach().numpy() * weight[n]

    return wd

def wd_original(X, Xflipall, y, key, weight = None):
    """
    Number of classes k>2
    Evaluate the 2-Wasserstein Distance in the original space space using Sinkhorn Iterations 
    X -- original distribution
    Xflipall: dictionary that contains flipped X from all the other classes. The key of Xflipall 
              represents which X_i it is from
    """

    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
    class_list = list(unique_labels(y))
    n_classes = len(class_list)

    if weight is None:
        weight = np.ones(n_classes) / n_classes 

    # k(k-1) terms in total
    wd = 0
    for l in class_list:
        X_temp = X[np.nonzero(y == l)[0]]
        for m in class_list:
            if m != l:
                # for each m, find Xflip from X_m and compare with the real X_l
                Xflip_dict = Xflipall[m]
                Xflip = Xflip_dict[key]
                # find the fake X_l from X_m
                Xflip_temp = Xflip[np.nonzero(y == l)[0]]
                X_temp = torch.Tensor(X_temp)
                Xflip_temp = torch.Tensor(Xflip_temp)
                wd += sinkhorn(X_temp, Xflip_temp)
    wd = wd/(l-1)/l
    return wd.detach().numpy()


class SinkhornDistance(nn.Module):
    r"""
    from https://github.com/dfdazac/wassdistance/blob/master/layers.py

    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


