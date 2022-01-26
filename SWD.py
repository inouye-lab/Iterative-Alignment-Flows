import torch
import numpy as np
from sklearn.utils.multiclass import unique_labels
import itertools

# Codes in this file are modified based on https://github.com/biweidai/SINF/blob/master/sinf/SlicedWasserstein.py
# Though we used the old version of their implementation
# The link to their old repository is listed below (Update: it has expired)

def SlicedWassersteinDistance(x, x2, q, p, perdim=True):
    '''
    Modified from https://github.com/biweidai/SIG_GIS/blob/master/SlicedWasserstein.py
    '''
    px = torch.sort(x, dim=-1)[0]
    px2 = torch.sort(x2, dim=-1)[0]

    if perdim:
        WD = torch.mean(torch.abs(px-px2) ** p)
    else:
        WD = torch.mean(torch.abs(px-px2) ** p, dim=-1)
    return WD


def maxSWDdirection(x, x2, n_component=None, maxiter=200, Npercentile=None, p=2, eps=1e-6):
    '''
    Modified from https://github.com/biweidai/SIG_GIS/blob/master/SlicedWasserstein.py
    '''
    assert x.shape[0] == x2.shape[0]
    assert x.shape[1] == x2.shape[1]
    
    ndim = x.shape[1]
    if n_component is None:
        n_component = ndim
    q = None
    
    #initialize w. algorithm from https://arxiv.org/pdf/math-ph/0609050.pdf
    wi = torch.randn(ndim, n_component, device=x.device)
    Q, R = torch.qr(wi)
    L = torch.sign(torch.diag(R))
    w = (Q * L).T

    lr = 0.1
    down_fac = 0.5
    up_fac = 1.5
    c = 0.5
    
    #algorithm from http://noodle.med.yale.edu/~hdtag/notes/steifel_notes.pdf
    #note that here w = X.T
    #use backtracking line search
    w1 = w.clone()
    w.requires_grad_(True)
    loss = -SlicedWassersteinDistance(w @ x.T, w @ x2.T, q, p)
    loss1 = loss
    for i in range(maxiter):
        GT = torch.autograd.grad(loss, w)[0]
        w.requires_grad_(False)
        WT = w.T @ GT - GT.T @ w
        e = - w @ WT #dw/dlr
        m = torch.sum(GT * e) #dloss/dlr

        lr /= down_fac
        while loss1 > loss + c*m*lr:
            lr *= down_fac
            if 2*n_component < ndim:
                UT = torch.cat((GT, w), dim=0).double()
                V = torch.cat((w.T, -GT.T), dim=1).double()
                w1 = (w.double() - lr * w.double() @ V @ torch.pinverse(torch.eye(2*n_component, dtype=torch.double, device=x.device)+lr/2*UT@V) @ UT).to(torch.get_default_dtype())
            else:
                w1 = (w.double() @ (torch.eye(ndim, dtype=torch.double, device=x.device)-lr/2*WT.double()) @ torch.pinverse(torch.eye(ndim, dtype=torch.double, device=x.device)+lr/2*WT.double())).to(torch.get_default_dtype())
            
            w1.requires_grad_(True)

            loss1 = -SlicedWassersteinDistance(w1 @ x.T, w1 @ x2.T, q, p)
        
        if torch.max(torch.abs(w1-w)) < eps:
            w = w1
            break
        
        lr *= up_fac
        loss = loss1
        w = w1

    WD = SlicedWassersteinDistance(w @ x.T, w @ x2.T, q, p, perdim=False)
    return w.T, WD**(1/p)


def barySlicedWassersteinDistance(w, x_dict, q, p, n_classes, perdim=True):
    '''
    Modified from https://github.com/biweidai/SIG_GIS/blob/master/SlicedWasserstein.py
    Designed for calculating distance for k>2
    '''
    for i, x in enumerate(x_dict.values()):
        if i == 0:
            bary = torch.sort(w @ x.T, dim=-1)[0]
        else:
            bary += torch.sort(w @ x.T, dim=-1)[0]
    bary = bary / n_classes
    
    for i,x in enumerate(x_dict.values()):
        x1 = torch.sort(w@x.T, dim=-1)[0]
        if i == 0:
            if perdim:
                WD = torch.mean(torch.abs(x1-bary) ** p)
            else:
                WD = torch.mean(torch.abs(x1-bary) ** p, dim=-1)
        else:
            if perdim:
                WD += torch.mean(torch.abs(x1-bary) ** p)
            else:
                WD += torch.mean(torch.abs(x1-bary) ** p, dim=-1)
    return WD


def multimaxSWDdirection(X, y, n_component=None, maxiter=200, Npercentile=None, p=2, eps=1e-6, weight=None):

    '''
    Modified from https://github.com/biweidai/SIG_GIS/blob/master/SlicedWasserstein.py
    Designed for calculating distance for k>2
    '''
    # unifrom weight if no weight is assigned
    classes = unique_labels(y)
    n_classes = len(classes)
    if weight is None:
        weight = np.ones((1,n_classes)) / n_classes
    assert weight.shape[0] == 1

    X_list = dict()
    for t in classes:
        X_list[t] = X[np.nonzero(y == t)[0]]

    ndim = X.shape[1]
    if n_component is None:
        n_component = ndim

    q = None

    #initialize w. algorithm from https://arxiv.org/pdf/math-ph/0609050.pdf
    wi = torch.randn(ndim, n_component, device=X.device)
    Q, R = torch.qr(wi)
    L = torch.sign(torch.diag(R))
    w = (Q * L).T

    lr = 0.1
    down_fac = 0.5
    up_fac = 1.5
    c = 0.5
    
    #algorithm from http://noodle.med.yale.edu/~hdtag/notes/steifel_notes.pdf
    #note that here w = X.T
    #use backtracking line search
    w1 = w.clone()
    w.requires_grad_(True)

    idx = list(itertools.combinations(classes, 2))
    n_scale = len(idx)

    loss = -barySlicedWassersteinDistance(w,X_list,q,p,n_classes)
    loss1 = loss
    for i in range(maxiter):
        GT = torch.autograd.grad(loss, w)[0]
        w.requires_grad_(False)
        WT = w.T @ GT - GT.T @ w
        e = - w @ WT #dw/dlr
        m = torch.sum(GT * e) #dloss/dlr
        lr /= down_fac
        z = 0
        while loss1 > loss + c*m*lr:
            lr *= down_fac
            if 2*n_component < ndim:
                UT = torch.cat((GT, w), dim=0).double()
                V = torch.cat((w.T, -GT.T), dim=1).double()
                w1 = (w.double() - lr * w.double() @ V @ torch.pinverse(torch.eye(2*n_component, dtype=torch.double, device=X.device)+lr/2*UT@V) @ UT).to(torch.get_default_dtype())
            else:
                w1 = (w.double() @ (torch.eye(ndim, dtype=torch.double, device=X.device)-lr/2*WT.double()) @ torch.pinverse(torch.eye(ndim, dtype=torch.double, device=X.device)+lr/2*WT.double())).to(torch.get_default_dtype())
            w1.requires_grad_(True)

            loss1 = - barySlicedWassersteinDistance(w1,X_list,q,p,n_classes)

        if torch.max(torch.abs(w1-w)) < eps:
            w = w1
            break
        
        lr *= up_fac
        loss = loss1
        w = w1


    WD = barySlicedWassersteinDistance(w,X_list,q,p,n_classes,perdim=False)
    return w.T, WD**(1/p)
