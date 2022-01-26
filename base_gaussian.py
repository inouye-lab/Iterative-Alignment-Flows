import scipy.stats
import torch.nn as nn
from ddl.independent import IndependentDensity, IndependentDestructor, IndependentInverseCdf
from ddl.univariate import ScipyUnivariateDensity

# Used for pre and post processing data
class GaussianInverseCDF(nn.Module):
    '''
    Wrap scikit-learn based method in Pytorch form
    '''
    def __init__(self):

        super().__init__()
        self.cdf = IndependentInverseCdf()

    def fit_transform(self, X, y):
        Z = self.cdf.fit_transform(X, y)
        return Z

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self
    
    def forward(self, X, y):
        Z = self.cdf.transform(X, y)
        return Z

    def inverse(self, X, y):
        try:
            Z =self.cdf.inverse_transform(X.detach().numpy(), y)
        except:
            Z =self.cdf.inverse_transform(X, y)
        return Z

class GaussianCDF(nn.Module):
    '''
    Wrap scikit-learn based method in Pytorch form
    '''
    def __init__(self):

        super().__init__()
        self.cdf = IndependentDestructor(IndependentDensity(ScipyUnivariateDensity(
            scipy_rv=scipy.stats.norm, scipy_fit_kwargs=dict(floc=0, fscale=1)
        )))

    def fit_transform(self, X, y):
        try:
            Z = self.cdf.fit_transform(X.detach().numpy(), y)
        except:
            Z = self.cdf.fit_transform(X, y)
        return Z

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self
    
    def forward(self, X, y):
        try:
            Z = self.cdf.transform(X.detach().numpy(), y)
        except:
            Z = self.cdf.transform(X, y)
        return Z

    def inverse(self, X, y):
        Z =self.cdf.inverse_transform(X, y)
        return Z
