""" Dimensionality Reduction
@author: Soufiane Mourragui
This module extracts the domain-specific factors from the high-dimensional omics
dataset. Several methods are here implemented and they can be directly
called from string name in main method method. All the methods
use scikit-learn implementation.
Notes
-------
	-
	
References
-------
	[1] Pedregosa, Fabian, et al. (2011) Scikit-learn: Machine learning in Python.
	Journal of Machine Learning Research
"""

import numpy as np
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, NMF, SparsePCA
from sklearn.cross_decomposition import PLSRegression


def process_dim_reduction(method='pca', n_dim=10):
    """
    Default linear dimensionality reduction method. For each method, return a
    BaseEstimator instance corresponding to the method given as input.
	Attributes
    -------
    method: str, default to 'pca'
    	Method used for dimensionality reduction.
    	Implemented: 'pca', 'ica', 'fa' (Factor Analysis), 
    	'nmf' (Non-negative matrix factorisation), 'sparsepca' (Sparse PCA).
    
    n_dim: int, default to 10
    	Number of domain-specific factors to compute.
    Return values
    -------
    Classifier, i.e. BaseEstimator instance
    """

    if method.lower() == 'pca':
        clf = PCA(n_components=n_dim)

    elif method.lower() == 'ica':
        print('ICA')
        clf = FastICA(n_components=n_dim)

    elif method.lower() == 'fa':
        clf = FactorAnalysis(n_components=n_dim)

    elif method.lower() == 'nmf':
        clf = NMF(n_components=n_dim)

    elif method.lower() == 'sparsepca':
        clf = SparsePCA(n_components=n_dim, alpha=10., tol=1e-4, verbose=10, n_jobs=1)

    elif method.lower() == 'pls':
        clf = PLS(n_components=n_dim)
		
    else:
        raise NameError('%s is not an implemented method'%(method))

    return clf


class PLS():
    """
    Implement PLS to make it compliant with the other dimensionality
    reduction methodology.
    (Simple class rewritting).
    """
    def __init__(self, n_components=10):
        self.clf = PLSRegression(n_components)

    def get_components_(self):
        return self.clf.x_weights_.transpose()

    def set_components_(self, x):
        pass

    components_ = property(get_components_, set_components_)

    def fit(self, X, y):
        self.clf.fit(X,y)
        return self

    def transform(self, X):
        return self.clf.transform(X)

    def predict(self, X):
        return self.clf.predict(X)