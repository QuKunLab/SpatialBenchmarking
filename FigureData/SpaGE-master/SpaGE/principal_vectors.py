""" Principal Vectors
@author: Soufiane Mourragui
This module computes the principal vectors from two datasets, i.e.:
- perform linear dimensionality reduction independently for both dataset, resulting
in set of domain-specific factors.
- find the common factors using principal vectors [1]
This result in set of pairs of vectors. Each pair has one vector from the source and one
from the target. For each pair, a similarity score (cosine similarity) can be computed
between the principal vectors and the pairs are naturally ordered by decreasing order
of this similarity measure.
Example
-------
    Examples are given in the vignettes.
Notes
-------
	Examples are given in the vignette
	
References
-------
	[1] Golub, G.H. and Van Loan, C.F., 2012. "Matrix computations" (Vol. 3). JHU Press.
	[2] Mourragui, S., Loog, M., Reinders, M.J.T., Wessels, L.F.A. (2019)
    PRECISE: A domain adaptation approach to transfer predictors of drug response
    from pre-clinical models to tumors
"""

import numpy as np
import pandas as pd
import scipy
from pathlib import Path
from sklearn.preprocessing import normalize

from SpaGE.dimensionality_reduction import process_dim_reduction

class PVComputation:
    """
    Attributes
    -------
    n_factors: int
        Number of domain-specific factors to compute.
    n_pv: int
        Number of principal vectors.
    dim_reduction_method_source: str
        Dimensionality reduction method used for source data
    dim_reduction_target: str
        Dimensionality reduction method used for source data
    source_components_ : numpy.ndarray, shape (n_pv, n_features)
        Loadings of the source principal vectors ranked by similarity to the
        target. Components are in the row.
    source_explained_variance_ratio_: numpy.ndarray, shape (n_pv)
        Explained variance of the source on each source principal vector.
    target_components_ : numpy.ndarray, shape (n_pv, n_features)
        Loadings of the target principal vectors ranked by similarity to the
        source. Components are in the row.
    target_explained_variance_ratio_: numpy.ndarray, shape (n_pv)
        Explained variance of the target on each target principal vector.
    cosine_similarity_matrix_: numpy.ndarray, shape (n_pv, n_pv)
        Scalar product between the source and the target principal vectors. Source
        principal vectors are in the rows while target's are in the columns. If
        the domain adaptation is sensible, a diagonal matrix should be obtained.
    """

    def __init__(self, n_factors,n_pv,
                dim_reduction='pca',
                dim_reduction_target=None,
                project_on=0):
        """
        Parameters
        -------
        n_factors : int
            Number of domain-specific factors to extract from the data (e.g. using PCA, ICA).
        n_pv : int
            Number of principal vectors to find from the independently computed factors.
        dim_reduction : str, default to 'pca' 
            Dimensionality reduction method for the source data,
            i.e. 'pca', 'ica', 'nmf', 'fa', 'sparsepca', pls'.
        dim_reduction_target : str, default to None 
            Dimensionality reduction method for the target data,
            i.e. 'pca', 'ica', 'nmf', 'fa', 'sparsepca', pls'. If None, set to dim_reduction.
    	project_on: int or bool, default to 0
    		Where data should be projected on. 0 means source PVs, -1 means target PVs and 1 means
            both PVs.
        """
        self.n_factors = n_factors
        self.n_pv = n_pv
        self.dim_reduction_method_source = dim_reduction
        self.dim_reduction_method_target = dim_reduction_target or dim_reduction
        self.dim_reduction_source = self._process_dim_reduction(self.dim_reduction_method_source)
        self.dim_reduction_target = self._process_dim_reduction(self.dim_reduction_method_target)

        self.source_components_ = None
        self.source_explained_variance_ratio_ = None
        self.target_components_ = None
        self.target_explained_variance_ratio_ = None
        self.cosine_similarity_matrix_ = None

    def _process_dim_reduction(self, dim_reduction):
        if type(dim_reduction) == str:
            return process_dim_reduction(method=dim_reduction, n_dim=self.n_factors)
        else:
            return dim_reduction

    def fit(self, X_source, X_target, y_source=None):
        """
    	Compute the common factors between two set of data.
    	IMPORTANT: Same genes have to be given for source and target, and in same order
        Parameters
        -------
        X_source : np.ndarray, shape (n_components, n_genes)
            Source dataset
        X_target : np.ndarray, shape (n_components, n_genes)
            Target dataset
        y_source : np.ndarray, shape (n_components, 1) (optional, default to None)
            Eventual output, in case one wants to give ouput (for instance PLS)
        Return values
        -------
        self: returns an instance of self.
        """
        # Compute factors independently for source and target. Orthogonalize the basis
        Ps = self.dim_reduction_source.fit(X_source, y_source).components_
        Ps = scipy.linalg.orth(Ps.transpose()).transpose()

        Pt = self.dim_reduction_target.fit(X_target, y_source).components_
        Pt = scipy.linalg.orth(Pt.transpose()).transpose()

        # Compute the principal factors
        self.compute_principal_vectors(Ps, Pt)

        # Compute variance explained
        self.source_explained_variance_ratio_ = np.var(self.source_components_.dot(X_source.transpose()), axis=1)/\
                                                np.sum(np.var(X_source), axis=0)
        self.target_explained_variance_ratio_ = np.var(self.target_components_.dot(X_target.transpose()), axis=1)/\
                                                np.sum(np.var(X_target), axis=0)

        return self

    def compute_principal_vectors(self, source_factors, target_factors):
        """
    	Compute the principal vectors between the already computed set of domain-specific
        factors, using approach presented in [1,2].
    	IMPORTANT: Same genes have to be given for source and target, and in same order
        Parameters
        -------
    	source_factors: np.ndarray, shape (n_components, n_genes)
    		Source domain-specific factors.
    	target_factors: np.ndarray, shape (n_components, n_genes)
    		Target domain-specific factors.
        Return values
        -------
        self: returns an instance of self.
        """

        # Find principal vectors using SVD
        u,sigma,v = np.linalg.svd(source_factors.dot(target_factors.transpose()))
        self.source_components_ = u.transpose().dot(source_factors)[:self.n_pv]
        self.target_components_ = v.dot(target_factors)[:self.n_pv]
        # Normalize to make sure that vectors are unitary
        self.source_components_ = normalize(self.source_components_, axis = 1)
        self.target_components_ = normalize(self.target_components_, axis = 1)

        # Compute cosine similarity matrix
        self.initial_cosine_similarity_matrix_ = source_factors.dot(target_factors.transpose())
        self.cosine_similarity_matrix_ = self.source_components_.dot(self.target_components_.transpose())

        # Compute angles
        self.angles_ = np.arccos(np.diag(self.cosine_similarity_matrix_))

        return self


    def transform(self, X, project_on=None):
        """
    	Projects data onto principal vectors.
        Parameters
        -------
        X : numpy.ndarray, shape (n_samples, n_genes)
            Data to project.
        project_on: int or bool, default to None
            Where data should be projected on. 0 means source PVs, -1 means target PVs and 1 means
            both PVs. If None, set to class instance value.
    	Return values
        -------
        Projected data as a numpy.ndarray of shape (n_samples, n_factors)
        """

        project_on = project_on or self.project_on

        # Project on source
        if project_on == 0:
            return X.dot(self.source_components_.transpose())

        # Project on target
        elif project_on == -1:
            return X.dot(self.target_components_.transpose())

        # Project on both
        elif project_on == 1:
            return X.dot(np.concatenate([self.source_components_.transpose(), self.target_components_.transpose()]))

        else:
            raise ValueError('project_on should be 0 (source), -1 (target) or 1 (both). %s not correct value'%(project_on))