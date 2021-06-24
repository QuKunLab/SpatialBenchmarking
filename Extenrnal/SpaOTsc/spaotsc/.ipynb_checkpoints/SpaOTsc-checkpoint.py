import pandas as pd
import numpy as np
import networkx as nx
import igraph
import ot
import dit
from dit.pid.lattice import sort_key
from astropy.stats import bayesian_blocks
from astropy.utils.exceptions import AstropyWarning
import warnings
import louvain
from scipy.stats import spearmanr, pearsonr, ranksums
from scipy.spatial import distance_matrix
from sklearn.metrics import roc_auc_score
import progressbar
import pickle

from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import umap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.colors as colors

from .utils import usot

warnings.simplefilter('ignore', category=AstropyWarning)

def compute_pairwise_scc(X1, X2):
    X1 = X1.argsort(axis=1).argsort(axis=1)
    X2 = X2.argsort(axis=1).argsort(axis=1)
    X1 = (X1-X1.mean(axis=1, keepdims=True))/X1.std(axis=1, keepdims=True)
    X2 = (X2-X2.mean(axis=1, keepdims=True))/X2.std(axis=1, keepdims=True)
    sccmat = np.empty([X1.shape[0], X2.shape[0]], float)
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            c = np.dot( X1[i,:], X2[j,:]) / float(X1.shape[1])
            sccmat[i,j] = c
    return sccmat

class spatial_sc(object):
    """An object for connecting and analysis of spatial data and single-cell transcriptomics data.
    
    A minimal example usage:
    Assume we have (1) a pandas DataFrame for single-cell data ``df_sc`` with rows being cells and columns being genes
    (2) a numpy array for distance matrix among spatial locations ``is_dmat``
    (3) a numpy array for dissimilarity between single-cell data and spatial data ``cost_matrix``
    (4) a numpy array for dissimilarity matrix within single-cell data ``sc_dmat``
    
    >>> import spaotsc
    >>> spsc = spaotsc.SpaOTsc.spatial_sc(sc_data=df_sc, is_dmat=is_dmat, sc_dmat=sc_dmat)
    >>> spsc.transport_plan(cost_matrix)
    >>> spsc.cell_cell_distance(use_landmark=True)
    >>> spsc.clustering()
    >>> spsc.spatial_signaling_ot('Wnt5',['fz'],DSgenes_up=['CycD'],DSgenes_down=['dpp'])
    >>> signal_strengths,_=spsc.infer_signal_range_ml(['Wnt5'],['fz'],['CycD','dpp'], effect_ranges=[10,50,100])
    >>> intercellular_grn=spsc.spatial_grn_range(['Wnt5','fz','CycD','dpp'])

    :param sc_data: single-cell data of size (n_cells, n_genes)
    :type sc_data: class:`pandas.DataFrame`
    :param is_data: spatial data of size (n_locations, n_genes)
    :type is_data: class:`pandas.DataFrame`
    :param sc_data_bin: binarized single-cell data
    :type sc_data_bin: class:`pandas.DataFrame`
    :param is_data_bin: binarized spatial data
    :type is_data_bin: class:`pandas.DataFrame`
    :param is_pos: coordinates of spatial locations (n_locations, n_dimensions)
    :type is_pos: class:`numpy.ndarray`
    :param is_dmat: distance matrix for spatial locations (n_locations, n_locations)
    :type is_dmat: class:`numpy.ndarray`
    :param sc_dmat: dissimilarity matrix for single-cell data (n_cells, n_cells)
    :type sc_dmat: class:`numpy.ndarray`

    List of instance attributes:

    :ivar sc_data: single-cell data of size (n_cells, n_genes) ``__init__``
    :vartype sc_data: class:`pandas.DataFrame`
    :ivar is_data: spatial data of size (n_locations, n_genes) ``__init__``
    :vartype is_data: class:`pandas.DataFrame`
    :ivar sc_data_bin: binarized single-cell data ``__init__``
    :vartype sc_data_bin: class:`pandas.DataFrame`
    :ivar is_data_bin: binarized spatial data ``__init__``
    :vartype is_data_bin: class:`pandas.DataFrame`
    :ivar is_pos: coordinates of spatial locations (n_locations, n_dimensions) ``__init__``
    :vartype is_pos: class:`numpy.ndarray`
    :ivar is_dmat: distance matrix for spatial locations (n_locations, n_locations) ``__init__``
    :vartype is_dmat: class:`numpy.ndarray`
    :ivar sc_dmat: dissimilarity matrix for single-cell data (n_cells, n_cells) ``__init__``
    :vartype sc_dmat: class:`numpy.ndarray`
    :ivar gamma_mapping: the mapping matrix between single-cell data and spatial data (n_cells, n_locations) ``transport_plan``
    :vartype gamma_mapping: class:`numpy.ndarray`
    :ivar sc_dmat_spatial: the spatial cell-cell distance for single-cell data (n_cells, n_cells) ``cell_cell_distance``
    :vartype sc_dmat_spatial: class:`numpy.ndarray`
    :ivar clustering_ncluster_org: number of clusters in original clustering of single-cell data ``clustering``
    :vartype clustering_ncluster_org: int
    :ivar clustering_nsubcluster: number of cell spatial subclusters within each original cluster ``clustering``
    :vartype clustering_nsubcluster: list of int
    :ivar clustering_partition_org: the cell indices for each original cluster ``clustering``
    :vartype clustering_partition_org: list of numpy integer arrays
    :ivar clustering_partition_inds: the cell indices for the cell spatial subclusters, e.g. the key (0,1) returns the cell indices for the second spatial subcluster within the first original cell cluster. ``clustering``
    :vartype clustering_partition_inds: dictionary 
    :ivar gene_cor_scc: the intracellular spearmanr correlation between genes ``nonspatial_correlation``
    :vartype gene_cor_scc: class:`pandas.DataFrame`
    :ivar gene_cor_is: the intercellular spatial correlation between genes ``spatial_correlation``
    :vartype gene_cor_is: class:`pandas.DataFrame`
    :ivar g_bin_edges: the bin edges for the discretization of gene expressions with gene name string as dictionary key ``discretize_expression``
    :vartype g_bin_edges: dictionary

    
    """
    def __init__(self, sc_data = None, is_data = None,
                 sc_data_bin = None, is_data_bin = None,
                 is_pos = None, is_dmat = None, sc_dmat = None):
        """
        Inputs
        ------
        sc_data : (nc,ng_sc) pandas DataFrame
        is_data : (ns,ng_is) pandas DataFrame
        """
        self.sc_data = sc_data
        self.is_data = is_data
        if not sc_data is None:
            self.sc_ncell = self.sc_data.shape[0]
        self.sc_data_bin = sc_data_bin
        self.is_data_bin = is_data_bin
        self.is_pos = is_pos
        self.is_dmat = is_dmat
        self.sc_dmat = sc_dmat
        if not sc_data is None:
            self.sc_genes = list(sc_data.columns.values)
        if not is_data is None:
            self.is_genes = list(is_data.columns.values)
        elif not is_data_bin is None:
            self.is_genes = list(is_data_bin.columns.values)
        self.g_bin_edges = {}
    
    def transport_plan(self,
        cost_matrix,
        cor_matrix = None,
        alpha = 0.1,
        epsilon = 1.0,
        rho = 100.0,
        G_sc = None,
        G_is = None,
        scaling = False):

        """Mapping between single cells and spatial data as transport plan.
        
        Generates: `self.gamma_mapping`: (n_cells, n_locations) `numpy.ndarray`

        :param cost_matrix: dissimilarity matrix between single-cell data and spatial data (cells, locations)
        :type cost_matrix: class:`numpy.ndarray`
        :param cor_matrix: similarity matrix between single-cell data and spatial data (cells, locations)
        :type cor_matrix: class:`numpy.ndarray`, optional
        :param alpha: weight for structured part (Gromov-Wassertein loss term)
        :type alpha: float, [0,1], defaults to 0.1
        :param epsilon: weight for entropy regularization term
        :type epsilon: float, defaults to 1.0
        :param rho: weight for KL divergence penalizing unbalanced transport
        :type rho: float, defaults to 100.0
        :param G_sc: dissimilarity matrix within single-cell data (cells, cells)
        :type G_sc: class:`numpy.ndarray`
        :param G_is: distance matrix within spatial data (locations, locations)
        :type G_is: class:`numpy.ndarray`
        :param scaling: whether scale the cost_matrix to have max=1
        :type scaling: boolean, defaults to False
        :return: a mapping between single-cell data and spatial data (cells, locations)
        :rtype: class:`numpy.ndarray`
        """

        if not cor_matrix is None:
            weight_matrix = np.exp(cor_matrix)
        else:
            weight_matrix = np.exp(1-cost_matrix)
        w_a = np.sum(weight_matrix, axis=1)
        w_b = np.sum(weight_matrix, axis=0)
        # w_a = np.ones(cost_matrix.shape[0])
        # w_b = np.ones(cost_matrix.shape[1])
        w_a = w_a/np.sum(w_a); w_b = w_b/np.sum(w_b)
        if alpha > 0.0:
            G_sc = self.sc_dmat/np.max(self.sc_dmat)
            G_is = self.is_dmat/np.max(self.is_dmat)
        if scaling:
            cost_matrix = cost_matrix/np.max(cost_matrix)
        if alpha == 0.0 and np.isinf(rho):
            gamma = ot.sinkhorn(w_a, w_b, cost_matrix, epsilon)
        elif alpha == 0.0 and not np.isinf(rho):
            gamma = usot.uot(w_a, w_b, cost_matrix, epsilon, rho = rho)
        else:
            gamma = usot.usot(w_a, w_b, cost_matrix, G_sc, G_is, alpha,
                              epsilon = epsilon, rho = rho)
        self.gamma_mapping = gamma
        return gamma

    def cell_cell_distance(self,
        epsilon = 0.01,
        rho = np.Inf,
        scaling = True,
        sc_dmat_spatial = None,
        use_landmark = False,
        n_landmark = 100):

        """Compute spatial distance between single cells using optimal transport.
        
        Generates: `self.sc_dmat_spatial`: (n_cell, n_cell) `numpy.ndarray`

        Requires: `self.gamma_mapping`, `self.is_dmat`

        :param epsilon: weight for entropy regularization term
        :type epsilon: float, defaults to 0.01
        :param rho: weight for KL divergence penalizing unbalanced transport
        :type rho: float, defaults to inf
        :param scaling: whether to scale the cost_matrix (is_dmat) to avoid numerical overflow
        :type scaling: boolean, defaults to True
        :param sc_dmat_spatial: the spatial distance matrix for single cells (n_cells, n_cells). If given, simply set the distance matrix without computing.
        :type sc_dmat_spatial: class:`numpy.ndarray`, optional
        :param use_landmark: whether to use landmark points for computing transport distance.
        :type use_landmark: boolean, defaults to False
        :param n_landmark: number of landmark points to use if use_landmark
        :type n_landmark: int, defaults to 100
        :return: (spatial) cell-cell distance matrix (n_cells, n_cells)
        :rtype: class:`numpy.ndarray`
        """

        if sc_dmat_spatial is None:
            sc_dmat_spatial = np.zeros([self.sc_ncell, self.sc_ncell], float)
            gamma_org = self.gamma_mapping
            is_dmat_org = self.is_dmat
            if use_landmark:
                ind_select, asmat = choose_landmarks(self.is_pos, n_landmark, dmat = is_dmat_org)
                gamma = gamma_org.dot(asmat.T)
                is_dmat = is_dmat_org[ind_select,:][:,ind_select]
            else:
                is_dmat = is_dmat_org
                gamma = gamma_org
            for i in range(gamma.shape[0]):
                gamma[i,:] = gamma[i,:]/np.sum(gamma[i,:])
            if scaling:
                is_dmat_max = is_dmat.max()
                is_dmat = is_dmat/np.max(is_dmat)
            print("computing cell cell distance via optimal transport")
            bar = progressbar.ProgressBar(maxval=(self.sc_ncell-1)*self.sc_ncell/2)
            bar.start(); bar_cnt = 0
            for i in range(self.sc_ncell-1):
                print(i)
                for j in range(i+1, self.sc_ncell):
                    if np.isinf(rho):
                        g = ot.sinkhorn(gamma[i,:], gamma[j,:], is_dmat, epsilon)
                    else:
                        g = usot.uot(gamma[i,:], gamma[j,:], is_dmat, epsilon = epsilon, rho = rho)
                    d = np.sum(g*is_dmat)
                    # print(i,j,d)
                    sc_dmat_spatial[i,j] = d; sc_dmat_spatial[j,i] = d
                    bar_cnt += 1; bar.update(bar_cnt)
                    # if bar_cnt%100 == 0:
                    #     print(float(bar_cnt)/((self.sc_ncell-1)*self.sc_ncell*0.5))
            bar.finish()
            if scaling:
                sc_dmat_spatial = sc_dmat_spatial * is_dmat_max
        self.sc_dmat_spatial = sc_dmat_spatial
        return sc_dmat_spatial

    def gene_gene_distance(self,
        genes = None,
        epsilon = 0.01,
        rho = np.Inf,
        scaling = True,
        sc_dmat_spatial = None,
        use_landmark = False,
        n_landmark = 100):

        """Compute Wasserstein distance between gene expressions in scRNA-seq data.
        
        :param genes: the gene names to compute distance
        :type genes: list of str
        :param epsilon: weight for entropy regularization term
        :type epsilon: float, defaults to 0.01
        :param rho: weight for KL divergence penalizing unbalanced transport
        :type rho: float, defaults to inf
        :param scaling: whether to scale the cost matrix
        :type scaling: boolean, defaults to True
        :param sc_dmat_spatial: spatial distance matrix over the single cells
        :type sc_dmat_spatial: class:`numpy.ndarray`
        :param use_landmark: whether to use landmark points to accelarate computation
        :type use_landmark: boolean, defaults to False
        :param n_landmark: number of landmark genes to use
        :type n_landmark: int, defaults to 100
        :return: gene-gene distance matrix
        :rtype: class:`numpy.ndarray`
        """

        if sc_dmat_spatial is None:
            sc_dmat_spatial = self.sc_dmat_spatial
        if genes is None:
            print("Please provide a list of genes for computation.")
        X_org = np.array( self.sc_data[genes] )
        dmat_org = sc_dmat_spatial
        D = np.zeros([len(genes), len(genes)], float)
        if use_landmark:
            ind_select, asmat = choose_landmarks(None, n_landmark, dmat=sc_dmat_spatial)
            X = (X_org.T).dot(asmat.T)
            dmat = dmat_org[ind_select,:][:,ind_select]
        else:
            X = X_org.T
            dmat = dmat_org
        if scaling:
            dmat = dmat/np.max(dmat)
        for i in range(X.shape[0]):
            X[i,:] = X[i,:]/np.sum(X[i,:])
        nz_inds = []
        for i in range(len(genes)):
            nz_ind = np.where(X[i,:] > 0)[0]
            nz_inds.append(nz_ind)
        bar = progressbar.ProgressBar(maxval=(len(genes)-1)*len(genes)/2)
        bar.start(); bar_cnt=0
        for i in range(len(genes)-1):
            nz_ind_i = nz_inds[i]
            for j in range(i+1, len(genes)):
                nz_ind_j = nz_inds[j]
                if np.isinf(rho):
                    g = ot.sinkhorn(X[i,nz_ind_i], X[j,nz_ind_j], dmat[nz_ind_i,:][:,nz_ind_j], epsilon)
                else:
                    g = usot.uot(X[i,nz_ind_i], X[j,nz_ind_j], dmat[nz_ind_i,:][:,nz_ind_j], epsilon=epsilon, rho=rho)
                d = np.sum(g*dmat[nz_ind_i,:][:,nz_ind_j])
                D[i,j] = d; D[j,i] = d
                # print(i,j,d)
                bar_cnt += 1; bar.update(bar_cnt)
        bar.finish()
        return D

    def gene_clustering(self, gene_dmat, res=3, k=5, rng_seed=48823):
        """
        Cluster the genes based on their spatial pattern difference.

        :param gene_dmat: the distance matrix  for genes (n_gene, n_gene)
        :type gene_dmat: class:`numpy.ndarray`
        :param res: resolution parameter used by louvain clustering, higher res gives more clusters
        :type res: float, defaults to 3
        :param k: the k for knn graph fed to louvain algorithm
        :type k: int, defaults to 5
        :param rng_seed: random seed for louvain algorithm to get consistent results
        :type rng_seed: int
        :return: a list of index vectors for the clusters
        :rtype: list of list of int
        """
        G = knn_graph(gene_dmat, k)
        louvain.set_rng_seed(rng_seed)
        partition = louvain.find_partition(G, \
                            louvain.RBConfigurationVertexPartition, \
                            resolution_parameter=res, \
                            weights=None)
        return partition

    def clustering(self, genes=None, pca_n_components=None, res_sc=0.5, res_is=0.3, min_n = 3):
        """Clustering and spatial subclustering.
        
        Generates:
        
        `self.clustering_nsubcluster`: list of int, numbers of subclusters in each cluster obtained in regular clustering of single-cell data

        `self.clustering_partition_inds`: list of cell index arrays for clusters

        `self.clustering_partition_org`: a dictionary for cell index arrays of spatial subclusters. The key (1,0) gives the first subcluster for the second cluster.

        Requires:

        `self.sc_dmat_spatial`, `self.sc_data`

        :param genes: genes to use when clustering single-cell data. All genes in self.sc_data are used if not specified.
        :type genes: list
        :param pca_n_components: number of pca components when clustering single-cell data
        :type pca_n_components: int
        :param res_sc: resolution parameter in louvain clustering for single-cell data
        :type res_sc: float, defaults to 0.5
        :param res_is: resolution parameter in louvain clustering for spatial subclustering of single-cel data
        :type res_is: float, defaults to 0.3
        :param min_n: minimum number of members to be considered a cluster
        :type min_n: int, defaults to 3
        """

        # need to set random seed for both numpy and louvain to be consistent
        np.random.seed(92614)
        louvain.set_rng_seed(48823)
        if genes is None:
            genes = self.sc_genes
        X = np.array( self.sc_data[genes], float )
        if not pca_n_components is None:
            X_pca = PCA(n_components=pca_n_components).fit_transform(X)
            X = X_pca
        dmat_sc = distance_matrix(X_pca, X_pca)
        dmat_is = self.sc_dmat_spatial
        G_sc = knn_graph(dmat_sc, 10) # 50 for drosophila and 10 for zebrafish?
        G_is = knn_graph(dmat_is, 50)
        weights = np.array(G_sc.es["weight"]).astype(np.float64)
        partition_sc = louvain.find_partition(G_sc, \
                       louvain.RBConfigurationVertexPartition, \
                       resolution_parameter=res_sc, \
                       weights=None)
        ncluster_sc = len(partition_sc)
        sub_partitions = []
        partition_inds = {}
        subcluster_gene_rankings = {}
        gene_rankings = {}
        nsubcluster = np.empty(ncluster_sc, int)
        for i in range(ncluster_sc):
            cid = partition_sc[i]
            gene_rankings[i] = self.rank_marker_genes(cid)
            G_is_sub_vs = G_is.vs.select(cid)
            G_is_sub = G_is.subgraph(G_is_sub_vs)
            weights = np.array(G_is_sub.es["weight"]).astype(np.float64)
            tmp_partition = louvain.find_partition(G_is_sub, \
                            louvain.RBConfigurationVertexPartition, \
                            resolution_parameter=res_is, \
                            weights=None)
            sub_partitions.append(tmp_partition)
            print(i, len(tmp_partition), [len(tmp_partition[j]) for j in range(len(tmp_partition))] )
            cnt = 0
            for j in range(len(tmp_partition)):
                if len(tmp_partition[j]) > min_n:
                    partition_inds[(i,cnt)] = np.array(partition_sc[i],int)[np.array(tmp_partition[j],int)]
                    subcluster_gene_rankings[(i,cnt)] = self.rank_marker_genes(partition_inds[(i,cnt)])
                    cnt += 1
            nsubcluster[i] = cnt
        for i in range(ncluster_sc):
            print(i, len(partition_sc[i]), self.sc_genes[gene_rankings[i][0]])
            for k in range(5):
                print(self.sc_genes[gene_rankings[i][k]])
            for j in range(nsubcluster[i]):
                print(i,j, len(partition_inds[(i,j)]), self.sc_genes[subcluster_gene_rankings[(i,j)][0]])
                for k in range(5):
                    print(self.sc_genes[subcluster_gene_rankings[(i,j)][k]])
        self.clustering_ncluster_org = ncluster_sc
        self.clustering_nsubcluster = nsubcluster
        self.clustering_partition_inds = partition_inds
        self.clustering_partition_org = partition_sc
        self.clustering_dmat_sc = dmat_sc
        self.clustering_subcluster_gene_rankings = subcluster_gene_rankings
        self.clustering_cluster_gene_rankings = gene_rankings

    def rank_marker_genes(self, cid, genes=None, method='ranksum', return_scores=False):
        """
        Rank genes to identify markers for cell clusters.

        :param cid: cell indices for the cluster
        :type cid: class:`numpy.1darray`
        :param genes: candidate genes to examine. If not specified, all genes are used.
        :type genes: list
        :param method: method to use. 1. 'roc', using auc-roc score to rank; 2. 'ranksum', using ranksum statistics.
        :type method: str, defaults to 'ranksum'
        :param return_scores: whether to return scores instead of sorted gene indices
        :type return_scores: boolean, defaults to False
        :return: sorted gene indices (if return_scores==False) or gene scores (if return_scores==True)
        :rtype: class:`numpy.1darray`
        """
        if genes is None:
            genes = self.sc_genes
        X = np.array( self.sc_data[genes], float )
        ncell = X.shape[0]
        if method == 'diff':
            w = -np.ones(ncell, float)
            w[cid] = 1.0
            scores = np.dot(X.T, w.reshape(-1,1)).reshape(-1)
        elif method == 'roc':
            y_true = np.zeros(ncell, int)
            y_true[cid] = 1
            scores = []
            for i in range(X.shape[1]):
                auc = roc_auc_score(y_true, X[:,i])
                scores.append(auc)
            scores = np.array(scores, float)
        elif method == 'ranksum':
            ncid = []
            for i in range(ncell):
                if not i in cid:
                    ncid.append(i)
            ncid = np.array( ncid, int )
            scores = []
            for i in range(X.shape[1]):
                scores.append( ranksums(X[cid,i], X[ncid,i])[0] )
            scores = np.array(scores, float)
        sorted_ind = np.argsort(-scores)
        # print(scores[sorted_ind[0]])
        if return_scores:
            return scores
        else:
            return sorted_ind

    def spatial_signaling_scoring(self,
        Lgene,
        Rgene,
        Rbgene = None,
        Tgenes=None,
        DSgenes_up = None,
        DSgenes_down = None,
        effect_range = None,
        kernel = 'exp',
        kernel_nu = 5,
        gene_eta = None,
        penalty_type="addition"):

        """Generate cell-cell signaling using predefined scoring function.

        Requires: `self.sc_dmat_spatial`, `self.sc_data`

        :param Lgene: name of the ligand gene
        :type Lgene: str
        :param Rgene: name list of receptor genes
        :type Rgene: list of str
        :param Rbgene: name list of genes for proteins bound to receptor for the receptor to work 
        :type Rbgene: list of str, optional
        :param Tgenes: name list of genes for transporters of ligands
        :type Tgenes: list of str, optional
        :param DSgenes_up: name list of up regulated genes by the ligand-receptor
        :type DSgenes_up: list of str
        :param DSgenes_down: name list of down regulated genes by the ligand-receptor
        :type DSgenes_down: list of str
        :param effect_range: spatial distance cutoff for the signaling
        :type effect_range: float
        :param kernel: weight kernel to use for soft thresholding
        :type kernel: str, defaults to 'exp'
        :param kernel_nu: power for weight kernel, a higher power gives a shaper edge
        :type kernel_nu: float, defaults to 5
        :param gene_eta: a list of threshold values for the downstream genes
        :type gene_eta: list of float, defaults to 1s
        :param penalty_type: how to penalize inconsistency of downstream genes. 'addition': relaxed penalty; 'multiplication': strict penalty
        :type penalty_type: str, defaults to 'addition'
        :return: a scoring matrix for the given signaling genes (cells, cells), (i,j) entry is the score for cell i sending signals to cell j
        :rtype: class:`numpy.ndarray`
        """

        L = np.array( self.sc_data[Lgene], float )
        Rs = np.array( self.sc_data[Rgene], float )
        R = np.mean(Rs, axis=1)
        if not Rbgene is None:
            Rbs = np.array( self.sc_data[Rbgene], float )
            Rb = np.mean(Rbs, axis=1)
            R = R * Rb
        if not Tgenes is None:
            Ts = np.array( self.sc_data[Tgenes], float )
            T = np.mean(Ts, axis=1)
            L = L * T
        if not DSgenes_up is None:
            Du = np.array( self.sc_data[DSgenes_up], float )
            nu = len(DSgenes_up)
            Du = Du.reshape(-1,nu)
        else:
            nu = 0
        if not DSgenes_down is None:
            Dd = np.array( self.sc_data[DSgenes_down], float )
            nd = len(DSgenes_down)
            Dd = Dd.reshape(-1,nd)
        else:
            nd = 0
        if nu > 0 and nd > 0:
            DS = np.concatenate((Du, Dd), axis=1)
        elif nu == 0 and nd > 0:
            DS = Dd
        elif nu > 0 and nd == 0:
            DS = Du

        P = np.zeros([nu+nd], float)
        P[:nu] = -1.0; P[nu:nu+nd] = 1.0
        alpha = phi_exp(np.abs(np.outer(L,R)), 1.0, kernel_nu, -1.0)
        betas = np.empty_like(DS)
        for i in range(betas.shape[1]):
            betas[:,i] = phi_exp(DS[:,i], 1.0, kernel_nu, P[i])
        if penalty_type == 'multiplication':
            beta = np.prod(betas, axis=1)
        elif penalty_type == 'addition':
            beta = np.mean(betas, axis=1)
        nzind = np.where(alpha+beta!=0)
        S = np.zeros_like(alpha)
        S[nzind] = (alpha*beta)[nzind]/(alpha+beta)[nzind]

        sc_dmat_spatial = self.sc_dmat_spatial
        W_insitu = phi_exp(sc_dmat_spatial, effect_range, 5, 1)
        S_insitu = S * W_insitu
        self.signal_P = S
        self.signal_P_spatial = S_insitu
        self.signal_W_spatial = W_insitu
        return S_insitu

    def spatial_signaling_ot(self,
        Lgene,
        Rgene,
        Tgenes = None,
        Rbgene = None,
        DSgenes_up = None,
        DSgenes_down = None,
        effect_range = None,
        rho = 10.0,
        epsilon = 0.2):

        """Generate cell-cell signaling using optimal transport.

        Requires: `self.sc_dmat_spatial`, `self.sc_data`

        :param Lgene: name of the ligand gene
        :type Lgene: str
        :param Rgene: name list of receptor genes
        :type Rgene: list of str
        :param Tgenes: name list of genes for transporters of ligands
        :type Tgenes: list of str, optional
        :param Rbgene: name list of genes for proteins bound to receptor for the receptor to work 
        :type Rbgene: list of str, optional
        :param DSgenes_up: name list of up regulated genes by the ligand-receptor
        :type DSgenes_up: list of str
        :param DSgenes_down: name list of down regulated genes by the ligand-receptor
        :type DSgenes_down: list of str
        :param effect_range: spatial distance cutoff for the signaling
        :type effect_range: float
        :param epsilon: weight for entropy regularization term
        :type epsilon: float, defaults to 0.2
        :param rho: weight for KL divergence penalizing unbalanced transport
        :type rho: float, defaults to inf
        :return: a scoring matrix for the given signaling genes (cells, cells), (i,j) entry is the score for cell i sending signals to cell j
        :rtype: class:`numpy.ndarray`
        """

        kernel_nu = 5
        L = np.array( self.sc_data[Lgene], float )
        Rs = np.array( self.sc_data[Rgene], float )
        R = np.mean( Rs, axis=1 )
        if not Rbgene is None:
            Rb = np.array( self.sc_data[Rbgene], float )
            R = R * Rb
        if not DSgenes_up is None:
            Du = np.array( self.sc_data[DSgenes_up], float )
            nu = len(DSgenes_up)
            Du = Du.reshape(-1,nu)
        else:
            nu = 0
        if not DSgenes_down is None:
            Dd = np.array( self.sc_data[DSgenes_down], float )
            nd = len(DSgenes_down)
            Dd = Dd.reshape(-1,nd)
        else:
            nd = 0
        if nu > 0 and nd > 0:
            DS = np.concatenate((Du, Dd), axis=1)
        elif nu == 0 and nd > 0:
            DS = Dd
        elif nu > 0 and nd == 0:
            DS = Du
        
        P = np.zeros([nu+nd], float)
        P[:nu] = -1.0; P[nu:nu+nd] = 1.0
        betas = np.empty_like(DS)
        for i in range(betas.shape[1]):
            betas[:,i] = phi_exp(DS[:,i], 1.0, kernel_nu, P[i])
        beta = np.mean(betas, axis=1)
        beta = beta * R
        
        w_a = L/np.sum(L)
        w_b = beta/np.sum(beta)
        penalty_scale = 10
        M = self.sc_dmat_spatial
        if not effect_range is None:
            M[np.where(M > effect_range)] *= penalty_scale
            M = M/effect_range
        else:
            M = M/M.max()
        S = np.zeros([len(L), len(L)], float)
        nzind_L = np.where(L > 0)[0]
        print(len(nzind_L))
        nzind_beta = np.where(beta > 0)[0]
        print(len(nzind_beta))
        gamma = ot.sinkhorn(w_a[nzind_L], w_b[nzind_beta], M[nzind_L,:][:,nzind_beta], epsilon)
        # gamma = usot.uot(w_a[nzind_L], w_b[nzind_beta], M[nzind_L,:][:,nzind_beta], epsilon, rho=rho, niter=100)
        for i in range(len(nzind_L)):
            for j in range(len(nzind_beta)):
                S[nzind_L[i],nzind_beta[j]] = gamma[i,j]
        S = S/S.max()
        return S


    def spatial_signaling_ot_new(self,
        Lgenes,
        Rgenes,
        Tgenes = [],
        Rbgenes = [],
        DSgenes_up = [],
        DSgenes_down = [],
        gene_bandwidth = {},
        effect_range = None,
        rho = 10.0,
        epsilon = 0.2,
        kernel_nu = 5,
        use_kernel_ligand = False,
        use_kernel_receptor = False,
        return_weight_only = False):

        nc = self.sc_data.shape[0]
        # Ligand genes
        Ls = np.array( self.sc_data[Lgenes], float ).reshape(-1,len(Lgenes))
        if not use_kernel_ligand:
            Lscore = np.mean( Ls, axis=1 )
        else:
            Lscores = np.empty_like(Ls)
            for i in range(len(Lgenes)):
                gene = Lgenes[i]
                if gene in gene_bandwidth.keys():
                    eta = gene_bandwidth[gene]
                else:
                    eta = 1.0
                Lscores[:,i] = phi_exp(Ls[:,i], eta, kernel_nu, -1)
            Lscore = np.mean( Lscores, axis=1 )
        # Receptor genes
        Rs = np.array( self.sc_data[Rgenes], float ).reshape(-1,len(Rgenes))
        if not use_kernel_receptor:
            Rscore = np.mean( Rs, axis=1 )
        else:
            Rscores = np.empty_like(Rs)
            for i in range(len(Rgenes)):
                gene = Rgenes[i]
                if gene in gene_bandwidth.keys():
                    eta = gene_bandwidth[gene]
                else:
                    eta = 1.0
                Rscores[:,i] = phi_exp(Rs[:,i], eta, kernel_nu, -1)
            Rscore = np.mean( Rscores, axis=1 )
        # Receptor bound genes
        if len(Rbgenes) > 0:
            Rbs = np.array( self.sc_data[Rbgenes], float ).reshape(-1,len(Rbgenes))
            Rbscores = np.empty_like(Rbs)
            for i in range(len(Rbgenes)):
                gene = Rbgenes[i]
                if gene in gene_bandwidth.keys():
                    eta = gene_bandwidth[gene]
                else:
                    eta = 1.0
                Rbscores[:,i] = phi_exp(Rbs[:,i], eta, kernel_nu, -1)
            Rbscore = np.mean( Rbscores, axis=1 )
        else:
            Rbscore = np.ones( [nc], float )
        # Ligand transporter genes
        if len(Tgenes) > 0:
            Ts = np.array( self.sc_data[Tgenes], float ).reshape(-1,len(Tgenes))
            Tscores = np.empty_like(Ts)
            for i in range(len(Tgenes)):
                gene = Tgenes[i]
                if gene in gene_bandwidth.keys():
                    eta = gene_bandwidth[gene]
                else:
                    eta = 1.0
                Tscores[:,i] = phi_exp(Ts[:,i], eta, kernel_nu, -1)
            Tscore = np.mean( Tscores, axis=1 )
        else:
            Tscore = np.ones( [nc], float )
        # Downstream genes
        if len(DSgenes_up) > 0:
            Dups = np.array( self.sc_data[DSgenes_up], float ).reshape(-1,len(DSgenes_up))
            Dupscores = np.empty_like(Dups)
            for i in range(len(DSgenes_up)):
                gene = DSgenes_up[i]
                if gene in gene_bandwidth.keys():
                    eta = gene_bandwidth[gene]
                else:
                    eta = 1.0
                Dupscores[:,i] = phi_exp(Dups[:,i], eta, kernel_nu, -1)
        if len(DSgenes_down) > 0:
            Ddowns = np.array( self.sc_data[DSgenes_down], float ).reshape(-1,len(DSgenes_down))
            Ddownscores = np.empty_like(Ddowns)
            for i in range(len(DSgenes_down)):
                gene = DSgenes_down[i]
                if gene in gene_bandwidth.keys():
                    eta = gene_bandwidth[gene]
                else:
                    eta = 1.0
                Ddownscores[:,i] = phi_exp(Ddowns[:,i], eta, kernel_nu, 1)
        if len(DSgenes_up) > 0 and len(DSgenes_down) > 0:
            DSscore = np.mean( np.concatenate((Dupscores, Ddownscores),axis=1), axis=1 )
        elif len(DSgenes_up) > 0:
            DSscore = np.mean( Dupscores, axis=1 )
        elif len(DSgenes_down) > 0:
            DSscore = np.mean (Ddownscores, axis=1 )
        else:
            DSscore = np.ones( [nc], float )
        # Do the transport problem
        # setup source and target distribution
        w_s = Lscore * Tscore
        w_s_sum = np.sum(w_s)
        w_s = w_s/np.sum(w_s)
        w_t = Rscore * Rbscore * DSscore
        w_t_sum = np.sum(w_t)
        w_t = w_t/np.sum(w_t)
        if return_weight_only:
            return w_s_sum, w_t_sum
        # setup the transport cost
        penalty_scale = 10
        M = self.sc_dmat_spatial
        if not effect_range is None:
            M[np.where(M > effect_range)] *= penalty_scale
            M = M/effect_range
        else:
            M = M/M.max()
        # obtain the transport plan
        S = np.zeros([len(w_s), len(w_t)], float)
        nzind_s = np.where(w_s > 0)[0]
        nzind_t = np.where(w_t > 0)[0]
        gamma = ot.sinkhorn(w_s[nzind_s], w_t[nzind_t], M[nzind_s,:][:,nzind_t], epsilon)
        # gamma = usot.uot(w_a[nzind_L], w_b[nzind_beta], M[nzind_L,:][:,nzind_beta], epsilon, rho=rho, niter=100)
        for i in range(len(nzind_s)):
            for j in range(len(nzind_t)):
                S[nzind_s[i],nzind_t[j]] = gamma[i,j]
        S = S/S.max()

        return S


    def nonspatial_correlation(self, genes=None):
        """
        Compute gene-gene correlation matrix for pre-screening of genes.

        Generates: `self.gene_cor_scc`

        Requires: self.sc_data`, `self.sc_genes`

        :param genes: list of gene names. If not specified, all genes in self.sc_data are used.
        :type genes: list of str
        """
        if genes is None:
            genes = self.sc_genes
        X = np.array( self.sc_data[genes], float )
        ngenes = X.shape[1]
        cor_mat = np.empty([ngenes, ngenes], float)
        # bar = progressbar.ProgressBar(maxval=(ngenes+1)*ngenes/2)
        # print("computing nonspatial correlation between genes")
        # bar.start(); bar_cnt = 0
        # for i in range(ngenes):
        #     for j in range(i, ngenes):
        #         c,p = spearmanr(X[:,i], X[:,j])
        #         cor_mat[i,j] = c; cor_mat[j,i] = c
        #         bar_cnt += 1
        #         bar.update(bar_cnt)
        # bar.finish()
        cor_mat = compute_pairwise_scc(X.T,X.T)
        print(cor_mat)
        df = pd.DataFrame(data=cor_mat, columns=genes, index=genes)
        self.gene_cor_scc = df

    def spatial_correlation(self, genes = None, effect_range = None, kernel = 'lorentz', kernel_nu = 10):
        """Computes spatial correlation between genes for pre-screening.

        Generates: `self.gene_cor_is` pandas DataFrame

        Requires: `self.sc_data`

        :param genes: list of gene to examine
        :type genes: list of str
        :param effect_range: spatial distance
        :type effect_range: float
        :param kernel: type of kernels for weight matrix
        :type kernel: str, defaults to 'lorentz'
        :param kernel_nu: power for weight kernel
        :type kernel_nu: int, defaults to 10
        """
        sc_dmat_spatial = self.sc_dmat_spatial
        if kernel == 'lorentz':
            W_insitu = 1.0/(1.0+np.power(sc_dmat_spatial/effect_range,kernel_nu))
        if genes is None:
            genes = self.sc_genes
        X = np.array( self.sc_data[genes], float )
        for i in range(W_insitu.shape[0]):
            W_insitu[i,i] = 0.0
        W_insitu = W_insitu/np.sum(W_insitu)
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i]-np.mean(X[:,i])) / np.std(X[:,i])
        ngenes = X.shape[1]
        cor_mat = np.empty([ngenes, ngenes], float)
        bar = progressbar.ProgressBar(maxval=(ngenes+1)*ngenes/2)
        print("computing spatial correlation between genes")
        bar.start(); bar_cnt = 0
        for i in range(ngenes):
            for j in range(i, ngenes):
                c = sci(X[:,i], X[:,j], W_insitu)
                cor_mat[i,j] = c; cor_mat[j,i] = c
                bar_cnt += 1
                bar.update(bar_cnt)
        bar.finish()
        df = pd.DataFrame(data=cor_mat, columns=genes, index=genes)
        self.gene_cor_is = df

    def discretize_expression(self, genes=None, p0=1E-15):
        """Discretize gene expression using Bayesian blocks.

        Generate: `self.g_bin_edges`: a dictionary of block edges with gene names as keys

        Requires: `self.sc_data`

        :param p0: the p0 score in Bayesian blocks. A smaller p0 has lower tolerance of false rate, i.e. resulting in fewer blocks.
        :type p0: float, defaults to 1E-15
        """
        if genes is None:
            genes = self.sc_genes
        X_sc = np.array( self.sc_data[genes], float )
        print("discretizing gene expressions")
        bar = progressbar.ProgressBar(maxval=(X_sc.shape[1]))
        bar_cnt = 0; bar.start()
        for i in range(len(genes)):
            bin_edges = bayesian_blocks(X_sc[:,i], p0=p0)
            self.g_bin_edges[genes[i]] = bin_edges
            bar_cnt += 1; bar.update(bar_cnt)
        bar.finish()

    def gene_pair_ml_effect_range(self,
        gene_1,
        gene_2,
        background_genes=None,
        cor_cut=None,
        n_top_g=None,
        effect_ranges=None,
        method='Importance'):

        """Deriving scores for intercellular gene regulation (how much effect does gene_1 in neiborhood have on gene_2) using random forest.

        Requires: `self.sc_dmat_spatial`, `self.sc_data`, `self.gene_cor_scc`

        :param gene_1: the name of source gene whose expression in the neighborhood will be examined
        :type gene_1: str
        :param gene_2: the name of target gene whose cellular expression will be used
        :type gene_2: str
        :param background_genes: a name list for gene that are correlated to gene_2
        :type background_genes: list of str
        :param cor_cut: the cut_off choosing background genes. used when background_genes is not specified
        :type cor_cut: float
        :param n_top_g: the number of genes with highest correlation to gene_2 to be used as background_genes. used when both background_genes and cor_cut are not specified
        :type n_top_g: int
        :param effect_ranges: list of spatial distances to consider
        :type effect_ranges: list of float
        :param method: 'Importance': interpret the feature importance as regulation strength; 'Prediction': interpret prediction accuracy in cross-validation as regulation strength.
        :type method: str, defaults to 'Importance'
        :return: a (n_distance, 2) array with the first row recording the spatial distances examined and the second row being the effect strength
        :rtype: class:`numpy.ndarray`
        """

        sc_dmat_spatial = self.sc_dmat_spatial
        X = np.array( self.sc_data.values, float )
        X_gene_1 = np.array( self.sc_data[gene_1], float )
        gene_2_id = self.sc_genes.index(gene_2)
        if background_genes is None:
            if cor_cut is None and n_top_g is None:
                print("No background genes specified in function gene_pair_xgboost_effect_range")
            else:
                sc_genes_selected = list( self.gene_cor_scc.columns.values )
                cor_tmp = np.array( self.gene_cor_scc[gene_2], float )
                background_genes = []
                if not cor_cut is None:
                    id_tmp = np.where(np.abs(cor_tmp) >= cor_cut)[0]
                elif not n_top_g is None:
                    id_tmp = np.argsort(-np.abs(cor_tmp))[1:n_top_g+1]
                for i in id_tmp:
                    if self.sc_genes[i] != gene_2:
                        background_genes.append(sc_genes_selected[i])
        # background_genes.append(gene_1)
        ml_x_background = np.array( self.sc_data[background_genes], float )
        ml_y = np.array( self.sc_data[gene_2], float )
        ml_x = np.array( ml_x_background, float )

        if method == 'Prediction':
            kf = KFold(n_splits=5, random_state=92614, shuffle=True)
            pcc = []
            effect_range_old = 0.1
            for effect_range in effect_ranges:
                W_insitu = phi_exp(sc_dmat_spatial, effect_range, 5, 1)
                # W_insitu = phi_exp(sc_dmat_spatial, effect_range, 5, 1) - phi_exp(sc_dmat_spatial, effect_range_old, 5, 1)
                effect_range_old = effect_range
                for i in range(W_insitu.shape[0]):
                    W_insitu[i,i] = 0.0
                    W_insitu[i,:] = W_insitu[i,:]/np.sum(W_insitu[i,:])
                X_gene_1_nb = np.dot(W_insitu, X_gene_1)
                # ml_x = np.concatenate((ml_x, X_gene_1_nb.reshape(-1,1)), axis=1)
                ml_x = np.concatenate((ml_x_background, X_gene_1_nb.reshape(-1,1)), axis=1)
                cs = []
                for irepeat in range(10):
                    prediction = np.empty( [len(X_gene_1)], float )
                    for train, test in kf.split(ml_x):
                        rf = RandomForestRegressor()
                        rf.fit(ml_x[train], ml_y[train])
                        ml_y_pred = rf.predict(ml_x[test])
                        prediction[test] = ml_y_pred
                    c,p = pearsonr(prediction, ml_y)
                    cs.append(c)
                pcc.append(np.mean(cs))
            effect_strength = pcc
        elif method == 'Importance':
            effect_range_old = 0.1
            effect_strength = []
            for effect_range in effect_ranges:
                W_insitu = phi_exp(sc_dmat_spatial, effect_range, 5, 1)
                for i in range(W_insitu.shape[0]):
                    W_insitu[i,i] = 0.0
                    W_insitu[i,:] = W_insitu[i,:]/np.sum(W_insitu[i,:])
                X_gene_1_nb = np.dot(W_insitu, X_gene_1)
                ml_x = np.concatenate((ml_x_background, X_gene_1_nb.reshape(-1,1)), axis=1)
                tmp = []
                for irepeat in range(10):
                    rf = RandomForestRegressor(n_estimators=100)
                    rf.fit(ml_x, ml_y)
                    tmp.append(rf.feature_importances_[-1])
                effect_strength.append(np.mean(tmp))

        effect_ranges = np.array( effect_ranges )
        effect_strength = np.array( effect_strength )
        output = np.concatenate((effect_ranges.reshape(-1,1), effect_strength.reshape(-1,1)), axis=1)
        return output

    def construct_weight_matrix(self, eta, nu, p):
        self.W_insitu = phi_exp(self.sc_dmat_spatial, eta, nu, p)
        for i in range(self.W_insitu.shape[0]):
            self.W_insitu[i,:] = self.W_insitu[i,:]/np.sum(self.W_insitu[i,:])

    def gene_pair_pid_effect_range(self,
        gene_1,
        gene_2,
        background_genes = None,
        cor_cut = None,
        n_top_g = None,
        effect_ranges = None,
        p0 = 1E-15,
        cell_id = None,
        output_individual = False):

        """The unique information provided by G1_nb (within various ranges) to
           G2 considering background genes Gi

        Requires: `self.sc_dmat_spatial`, `self.sc_data`, `self.gene_cor_scc`

        :param gene_1: the name of source gene whose expression in the neighborhood will be examined
        :type gene_1: str
        :param gene_2: the name of target gene whose cellular expression will be used
        :type gene_2: str
        :param background_genes: a name list for gene that are correlated to gene_2
        :type background_genes: list of str
        :param cor_cut: the cut_off choosing background genes. used when background_genes is not specified
        :type cor_cut: float
        :param n_top_g: the number of genes with highest correlation to gene_2 to be used as background_genes. used when both background_genes and cor_cut are not specified
        :type n_top_g: int
        :param effect_ranges: list of spatial distances to consider
        :type effect_ranges: list of float
        :param p0: the p0 score in Bayesian blocks. A smaller p0 has lower tolerance of false rate, i.e. resulting in fewer blocks.
        :type p0: float, defaults to 1E-15
        :param output_individual: where to output the information computed with each background gene
        :type output_individual: boolean, defaults to False
        :return: a (n_distance, 2) array with the first row recording the spatial distances examined and the second row being the effect strength
        :rtype: class:`numpy.ndarray`
        """

        pid_u_vec = []
        sc_dmat_spatial = self.sc_dmat_spatial
        X = np.array( self.sc_data.values, float )
        X_gene_1 = np.array( self.sc_data[gene_1], float )
        if len(X_gene_1.shape)==2:
            X_gene_1 = np.mean(X_gene_1, axis=1)
        gene_2_id = self.sc_genes.index(gene_2)
        if background_genes is None:
            if cor_cut is None and n_top_g is None:
                print("No background genes specified in function gene_pair_pid_effect_range")
            else:
                sc_genes_selected = list( self.gene_cor_scc.columns.values )
                cor_tmp = np.array( self.gene_cor_scc[gene_2], float )
                background_genes = []
                if not cor_cut is None:
                    id_tmp = np.where(np.abs(cor_tmp) >= cor_cut)[0]
                elif not n_top_g is None:
                    id_tmp = np.argsort(-np.abs(cor_tmp))[1:n_top_g+1]
                for i in id_tmp:
                    if self.sc_genes[i] != gene_2:
                        background_genes.append(sc_genes_selected[i])
        # Important!!! whether to include gene_1 in the same cell
        background_genes.append(gene_1)

        bar = progressbar.ProgressBar(maxval=len(effect_ranges))
        # print("computing effective ranges for "+gene_1+" to "+gene_2)
        bar.start(); bar_cnt = 0
        pid_bg_vecs = []
        for effect_range in effect_ranges:
            if not self.W_insitu is None:
                W_insitu = self.W_insitu
            else:
                W_insitu = phi_exp(sc_dmat_spatial, effect_range, 5, 1)
                for i in range(W_insitu.shape[0]):
                    # W_insitu[i,i] = 0.0
                    W_insitu[i,:] = W_insitu[i,:]/np.sum(W_insitu[i,:])
            X_gene_1_nb = np.dot(W_insitu, X_gene_1)
            bin_edges_gene_1_nb = bayesian_blocks(X_gene_1_nb, p0=p0)
            pid_u = 0.0
            # print('nb gene bins', bin_edges_gene_1_nb)
            pid_bg_vec = []
            for bg_gene in background_genes:
                bg_gene_id = self.sc_genes.index(bg_gene)
                X_tmp = np.concatenate((X_gene_1_nb.reshape(-1,1), X[:,np.array([gene_2_id, bg_gene_id],int)]), axis=1)
                if not cell_id is None:
                    X_tmp = X_tmp[cell_id,:]
                joint,_ = np.histogramdd(X_tmp, bins = [bin_edges_gene_1_nb, \
                                                self.g_bin_edges[gene_2], \
                                                self.g_bin_edges[bg_gene]])
                joint = joint/float(X_tmp.shape[0])
                # print(joint.shape, self.g_bin_edges[bg_gene], bg_gene)
                d = dit.Distribution.from_ndarray( joint )
                pid = dit.pid.PID_WB(d, output=[1])
                u = pid.get_partial( ((0,),) )
                pid_u += u
                pid_bg_vec.append(u)
            pid_u_vec.append(pid_u)
            pid_bg_vec = np.array( pid_bg_vec, float )
            pid_bg_vecs.append( pid_bg_vec )
            bar_cnt += 1; bar.update()
        bar.finish()
        effect_ranges = np.array( effect_ranges )
        pid_u_vec = np.array( pid_u_vec )
        output = np.concatenate((effect_ranges.reshape(-1,1), pid_u_vec.reshape(-1,1)), axis=1)
        if output_individual:
            return output, pid_bg_vecs
        else:
            return output


    def infer_signal_range_ml(self,
        Lgenes,
        Rgenes,
        Dgenes,
        n_top_g=50,
        effect_ranges=None,
        method='Importance',
        custom_dmat=None):

        """Determine spatial distance for given signaling using random forest.

        Requires: `self.sc_dmat_spatial`, `self.sc_data`, `self.gene_cor_scc`

        :param Lgenes: name list of ligand genes
        :type Lgenes: list of str
        :param Rgenes: name list of receptor genes
        :type Rgenes: list of str
        :param Dgenes: name list of downstream genes
        :type Dgenes: list of str
        :param n_top_g: number of background genes to use when building predictive model.
        :type n_top_g: int, defaults to 50
        :param effect_ranges: the spatial distances to examine
        :type effect_ranges: list of float
        :param method: the way of interpreting likelihood for each spatial distance
        :type method: str, defaults to 'Importance'
        :param custom_dmat: a cell-cell distance matrix given by user. `self.sc_dmat_spatial` is used if not given.
        :type custom_dmat: class:`numpy.ndarray`
        :return: (n_distance, 2) array for spatial distances (first row) and effect strengths (second row); and a (n_distance, n_DSgenes) array for the effect strength of each downstream genes.
        :rtype: two class:`numpy.ndarray`
        """

        L = np.array( self.sc_data[Lgenes], float ).reshape(-1,len(Lgenes))
        R = np.array( self.sc_data[Rgenes], float ).reshape(-1,len(Rgenes))
        D = np.array( self.sc_data[Dgenes], float ).reshape(-1,len(Dgenes))
        if custom_dmat is None:
            sc_dmat_spatial = self.sc_dmat_spatial
        else:
            sc_dmat_spatial = custom_dmat
        R_strength = np.mean(phi_exp(R, 1.0, 5, -1), axis=1)
        R_cut = 0.0
        cid = np.where(R_strength >= R_cut)[0]
        effect_strength = np.empty([len(effect_ranges), len(Dgenes)], float)
        effect_strength_scaled = np.empty([len(effect_ranges), len(Dgenes)], float)
        for i_d in range(len(Dgenes)):
            ml_y = D[cid, i_d]
            sc_genes_selected = list( self.gene_cor_scc.columns.values )
            cor_tmp = np.array( self.gene_cor_scc[Dgenes[i_d]], float )
            background_genes = []
            id_tmp = np.argsort(-np.abs(cor_tmp))[1:n_top_g+1]
            for i in id_tmp:
                if self.sc_genes[i] != Dgenes[i_d]:
                    background_genes.append(sc_genes_selected[i])
            # background_genes.extend(Lgenes)
            ml_x_background = np.array( self.sc_data[background_genes], float )[cid,:]
            ml_x_all = np.array( ml_x_background )
            r_old = 0.1

            for i_r in range(len(effect_ranges)):
                print(i_r)
                r = effect_ranges[i_r]
                W = phi_exp(sc_dmat_spatial[cid,:], r, 20, 1)
                # W = phi_exp(sc_dmat_spatial[cid,:], r, 5, 1) - phi_exp(sc_dmat_spatial[cid,:], r_old, 5, 1)
                r_old = r
                for i in range(W.shape[0]):
                    W[i,cid[i]] = 0.0
                    W[i,np.where(W[i,:]<0.1)[0]] = 0.0
                    W[i,np.where(W[i,:]>0.9)[0]] = 1.0
                    if np.sum(W[i,:]) > 0.0:
                        W[i,:] = W[i,:]/np.sum(W[i,:])
                L_nb = W.dot(L)
                ml_x = np.concatenate((ml_x_background, L_nb), axis=1)
                # ml_x = np.concatenate((ml_x, L_nb), axis=1)
                ml_x_all = np.concatenate((ml_x_all, L_nb), axis=1)
                if method == 'Prediction':
                    kf = KFold(n_splits=10, random_state=92614, shuffle=True)
                    tmp_strength = []
                    for i in range(10):
                        prediction = np.empty([len(cid)], float)
                        for train, test in kf.split(ml_x):
                            rf = RandomForestRegressor(n_estimators=10, n_jobs=10)
                            rf.fit(ml_x[train], ml_y[train])
                            pred = rf.predict(ml_x[test])
                            prediction[test] = pred[:]
                        c,p = pearsonr(prediction, ml_y)
                        tmp_strength.append(c)
                    effect_strength[i_r, i_d] = np.mean(tmp_strength)
                elif method == 'Importance':
                    tmp_strength = []
                    for i in range(10):
                        rf = RandomForestRegressor(n_estimators=100, n_jobs=10)
                        # rf = GradientBoostingRegressor(n_estimators=100)
                        rf.fit(ml_x, ml_y, sample_weight=R_strength)
                        tmp_strength.append(np.mean(rf.feature_importances_[-len(Lgenes):]))
                    effect_strength[i_r, i_d] = np.mean(tmp_strength)
            if method == 'ImportanceAll':
                tmp_strength = np.empty([len(effect_ranges), 0], float)
                for i in range(10):
                    rf = RandomForestRegressor(n_estimators=500)
                    rf.fit(ml_x_all, ml_y)
                    tmp_strength = np.concatenate((tmp_strength, rf.feature_importances_[-len(effect_ranges):].reshape(-1,1)), axis=1)
                effect_strength[:,i_d] = np.mean(tmp_strength, axis=1)
            effect_strength_scaled[:,i_d] = (effect_strength[:,i_d] - np.min(effect_strength[:,i_d]))/(np.max(effect_strength[:,i_d]) - np.min(effect_strength[:,i_d]))
        e = np.mean(effect_strength_scaled, axis=1)
        output = np.concatenate((effect_ranges.reshape(-1,1), e.reshape(-1,1)), axis=1)
        return output, effect_strength

    def spatial_grn_range(self,
        genes,
        effect_range = None,
        cor_cut = None,
        n_top_edge = None,
        cor_cut_bg = None,
        n_top_g_bg = None,
        method = 'pid',
        p0 = 1E-15,
        output_individual = False):

        """Generate the spatial map for intercellular gene-gene regulatory information flow.

        Requires: `self.sc_data`, `self.sc_dmat_spatial`, `self.gene_cor_scc`, `self.gene_cor_is`

        :param genes: name list of genes to be examined
        :type genes: list of str
        :param effect_range: spatial distance for analyzing the intercellular processes
        :type effect_range: float
        :param cor_cut: the cutoff for spatial correlation between two genes for further examination (used if n_top_edge not specified)
        :type cor_cut: float
        :param n_top_edge: the number of gene pairs to examine with highest spatial correlation 
        :type n_top_edge: int
        :param cor_cut_bg: the cutoff for intracellular gene correlation to select background genes
        :type cor_cut_bg: float
        :param n_top_g_bg: the number of genes with highest intracellular gene correlation with the target gene to use as background genes (used if cor_cut_bg not specified)
        :type n_top_g_bg: int
        :param p0: the p0 value for Bayesian blocks (lower p0 gives fewer number of bins)
        :type p0: float, defaults to 1E-15
        :param output_individual: whether to output the individual values computed with each background gene
        :type output_individual: boolean, defaults to False
        :return: a data frame with rows being source genes and columns being target genes
        :rtype: class:`pandas.DataFrame`
        """

        if effect_range is None:
            print("effect range not specified in spatial_grn_range")
        X = np.array( self.sc_data[genes], float )
        sc_dmat_spatial = self.sc_dmat_spatial
        W_insitu = phi_exp(sc_dmat_spatial, effect_range, 5, 1)
        for i in range(W_insitu.shape[0]):
            W_insitu[i,i] = 0.0
            W_insitu[i,:] = W_insitu[i,:]/np.sum(W_insitu[i,:])
        W_insitu_for_cor = W_insitu/np.sum(W_insitu)
        X_for_cor = np.array(X)
        for i in range(X_for_cor.shape[1]):
            X_for_cor[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
        ngene = len(genes)
        cor_mat = np.empty([ngene, ngene], float)
        for i in range(ngene):
            for j in range(i, ngene):
                c = sci(X_for_cor[:,i], X_for_cor[:,j], W_insitu_for_cor)
                cor_mat[i,j] = c; cor_mat[j,i] = c
        W_insitu_for_cor = None
        grn_is = np.zeros([ngene, ngene], float)
        pid_bg_vecs = {}
        if not n_top_edge is None:
            cormat_flat = []; inds = []
            for i in range(ngene):
                for j in range(i, ngene):
                    cormat_flat.append(cor_mat[i,j])
                    inds.append([i,j])
            sorted_ind = np.argsort(-np.abs(cormat_flat))
            for iedge in range(n_top_edge):
                i,j = inds[sorted_ind[iedge]]
                if method == 'pid':
                    tmp_out = self.gene_pair_pid_effect_range(genes[i], genes[j], n_top_g = n_top_g_bg, cor_cut = cor_cut_bg, effect_ranges=[effect_range], p0=p0)
                    u_ij = tmp_out[0,1]
                    tmp_out = self.gene_pair_pid_effect_range(genes[j], genes[i], n_top_g = n_top_g_bg, cor_cut = cor_cut_bg, effect_ranges=[effect_range], p0=p0)
                    u_ji = tmp_out[0,1]
                elif method == 'ml':
                    tmp_out = self.gene_pair_ml_effect_range(genes[i], genes[j], n_top_g = n_top_g_bg, cor_cut = cor_cut_bg, effect_ranges=[effect_range])
                    u_ij = tmp_out[0,1]
                    tmp_out = self.gene_pair_ml_effect_range(genes[j], genes[i], n_top_g = n_top_g_bg, cor_cut = cor_cut_bg, effect_ranges=[effect_range])
                    u_ji = tmp_out[0,1]
                grn_is[i,j] = u_ij; grn_is[j,i] = u_ji
        elif not cor_cut is None:
            for i in range(ngene):
                for j in range(i, ngene):
                    if np.abs(cor_mat[i,j]) >= cor_cut:
                        if method == 'pid':
                            if not output_individual:
                                tmp_out = self.gene_pair_pid_effect_range(genes[i], genes[j], n_top_g = n_top_g_bg, cor_cut = cor_cut_bg, effect_ranges=[effect_range], p0=p0)
                                u_ij = tmp_out[0,1]
                                tmp_out = self.gene_pair_pid_effect_range(genes[j], genes[i], n_top_g = n_top_g_bg, cor_cut = cor_cut_bg, effect_ranges=[effect_range], p0=p0)
                                u_ji = tmp_out[0,1]
                            else:
                                print(i,j)
                                tmp_out, pid_bg_vec = self.gene_pair_pid_effect_range(genes[i], genes[j], n_top_g = n_top_g_bg, cor_cut = cor_cut_bg, effect_ranges=[effect_range], p0=p0, output_individual=True)
                                u_ij = tmp_out[0,1]; pid_bg_vecs[(i,j)] = pid_bg_vec[0]
                                tmp_out, pid_bg_vec = self.gene_pair_pid_effect_range(genes[j], genes[i], n_top_g = n_top_g_bg, cor_cut = cor_cut_bg, effect_ranges=[effect_range], p0=p0, output_individual=True)
                                u_ji = tmp_out[0,1]; pid_bg_vecs[(j,i)] = pid_bg_vec[0]
                        elif method == 'ml':
                            tmp_out = self.gene_pair_ml_effect_range(genes[i], genes[j], n_top_g = n_top_g_bg, cor_cut = cor_cut_bg, effect_ranges=[effect_range])
                            u_ij = tmp_out[0,1]
                            tmp_out = self.gene_pair_ml_effect_range(genes[j], genes[i], n_top_g = n_top_g_bg, cor_cut = cor_cut_bg, effect_ranges=[effect_range])
                            u_ji = tmp_out[0,1]
                        grn_is[i,j] = u_ij; grn_is[j,i] = u_ji
        df = pd.DataFrame(data=grn_is, columns=genes, index=genes)
        if output_individual:
            return df, pid_bg_vecs
        else:
            return df

    def visualize_cells(self, type=1, method='tsne', perplexity=30.0, umap_n_neighbors=5, umap_min_dist=0.1):
        """Visualization of cells.
        
        :param type: the type of visualization type=1 dimension reduction with spatial distance, label with original clusters;
            type=2 dimension reduction with scRNAseq, label with spatial subclusters;
            type=3 dimension reduction with spatial distance, label with spatial subclusters;
            type=4 dimension reduction with scRNAseq, label with original clusters.
        :type type: int
        """

        # type 1: dim reduction with spatial, label with org cluster
        # type 2: dim reduction with sequence, label with subcluster
        # type 3: dim reduction with spatial, label with subcluster
        # type 4: dim reduction with sequence, label with org cluster
        random_state = 92614
        if type == 1 or type == 4:
            if type == 1:
                if method == 'tsne':
                    X = manifold.TSNE(metric='precomputed', random_state=random_state, perplexity=perplexity).fit_transform(self.sc_dmat_spatial)
                elif method == 'umap':
                    X = umap.UMAP(metric='precomputed', random_state=random_state, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist).fit_transform(self.sc_dmat_spatial)
            elif type == 4:
                if method == 'tsne':
                    X = manifold.TSNE(metric='precomputed', random_state=random_state, perplexity=perplexity).fit_transform(self.clustering_dmat_sc)
                elif method == 'umap':
                    X = umap.UMAP(metric='precomputed', random_state=random_state, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist).fit_transform(self.clustering_dmat_sc)
            for i in range(self.clustering_ncluster_org):
                ids = self.clustering_partition_org[i]
                plt.scatter(X[ids,0], X[ids,1], label=str(i+1), linewidth=0, s=15)
            plt.legend(fontsize=10,loc='center left', bbox_to_anchor=(1, 0.5))
            plt.axis('off')
            plt.tight_layout()
            # plt.show()
        if type == 2 or type == 3:
            if type == 2:
                if method == 'tsne':
                    X = manifold.TSNE(metric='precomputed', random_state=random_state, perplexity=perplexity).fit_transform(self.clustering_dmat_sc)
                elif method == 'umap':
                    X = umap.UMAP(metric='precomputed', random_state=random_state, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist).fit_transform(self.clustering_dmat_sc)
            elif type == 3:
                if method == 'tsne':
                    X = manifold.TSNE(metric='precomputed', random_state=random_state, perplexity=perplexity).fit_transform(self.sc_dmat_spatial)
                elif method == 'umap':
                    X = umap.UMAP(metric='precomputed', random_state=random_state, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist).fit_transform(self.sc_dmat_spatial)
            ncl =self.clustering_ncluster_org
            nscl = self.clustering_nsubcluster
            par_ids = self.clustering_partition_inds
            scl_names = []
            scl_ids = []
            dmat_sc = self.sc_dmat_spatial
            nosub_cnt = 0
            sub_cnt = 0
            scl_colors = []
            backup_colors = ['b','r','g','y','k']
            print(nscl)
            for i in range(ncl):
                if nscl[i] == 1:
                    scl_names.append(str(i+1))
                    scl_ids.append(par_ids[(i,0)])
                    scl_colors.append("C"+str(nosub_cnt))
                    nosub_cnt += 1
                else:
                    for j in range(nscl[i]):
                        scl_names.append(str(i+1)+'.'+str(j+1))
                        scl_ids.append(par_ids[(i,j)])
                        if sub_cnt > 9:
                            scl_colors.append(backup_colors[sub_cnt-10])
                        else:
                            scl_colors.append("C"+str(sub_cnt))
                        sub_cnt += 1
            for i in range(len(scl_ids)):
                tmp_str = scl_names[i].split('.')
                if len(tmp_str) == 2:
                    marker = 's'
                elif len(tmp_str) == 1:
                    marker = 'o'
                plt.scatter(X[scl_ids[i],0], X[scl_ids[i],1], linewidth=0, label=scl_names[i], marker=marker, c=scl_colors[i], s=15)
            plt.legend(fontsize=10,loc='center left', bbox_to_anchor=(1, 0.5))
            plt.axis('off')
            plt.tight_layout()
            # plt.show()

    def visualize_subclusters(self, pts=None, k=3, cut=None, vmin=1.0/200.0, vmax=1.0/30.0, umap_k=3, figsize = (20,20)):
        """Visualize subclusters as a summary and distributions over the original geometry (2D).

        :param pts: the coordinates of original geometry (n_locations, 2)
        :type pts: class:`numpy.ndarray`
        :param k: the number nearest neighbors to connect in the subcluster summary plot
        :type k: int
        :param vmin: the vmin for colormap of the edges in the summary plot
        :type vmin: float
        :param vmax: the vmax for colormap of the edges in the summary plot
        :type vmax: float
        :param umap_k: the n_neighbors parameter in umap dimension reduction
        :type umap_k: int
        """
        random_state = 92614
        ncl =self.clustering_ncluster_org
        nscl = self.clustering_nsubcluster
        par_ids = self.clustering_partition_inds
        scl_names = []
        scl_ids = []
        nosub_cnt = 0
        sub_cnt = 0
        scl_colors = []
        backup_colors = ['b','r','g','y','k']
        dmat_sc = self.sc_dmat_spatial
        for i in range(ncl):
            if nscl[i] == 1:
                scl_names.append(str(i+1))
                scl_ids.append(par_ids[(i,0)])
                scl_colors.append("C"+str(nosub_cnt))
                nosub_cnt += 1
            else:
                for j in range(nscl[i]):
                    scl_names.append(str(i+1)+'.'+str(j+1))
                    scl_ids.append(par_ids[(i,j)])
                    if sub_cnt > 9:
                        scl_colors.append(backup_colors[sub_cnt-10])
                    else:
                        scl_colors.append("C"+str(sub_cnt))
                    sub_cnt += 1
        n = np.sum(nscl)
        dmat_scl_is = np.zeros([n,n], float)
        for i in range(n-1):
            for j in range(i+1, n):
                d = np.mean(dmat_sc[scl_ids[i],:][:,scl_ids[j]])
                dmat_scl_is[i,j] = d; dmat_scl_is[j,i] = d
        # print(dmat_scl_is)
        X = umap.UMAP(metric='precomputed', n_neighbors=umap_k, random_state=random_state).fit_transform(dmat_scl_is+3.0)
        cm = plt.get_cmap('Greys')
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        connection = 'knn'
        for i in range(n):
            if not cut is None:
                for j in range(n):
                    if dmat_scl_is[i,j] < cut:
                        plt.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], c=scalarMap.to_rgba(1.0/dmat_scl_is[i,j]))
            elif not k is None:
                sorted_arg = np.argsort(dmat_scl_is[i,:])[1:k+1]
                for j in sorted_arg:
                    # if dmat_scl_is[i,j] < 150:
                    plt.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], c=scalarMap.to_rgba(1.0/dmat_scl_is[i,j]))
        for i in range(n):
            tmp_str = scl_names[i].split('.')
            if len(tmp_str) == 2:
                marker='s'
            elif len(tmp_str) == 1:
                marker='o'
            plt.scatter([X[i,0]], [X[i,1]], s=200, marker=marker, c=scl_colors[i])
            plt.annotate(scl_names[i], (X[i,0], X[i,1]), fontsize=24)
        plt.axis('off')

        # Show origins of the subclusters
        if pts is None:
            print("input geometry as nx2 array")
        else:
            fig = plt.figure(figsize=figsize)
            for i in range(n):
                tmp_gamma = self.gamma_mapping[scl_ids[i],:]
                for j in range(tmp_gamma.shape[0]):
                    tmp_gamma[j,:] = tmp_gamma[j,:]/np.sum(tmp_gamma[j,:])
                p = np.sum(tmp_gamma, axis=0)
                p = p/np.max(p)
                ax = fig.add_subplot(n/4+1,4,i+1)
                ax.scatter(pts[:,0], pts[:,1], c=p, cmap='winter', s=10)
                ax.axis('off')
                ax.set_title(scl_names[i], fontsize=20)
            plt.tight_layout()

    def mapping_loo_validation(self, cor_matrices = None, cor_type = 'mcc', alpha = 0.0, epsilon = 0.1, rho = np.Inf, scaling = True, output_prediction=False):
        ng = len(self.is_genes)
        nc = self.sc_data.values.shape[0]
        if not self.is_data is None:
            ns = self.is_data.values.shape[0]
        elif not self.is_data_bin is None:
            ns = self.is_data_bin.values.shape[0]
        else:
            print("No spatial data in mapping_loo_validation")
        if cor_type == 'mcc':
            X_sc = np.array( self.sc_data_bin[self.is_genes], float )
            X_is = np.array( self.is_data_bin[self.is_genes], float )
        elif cor_type == 'scc':
            X_sc = np.array( self.sc_data[self.is_genes], float )
            X_is = np.array( self.is_data[self.is_genes], float )
        if cor_matrices is None:
            cor_matrices = np.empty([ng,nc,ns],float)
            for ig in ng:
                train = np.array([i for i in range(ng) if i!= ig], int)
                if cor_type == 'mcc':
                    cor = np.asarray([compute_mcc(p, g) \
                                     for p in X_sc[:,train] \
                                     for g in X_is[:,train]])
                elif cor_type == 'spc':
                    cor = np.asarray([spearmanr(p,g)[0] \
                                     for p in X_sc[:,train] \
                                     for g in X_is[:,train]])
                cor_matrices[ig,:,:] = cor[:,:]
        X_is_pred = np.empty(X_is.shape, float)
        aucs = []
        for ig in range(ng):
            print(ig)
            cor = cor_matrices[ig,:,:]
            cor_weight = np.exp(cor)
            for i in range(ns):
                cor_weight[:,i] = cor_weight[:,i]/np.sum(cor_weight[:,i])
            # C = np.exp(1-cor)
            C = ( np.exp(-cor) - np.exp(-1) ) / ( np.exp(1) - np.exp(-1) )
            w_a = np.sum(cor_weight,axis=1); w_a = w_a/np.sum(w_a)
            w_b = np.sum(cor_weight,axis=0); w_b = w_b/np.sum(w_b)
            # w_a = np.ones(C.shape[0])/C.shape[0]
            # w_b = np.ones(C.shape[1])/C.shape[1]
            if alpha == 0.0 and np.isinf(rho):
                gamma = ot.sinkhorn(w_a, w_b, C, epsilon)
            elif alpha == 0.0 and not np.isinf(rho):
                gamma = usot.uot(w_a, w_b, C, epsilon, rho=rho)
            else:
                C_is = self.is_dmat/np.max(self.is_dmat)
                C_sc = self.sc_dmat/np.max(self.sc_dmat)
                gamma = usot.usot(w_a, w_b, C, C_sc, C_is, alpha, epsilon=epsilon, rho=rho)
            for i in range(ns):
                gamma[:,i] = gamma[:,i]/np.sum(gamma[:,i])
            X_is_pred[:,ig] = np.matmul(gamma.T,
                np.array(self.sc_data[self.is_genes[ig]]).reshape(-1,1) )[:,0]
            auc = roc_auc_score(list(self.is_data_bin[self.is_genes[ig]]),
                                X_is_pred[:,ig])
            aucs.append( auc )
            print(ig, auc)
        print(np.mean(aucs))
        if output_prediction:
            return X_is_pred

def choose_landmarks(pts, n, dmat = None, method = 'maxmin', assignment = 'nearest'):
    """Choose a set of landmark points from a set of points.

    [1] De Silva, Vin, and Gunnar E. Carlsson. "Topological estimation using
    witness complexes." SPBG 4 (2004): 157-166.

    :param pts: coordinates of points (n_points, nD) needed if dmat not given
    :type pts: class:`numpy.ndarray`
    :param n: number of landmark points to select
    :type n: int
    :param dmat: the distance matrix for the points (n_points, n_points)
    :type dmat: class:`numpy.ndarray`
    :return: the indices of selected points and an assignment matrix to assign original points to landmark points (n_landmarks, n_points)
    :rtype: class:`numpy.ndarray`
    """

    np.random.seed(92614)
    if dmat is None:
        dmat = distance_matrix(pts,pts)
    N = dmat.shape[0]
    ind_select = []; cnt_select = 0
    ind_candidate = list(np.arange(N)); cnt_candidate = N
    while cnt_select < n:
        if cnt_select == 0:
            tmp_ind = np.random.choice(N)
            ind_select.append(tmp_ind)
            del ind_candidate[tmp_ind]
        else:
            tmp_dmat = dmat[ind_candidate,:][:,ind_select]
            tmp_dmat = tmp_dmat.reshape(len(ind_candidate), len(ind_select))
            tmp_dmat_min = np.min(tmp_dmat, axis=1)
            tmp_ind = np.argmax(tmp_dmat_min)
            ind_select.append(ind_candidate[tmp_ind])
            del ind_candidate[tmp_ind]
        cnt_select += 1
    asmat = np.zeros( [n,N], float )
    if assignment == 'nearest':
        for i in range(N):
            tmp_ind = np.argmin(dmat[i,ind_select])
            asmat[tmp_ind,i] = 1.0

    return ind_select, asmat

def phi_exp(x, eta, nu, p):
    """The exponential weight kernel. Computes exp(-(x/eta)^(p*nu)).

    :param x: the input value
    :type x: float or class:`numpy.ndarray`
    :param eta: the cutoff for this soft thresholding kernel
    :type eta: float
    :param nu: a possitive integer for the power term, a bigger nu gives sharper threshold boundary
    :type nu: int
    :param p: p=1: emphasize elements lower than cutoff; p=-1: emphasize elements higher than cutoff
    :type p: int
    :return: the kernel output with same shape of x
    :rtype: same as x
    """
    if np.min(x) < 0:
        print("warning: phi_exp taking a negative number")
    epsilon = 1E-8
    y = np.empty_like(x)
    if p == -1:
        nz_ind = np.where(x > epsilon)
        y = np.zeros_like(x)
        y[nz_ind] = np.exp(-np.power(x[nz_ind]/eta, p*nu))
    else:
        y = np.exp(-np.power(x/eta, p*nu))
    return y

def sci(x,y,W,scale=False):
    """Computes the spatial correlation index in Eq. (9) of [1].

    [1] Chen, Yanguang. "A new methodology of spatial cross-correlation
    analysis." PloS one 10.5 (2015): e0126158.

    :param x: the variable's values at the spatial locations
    :type x: class:`numpy.ndarray`
    :param y: the other variable's values at the spatial locations
    :type y: class:`numpy.ndarray`
    :param W: weight matrix (symmetric) among the locations with W[i,i] = 0
    :type W: class:`numpy.ndarray`
    :param scale: whether to scale the inputs s.t. (1) \sum_{i,j}W_{ij} = 1 and (2) x = (x-\mu(x))/\sigma(x)
    :type scale: boolean
    :return: a global spatial cross correlation index
    :rtype: float
    """

    if scale:
        x = (x-np.mean(x))/np.std(x)
        y = (y-np.mean(y))/np.std(y)
        for i in range(W.shape[0]):
            W[i,i] = 0
        W = W/np.sum(W)
    sci = np.dot(x.reshape(1,-1), W.dot(y.reshape(-1,1)))
    return sci

def knn_graph(D,k):
    """Construct a k-nearest-neighbor graph as igraph object.

    :param D: a distance matrix for constructing the knn graph
    :type D: class:`numpy.ndarray`
    :param k: number of nearest neighbors
    :type k: int
    :return: a knn graph object
    :rtype: class:`igraph.Graph`
    """

    n = D.shape[0]
    G = igraph.Graph()
    G.add_vertices(n)
    edges = []
    weights = []
    for i in range(n):
        sorted_ind = np.argsort(D[i,:])
        for j in range(1,1+k):
            # if i < sorted_ind[j]:
            edges.append( (i, sorted_ind[j]) )
            weights.append(D[i, sorted_ind[j]])
                # weights.append(1.)
    G.add_edges(edges)
    G.es['weight'] = weights
    return G

def knn_graph_nx(D,k):
    """Construct a k-nearest-neighbor graph as networkx object.

    :param D: a distance matrix for constructing the knn graph
    :type D: class:`numpy.ndarray`
    :param k: number of nearest neighbors
    :type k: int
    :return: a knn graph object and a list of edges
    :rtype: class:`networkx.Graph`, list of tuples
    """
    G_nx = nx.Graph()
    G_nx.add_nodes_from([i for i in range(D.shape[0])])
    edge_list = []
    for i in range(D.shape[0]):
        tmp_ids = np.argsort(D[i,:])
        for j in range(1,k+1):
            G_nx.add_edge(i,tmp_ids[j])
            edge_list.append((i,tmp_ids[j]))
    return G_nx, edge_list

def compute_mcc(true_labels, pred_labels):
    """Compute matthew's correlation coefficient.

    :param true_labels: 1D integer array
    :type true_labels: class:`numpy.ndarray`
    :param pred_labels: 1D integer array
    :type pred_labels: class:`numpy.ndarray`
    :return: mcc
    :rtype: float
    """
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    mcc = (TP * TN) - (FP * FN)
    denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if denom==0:
        return 0
    return mcc / denom