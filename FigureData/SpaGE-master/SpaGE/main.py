""" SpaGE [1]
@author: Tamim Abdelaal
This function integrates two single-cell datasets, spatial and scRNA-seq, and 
enhance the spatial data by predicting the expression of the spatially 
unmeasured genes from the scRNA-seq data.
The integration is performed using the domain adaption method PRECISE [2]
	
References
-------
    [1] Abdelaal T., Mourragui S., Mahfouz A., Reiders M.J.T. (2020)
    SpaGE: Spatial Gene Enhancement using scRNA-seq
    [2] Mourragui S., Loog M., Reinders M.J.T., Wessels L.F.A. (2019)
    PRECISE: A domain adaptation approach to transfer predictors of drug response
    from pre-clinical models to tumors
"""

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.neighbors import NearestNeighbors
from SpaGE.principal_vectors import PVComputation

def SpaGE(Spatial_data,RNA_data,n_pv,genes_to_predict=None):
    """
        @author: Tamim Abdelaal
        This function integrates two single-cell datasets, spatial and scRNA-seq, 
        and enhance the spatial data by predicting the expression of the spatially 
        unmeasured genes from the scRNA-seq data.
        
        Parameters
        -------
        Spatial_data : Dataframe
            Normalized Spatial data matrix (cells X genes).
        RNA_data : Dataframe
            Normalized scRNA-seq data matrix (cells X genes).
        n_pv : int
            Number of principal vectors to find from the independently computed
            principal components, and used to align both datasets. This should
            be <= number of shared genes between the two datasets.
        genes_to_predict : str array 
            list of gene names missing from the spatial data, to be predicted 
            from the scRNA-seq data. Default is the set of different genes 
            (columns) between scRNA-seq and spatial data.
            
        Return
        -------
        Imp_Genes: Dataframe
            Matrix containing the predicted gene expressions for the spatial 
            cells. Rows are equal to the number of spatial data rows (cells), 
            and columns are equal to genes_to_predict,  .
    """
    
    if genes_to_predict is SpaGE.__defaults__[0]:
        genes_to_predict = np.setdiff1d(RNA_data.columns,Spatial_data.columns)
        
    RNA_data_scaled = pd.DataFrame(data=st.zscore(RNA_data,axis=0),
                                   index = RNA_data.index,columns=RNA_data.columns)
    Spatial_data_scaled = pd.DataFrame(data=st.zscore(Spatial_data,axis=0),
                                   index = Spatial_data.index,columns=Spatial_data.columns)
    Common_data = RNA_data_scaled[np.intersect1d(Spatial_data_scaled.columns,RNA_data_scaled.columns)]
    
    Imp_Genes = pd.DataFrame(np.zeros((Spatial_data.shape[0],len(genes_to_predict))),
                                 columns=genes_to_predict)
    
    pv_Spatial_RNA = PVComputation(
            n_factors = n_pv,
            n_pv = n_pv,
            dim_reduction = 'pca',
            dim_reduction_target = 'pca'
    )
    
    pv_Spatial_RNA.fit(Common_data,Spatial_data_scaled[Common_data.columns])
    
    S = pv_Spatial_RNA.source_components_.T
        
    Effective_n_pv = sum(np.diag(pv_Spatial_RNA.cosine_similarity_matrix_) > 0.3)
    S = S[:,0:Effective_n_pv]
    
    Common_data_projected = Common_data.dot(S)
    Spatial_data_projected = Spatial_data_scaled[Common_data.columns].dot(S)
        
    nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto',
                            metric = 'cosine').fit(Common_data_projected)
    distances, indices = nbrs.kneighbors(Spatial_data_projected)
    
    for j in range(0,Spatial_data.shape[0]):
    
        weights = 1-(distances[j,:][distances[j,:]<1])/(np.sum(distances[j,:][distances[j,:]<1]))
        weights = weights/(len(weights)-1)
        Imp_Genes.iloc[j,:] = np.dot(weights,RNA_data[genes_to_predict].iloc[indices[j,:][distances[j,:] < 1]])
        
    return Imp_Genes
