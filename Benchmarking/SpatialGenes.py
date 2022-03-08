import numpy as np
import pandas as pd
import sys
import pickle
import os
import time as tm
from functools import partial
import scipy.stats as st
from scipy.stats import wasserstein_distance
import scipy.stats
import copy
from sklearn.model_selection import KFold
import pandas as pd
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
from scipy import stats

from os.path import join

from scipy.spatial.distance import cdist
import h5py
from scipy.stats import spearmanr

import sys
from stPlus import *

class GenePrediction:
    def __init__(self, RNA_path, Spatial_path, location_path, device = None, train_list = None, test_list = None, outdir = None, modes = 'cells', annotate = None, CellTypeAnnotate = None):
        """
            @author: wen zhang
            This function integrates spatial and scRNA-seq data to predictes the expression of the spatially unmeasured genes from the scRNA-seq data.
            
            Please note that Tangram can be used in two ways : Tangram_image or Tangram_seq.
            Only when  you have information file that containing the number of cells in each spot, you can use Tangram_seq.
            
            A minimal example usage:
            Assume we have (1) scRNA-seq data count file named RNA_path
            (2) spatial transcriptomics count data file named Spatial_path
            (3) spatial spot coordinate file named location_path
            (4) gene list for integrations names train_list
            (5) gene list for prediction names test_list
            
            >>> import Benchmarking.SpatialGenes as SpatialGenes
            >>> test = SpatialGenes.GenePrediction(RNA_path, Spatial_path, location_path, train_list = train_list, test_list = test_list, outdir = outdir)
            >>> Methods = ['SpaGE','novoSpaRc','SpaOTsc','gimVI','Tangram_image','Seurat','LIGER']
            >>> Result = test.Imputing(Methods)
            
            Parameters
            -------
            RNA_path : str
            scRNA-seq data count file with Tab-delimited (genes X cells, each row is a gene and each col is a cell).
            
            Spatial_path : str
            spatial transcriptomics count data file with Tab-delimited (spots X genes, each col is a gene. Please note that the file has no index).
            
            location_path : str
            spatial spot coordinate file name with Tab-delimited (each col is a spot coordinate. Please note that the file has no index).
            default: None. It is necessary when you use SpaOTsc or novoSpaRc to integrate datasets.
            
            
            device : str
            Option,  [None,'GPU'], defaults to None
            
            train_list : list
            genes for integrations, Please note it must be a list.
            
            test_list : list
            genes for prediction, Please note it must be a list.
            
            outdir : str
            Outfile directory
            
            modes : str
            Only for Tangram. The default mapping mode is mode='cells',Alternatively, one can specify mode='clusters' which averages the single cells beloning to the same cluster (pass annotations via cluster_label). This is faster, and is our chioce when scRNAseq and spatial data come from different species 
           
            annoatet : str
            annotate for scRNA-seq data, if not None, you must be set CellTypeAnnotate labels for tangram.
            
            CellTypeAnnotate : dataframe
            CellType for scRNA-seq data, you can set this parameter for tangram.
            
            """
        
        self.RNA_file = RNA_path
        self.Spatial_file = Spatial_path
        self.locations = np.loadtxt(location_path, skiprows=1)
        self.train_list = train_list
        self.test_list = test_list
        self.RNA_data_adata = sc.read(RNA_path, sep = "\t",first_column_names=True).T
        self.Spatial_data_adata = sc.read(Spatial_path, sep = "\t")
        self.device = device
        self.outdir = outdir
        self.annotate = annotate
        self.CellTypeAnnotate = CellTypeAnnotate
        self.modes = modes
    
    
    def SpaGE_impute(self):
        sys.path.append("Extenrnal/SpaGE-master/")
        from SpaGE.main import SpaGE
        RNA_data = pd.read_table(self.RNA_file,header=0,index_col = 0)
        Spatial_data = pd.read_table(self.Spatial_file,sep='\t',header=0)
        RNA_data = RNA_data.loc[(RNA_data.sum(axis=1) != 0)]
        RNA_data = RNA_data.loc[(RNA_data.var(axis=1) != 0)]
        train_list, test_list = self.train_list, self.test_list
        predict = test_list
        feature = train_list
        pv = int(len(feature)/2)
        if pv > 100:
            pv = 100
        Spatial = Spatial_data[feature]
        Img_Genes = SpaGE(Spatial,RNA_data.T,n_pv=pv,genes_to_predict = predict)
        result = Img_Genes[predict]
        
        return result
    
    def gimVI_impute(self):
        import scvi
        import scanpy as sc
        from scvi.model import GIMVI
        import torch
        from torch.nn.functional import softmax, cosine_similarity, sigmoid
        Spatial_data_adata = self.Spatial_data_adata
        RNA_data_adata = self.RNA_data_adata
        train_list, test_list = self.train_list, self.test_list
        Genes  = train_list.copy()
        Genes.extend(test_list)
        rand_test_gene_idx = [Genes.index(x) for x in test_list]
        n_genes = len(Genes)
        rand_train_gene_idx = [Genes.index(x) for x in train_list]
        rand_train_genes = np.array(Genes)[rand_train_gene_idx]
        rand_test_genes = np.array(Genes)[rand_test_gene_idx]
        spatial_data_partial = Spatial_data_adata[:, rand_train_genes]
        sc.pp.filter_cells(spatial_data_partial, min_counts= 0)
        
        seq_data = copy.deepcopy(RNA_data_adata)
        
        seq_data = seq_data[:, Genes]
        sc.pp.filter_cells(seq_data, min_counts = 0)
        scvi.data.setup_anndata(spatial_data_partial)
        scvi.data.setup_anndata(seq_data)
        
        model = GIMVI(seq_data, spatial_data_partial)
        model.train(200)
        
        _, imputation = model.get_imputed_values(normalized=False)
        imputed = imputation[:, rand_test_gene_idx]
        result = pd.DataFrame(imputed, columns=rand_test_genes)
        return result
    
    def novoSpaRc_impute(self):
        import novosparc as nc
        RNA_data = pd.read_table(self.RNA_file,header=0,index_col = 0)
        Spatial_data = pd.read_table(self.Spatial_file,sep='\t',header=0)
        train_list, test_list = self.train_list, self.test_list
        gene_names = np.array(RNA_data.index.values)
        dge = RNA_data.values
        dge = dge.T
        num_cells = dge.shape[0]
        print ('number of cells and genes in the matrix:', dge.shape)
        
        hvg = np.argsort(np.divide(np.var(dge,axis=0),np.mean(dge,axis=0)+0.0001))
        dge_hvg = dge[:,hvg[-2000:]]
        
        num_locations = self.locations.shape[0]
        
        p_location, p_expression = nc.rc.create_space_distributions(num_locations, num_cells)
        cost_expression, cost_locations = nc.rc.setup_for_OT_reconstruction(dge_hvg,self.locations,num_neighbors_source = 5,num_neighbors_target = 5)
        
        insitu_matrix = np.array(Spatial_data[train_list])
        insitu_genes = np.array(Spatial_data[train_list].columns)
        test_genes = np.array(test_list)
        
        markers_in_sc = np.array([], dtype='int')
        for marker in insitu_genes:
            marker_index = np.where(gene_names == marker)[0]
            if len(marker_index) > 0:
                markers_in_sc = np.append(markers_in_sc, marker_index[0])
        
        cost_marker_genes = cdist(dge[:, markers_in_sc]/np.amax(dge[:, markers_in_sc]),insitu_matrix/np.amax(insitu_matrix))
        alpha_linear = 0.5
        gw = nc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,alpha_linear, p_expression, p_location,'square_loss', epsilon=5e-3, verbose=True)
        np.save(novo.npy, gw)
        sdge = np.dot(dge.T, gw)
        imputed = pd.DataFrame(sdge,index=RNA_data.index)
        result = imputed.loc[test_genes]
        result = result.T
        return result
    
    def SpaOTsc_impute(self):
        from spaotsc import SpaOTsc
        RNA_data = pd.read_table(self.RNA_file,header=0,index_col = 0)
        Spatial_data = pd.read_table(self.Spatial_file,sep='\t',header=0)
        train_list, test_list = self.train_list, self.test_list
        df_sc = RNA_data.T
        df_IS = Spatial_data
        pts = self.locations
        is_dmat = distance_matrix(pts, pts)
        
        
        df_is=df_IS.loc[:,train_list]
        
        gene_is=df_is.columns.tolist()
        gene_sc=df_sc.columns.tolist()
        gene_overloap=list(set(gene_is).intersection(gene_sc))
        a=df_is[gene_overloap]
        b=df_sc[gene_overloap]
        
        
        rho, pval = stats.spearmanr(a, b,axis=1)
        rho[np.isnan(rho)]=0
        mcc=rho[-(len(df_sc)):,0:len(df_is)]
        C = np.exp(1-mcc)
        
        issc = SpaOTsc.spatial_sc(sc_data=df_sc, is_data=df_is, is_dmat = is_dmat)
        
        issc.transport_plan(C**2, alpha=0, rho=1.0, epsilon=1.0, cor_matrix=mcc, scaling=False)
        gamma = issc.gamma_mapping
        np.save(novo.npy, gw)
        for j in range(gamma.shape[1]):
            gamma[:,j] = gamma[:,j]/np.sum(gamma[:,j])
        X_pred = np.matmul(gamma.T, np.array(issc.sc_data.values))
        
        result = pd.DataFrame(data=X_pred, columns=issc.sc_data.columns.values)
        test_genes = test_list
        result = result.loc[:,test_genes]
        return result

    def Tangram_impute(self):
        import torch
        from torch.nn.functional import softmax, cosine_similarity, sigmoid
        import tangram as tg
        RNA_data_adata = self.RNA_data_adata
        Spatial_data_adata = self.Spatial_data_adata
        train_list, test_list = self.train_list, self.test_list
        test_list = [x.lower() for x in test_list]
        if self.annotate == None:
            RNA_data_adata_label = RNA_data_adata
            sc.pp.normalize_total(RNA_data_adata_label)
            sc.pp.log1p(RNA_data_adata_label)
            sc.pp.highly_variable_genes(RNA_data_adata_label)
            RNA_data_adata_label = RNA_data_adata_label[:, RNA_data_adata_label.var.highly_variable]
            sc.pp.scale(RNA_data_adata_label, max_value=10)
            sc.tl.pca(RNA_data_adata_label)
            sc.pp.neighbors(RNA_data_adata_label)
            sc.tl.leiden(RNA_data_adata_label, resolution = 0.5)
            RNA_data_adata.obs['leiden']  = RNA_data_adata_label.obs.leiden
            tg.pp_adatas(RNA_data_adata, Spatial_data_adata, genes=train_list)
        else:
            CellTypeAnnotate = self.CellTypeAnnotate
            RNA_data_adata.obs['leiden']  = CellTypeAnnotate
            tg.pp_adatas(RNA_data_adata, Spatial_data_adata, genes=train_list)
        device = torch.device('cuda:0')
        if self.modes == 'clusters':
            ad_map = tg.map_cells_to_space(RNA_data_adata, Spatial_data_adata, device = device, mode = modes, cluster_label = 'leiden', density_prior = density)
            ad_ge = tg.project_genes(ad_map, RNA_data_adata, cluster_label = 'leiden')
        else:
            ad_map = tg.map_cells_to_space(RNA_data_adata, Spatial_data_adata, device = device)
            ad_ge = tg.project_genes(ad_map, RNA_data_adata)
        test_list = list(set(ad_ge.var_names) & set(test_list))
        test_list = np.array(test_list)
        pre_gene = pd.DataFrame(ad_ge[:,test_list].X, index=ad_ge[:,test_list].obs_names, columns=ad_ge[:,test_list].var_names)
        return pre_gene
    
    def stPlus_impute(self):
        outdir, train_list, test_list = self.outdir, self.train_list, self.test_list
        RNA_data = pd.read_table(self.RNA_file,header=0,index_col = 0)
        Spatial_data = pd.read_table(self.Spatial_file,sep='\t',header=0)
        save_path_prefix = join(outdir, 'process_file/stPlus-demo')
        if not os.path.exists(join(outdir, "process_file")):
            os.mkdir(join(outdir, "process_file"))
        stPlus_res = stPlus(Spatial_data[train_list], RNA_data.T, test_list, save_path_prefix)
        return stPlus_res

    def Imputing(self, need_tools):
        if "SpaGE" in need_tools:
            result_SpaGE = self.SpaGE_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_SpaGE.to_csv(self.outdir + "/SpaGE_impute.csv",header=1, index=1)
                
        if "gimVI" in need_tools:
            result_GimVI = self.gimVI_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_GimVI.to_csv(self.outdir + "gimVI_impute.csv",header=1, index=1)
                
        if "novoSpaRc" in need_tools:
            result_Novosparc = self.novoSpaRc_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_Novosparc.to_csv(self.outdir + "/novoSpaRc_impute.csv",header=1, index=1)
                
        if "SpaOTsc" in need_tools:
            result_Spaotsc = self.SpaOTsc_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_Spaotsc.to_csv(self.outdir + "/SpaOTsc_impute.csv",header=1, index=1)

        if "Tangram" in need_tools:
            result_Tangram = self.Tangram_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_Tangram.to_csv(self.outdir + "/Tangram_impute.csv",header=1, index=1)
                
        if "stPlus" in need_tools:
            result_stPlus = self.stPlus_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_stPlus.to_csv(self.outdir + "stPlus_impute.csv",header=1, index=1)

        if 'LIGER' in need_tools:
            train = ','.join(self.train_list)
            test = ','.join(self.test_list)
            os.system('Rscript Codes/Impute/LIGER.r ' + self.RNA_file + ' ' + self.Spatial_file + ' ' + train + ' ' + test + ' ' + self.outdir + '/LIGER_impute.txt')

        if 'Seurat' in need_tools:
            train = ','.join(self.train_list)
            test = ','.join(self.test_list)
            os.system ('Rscript Codes/Impute/Seurat.r ' + self.RNA_file + ' ' + self.Spatial_file + ' ' + train + ' ' + test + ' ' + self.outdir + '/Seurat_impute.txt')
