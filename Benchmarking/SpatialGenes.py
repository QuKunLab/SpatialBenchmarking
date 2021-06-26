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


from scipy.spatial.distance import cdist
import h5py
from scipy.stats import spearmanr


import torch

from torch.nn.functional import softmax, cosine_similarity, sigmoid
import sys

class GenePrediction:
    def __init__(self, RNA_path, Spatial_path, location_path, count_path = None, device = None, train_list = None, test_list = None, norm = 'count', outdir = None):
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
            
            count_path : str
            count files containing the number of cells in each spot for Tangram (spots X numbers).
            each row represents the number of cells in each spot.
            Please note that has no index and the file columns must be 'cell_counts'.
            Option,  default: None. It is necessary when you use Tangram_seq functions to integrate datasets.
            
            
            device : str
            Option,  [None,'GPU'], defaults to None
            
            train_list : list
            genes for integrations, Please note it must be a list.
            
            test_list : list
            genes for prediction, Please note it must be a list.
            
            norm : str
            Option,  ['count','norm'], defaults to count. if norm, Seurat and LIGER
            will normlize the  spatial and scRNA-seq data before intergration.
            
            outdir : str
            Outfile directory
            """
        
        self.RNA_file = RNA_path
        self.Spatial_file = Spatial_path
        self.locations = np.loadtxt(location_path, skiprows=1)
        self.train_list = train_list
        self.test_list = test_list
        self.RNA_data_adata = sc.read(RNA_path, sep = "\t",first_column_names=True).T
        self.Spatial_data_adata = sc.read(Spatial_path, sep = "\t")
        self.device = device
        self.norm = norm
        print ('Please note you are using ' + self.norm + ' expression matrix to predict')
        if count_path != None:
            self.count =pd.read_table(count_path,sep='\t').astype(int)
            self.count[self.count.cell_counts==0]=1
        self.outdir = outdir
    
    
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
        
        if (len(feature)) < 50:
            pv = int(len(feature)-3)
        else:
            pv = 50
        Spatial = Spatial_data[feature]
        Img_Genes = SpaGE(Spatial,RNA_data.T,n_pv=pv,genes_to_predict = predict)
        result = Img_Genes[predict]
        
        return result
    
    def gimVI_impute(self):
        import scvi
        import scanpy as sc
        from scvi.model import GIMVI
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
        for j in range(gamma.shape[1]):
            gamma[:,j] = gamma[:,j]/np.sum(gamma[:,j])
        X_pred = np.matmul(gamma.T, np.array(issc.sc_data.values))
        
        result = pd.DataFrame(data=X_pred, columns=issc.sc_data.columns.values)
        test_genes = test_list
        result = result.loc[:,test_genes]
        return result

    def Tangram_impute_image(self):
        sys.path.append("Extenrnal/Tangram-master/")
        import mapping.utils
        import mapping.mapping_optimizer
        import mapping.plot_utils
        train_list, test_list = self.train_list, self.test_list
        RNA_data = pd.read_table(self.RNA_file,header=0,index_col = 0).T
        adata= sc.AnnData(RNA_data)
        Spatial_data = pd.read_table(self.Spatial_file,sep='\t',header=0)
        device = self.device
        if self.device == 'GPU':
            device = torch.device('cuda:0')
        hyperparm = {'lambda_d' : 1, 'lambda_g1' : 1, 'lambda_g2' : 0, 'lambda_r' : 0,
            'lambda_count' : 1, 'lambda_f_reg' : 1}
        learning_rate = 0.1
        num_epochs = 1000
        
        gene_diff = train_list
        spatial_data = Spatial_data[gene_diff]
        space_data= sc.AnnData(spatial_data)
        
        S = np.array(adata[:, gene_diff] .X)
        G = np.array(space_data.X)
        d = np.full(G.shape[0], 1/G.shape[0])
        S = np.log(1+S)
        mapper = mapping.mapping_optimizer.MapperConstrained(S=S, G=G, d=d, device=device, **hyperparm, target_count=G.shape[0])
        output, F_out = mapper.train(learning_rate=learning_rate, num_epochs=num_epochs)
        pre_gene = np.dot(adata[:, test_list].X.T, output)
        pre_gene =pd.DataFrame(pre_gene,index=test_list,columns=space_data.obs_names).T
                                                             
        return pre_gene
    
    def Tangram_impute_seq(self):
        sys.path.append("Extenrnal/Tangram-master/")
        import mapping.utils
        import mapping.mapping_optimizer
        import mapping.plot_utils
        if self.device == 'GPU':
            device = torch.device('cuda:0')
        train_list, test_list = self.train_list, self.test_list
        RNA_data = pd.read_table(self.RNA_file,header=0,index_col = 0).T
        adata= sc.AnnData(RNA_data)
        Spatial_data = pd.read_table(self.Spatial_file,sep='\t',header=0)
        device = self.device
        hyperparm = {'lambda_d' : 1, 'lambda_g1' : 1, 'lambda_g2' : 0, 'lambda_r' : 0,
            'lambda_count' : 1, 'lambda_f_reg' : 1}
        learning_rate = 0.1
        num_epochs = 6000
        
        gene_diff = train_list
        spatial_data = Spatial_data[gene_diff]
        space_data = sc.AnnData(spatial_data)
        space_data.obs['cell_count'] = self.count.cell_counts.values
        
        S = np.array(adata[:, gene_diff].X)
        G = np.array(space_data.X)
        d = np.array(space_data.obs.cell_count)/space_data.obs.cell_count.sum()
        mapper = mapping.mapping_optimizer.MapperConstrained(S=S, G=G, d=d, device=device, **hyperparm, target_count = space_data.obs.cell_count.sum())
        output, F_out = mapper.train(learning_rate=learning_rate, num_epochs=num_epochs)
        pre_gene = np.dot(adata[:, test_list].X.T, output)
        pre_gene =pd.DataFrame(pre_gene,index=test_list,columns=space_data.obs_names).T
                                                             
        return pre_gene

    def Imputing(self, need_tools):
        if "SpaGE" in need_tools:
            result_SpaGE = self.SpaGE_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_SpaGE.to_csv(self.outdir + "/result_SpaGE.csv",header=1, index=1)
                
        if "gimVI" in need_tools:
            result_GimVI = self.gimVI_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_GimVI.to_csv(self.outdir + "result_gimVI.csv",header=1, index=1)
                
        if "novoSpaRc" in need_tools:
            result_Novosparc = self.novoSpaRc_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_Novosparc.to_csv(self.outdir + "/result_novoSpaRc.csv",header=1, index=1)
                
        if "SpaOTsc" in need_tools:
            result_Spaotsc = self.SpaOTsc_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_Spaotsc.to_csv(self.outdir + "/result_SpaOTsc.csv",header=1, index=1)
                
        if "Tangram_image" in need_tools:
            result_Tangram_image = self.Tangram_impute_image()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_Tangram_image.to_csv(self.outdir + "/result_Tangram_image.csv",header=1, index=1)
                
        if "Tangram_seq" in need_tools:
            result_Tangram_seq = self.Tangram_impute_seq()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_Tangram_seq.to_csv(self.outdir + "result_Tangram_seq.csv",header=1, index=1)

        if 'LIGER' in need_tools:
            train = ','.join(self.train_list)
            test = ','.join(self.test_list)
            os.system('Rscript Benchmarking/Liger.r ' + self.RNA_file + ' ' + self.Spatial_file + ' ' + train + ' ' + test + ' ' + self.norm  + ' ' + self.outdir + '/Result_LIGER.txt')

        if 'Seurat' in need_tools:
            train = ','.join(self.train_list)
            test = ','.join(self.test_list)
            os.system ('Rscript Benchmarking/Seurat.r ' + self.RNA_file + ' ' + self.Spatial_file + ' ' + train + ' ' + test + ' ' + self.norm + ' ' + self.outdir + '/Result_Seurat_.txt')
