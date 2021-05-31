#!/usr/bin/env python
# coding: utf-8



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



#SpaGE
from SpaGE.main import SpaGE



#spaotsc
from spaotsc import SpaOTsc
from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
from scipy import stats




#novosparc
import novosparc as nc
from scipy.spatial.distance import cdist
import h5py



#gimVI
import scvi
import scanpy as sc
from scvi.model import GIMVI
from scipy.stats import spearmanr




#tangram
import torch
from IPython.display import display
# torch imports
from torch.nn.functional import softmax, cosine_similarity, sigmoid
import sys
sys.path.append("Tangram-master/")
# Tangram imports
import mapping.utils
import mapping.mapping_optimizer
import mapping.plot_utils




def kfold(Gene_list,k):
    kf = KFold(n_splits=k)
    train_list = []
    test_list = []
    for train, test in kf.split(Gene_list):
            train_list.append(list(Gene_list[train.tolist()]))
            test_list.append(list(Gene_list[test.tolist()]))
    return train_list, test_list

def SelectHvg(SpatialFile,scFile,OutFile):
        osmFISH_data = pd.read_table(SpatialFile,sep='\t',header=0).T
        cell_count = np.sum(osmFISH_data,axis=0)
        def Log_Norm_spatial(x):
            return np.log(((x/np.sum(x))*np.median(cell_count)) + 1)
        dge = osmFISH_data.apply(Log_Norm_spatial,axis=0)
        dge = dge.T
        hvg = np.argsort(np.divide(np.var(dge,axis=0),np.mean(dge,axis=0)+0.0001))
        dge_hvg = dge[hvg[-1000:].index]
        scRNA = pd.read_table(scFile,sep='\t',header=0)
        Genes = scRNA.index & dge_hvg
        osmFISH_data[Genes].to_csv(OutFilels,sep ='\t',index=False)
        return osmFISH_data[Genes]



class Impute:
    #__slots__ = ['RNA_path', 'Spatial_path', 'location_path', 'count_path']
    def __init__(self, RNA_path, Spatial_path, location_path, count_path = None, device = None, train_list = None, test_list = None):
        self.RNA_data =  pd.read_table(RNA_path,header=0,index_col = 0)
        self.Spatial_data = pd.read_table(Spatial_path,sep='\t',header=0)
        self.locations = np.loadtxt(location_path, skiprows=1)
        self.train_list = train_list
        self.test_list = test_list
        self.RNA_data_adata = sc.read(RNA_path, sep = "\t",first_column_names=True).T
        self.Spatial_data_adata = sc.read(Spatial_path, sep = "\t")
        self.device = device
        if count_path != None:
            self.count =pd.read_table(count_path,sep='\t').astype(int)
            self.count[self.count.cell_counts==0]=1
        
        
    def SpaGE_impute(self, args):
        RNA_data = self.RNA_data.loc[(self.RNA_data.sum(axis=1) != 0)]
        RNA_data = self.RNA_data.loc[(self.RNA_data.var(axis=1) != 0)]
        train_list, test_list = args
        predict = test_list
        feature = train_list

        if (len(feature)) < 50: 
            pv = int(len(feature))
        else:
            pv = 50
        Spatial = self.Spatial_data.drop(predict,axis=1)
        Img_Genes = SpaGE(Spatial,self.RNA_data.T,n_pv=pv,genes_to_predict = predict)
        result = Img_Genes[predict]

        return result
    
    def GimVI_impute(self, args):
        train_list, test_list  = args
        Genes  = list(self.Spatial_data_adata.var_names)
        rand_train_genes = np.array(train_list)
        rand_test_genes = np.array(test_list)
        spatial_data_partial = self.Spatial_data_adata[:, rand_train_genes]
        
        sc.pp.filter_cells(spatial_data_partial, min_counts= 0)
        
        seq_data = copy.deepcopy(self.RNA_data_adata)
        
        seq_data = seq_data[:, Genes].copy()
        sc.pp.filter_cells(seq_data, min_counts = 1)
        print(spatial_data_partial.shape)
        print(seq_data.shape)
        
        scvi.data.setup_anndata(spatial_data_partial)
        scvi.data.setup_anndata(seq_data)
        
        model = GIMVI(seq_data, spatial_data_partial)
        model.train(200)
        
        _, fish_imputation = model.get_imputed_values(normalized=False)
        imputed = fish_imputation[:, test_list]
        print(imputed)
        result = pd.DataFrame(imputed, columns=rand_test_genes)
        return result
    
    def Novosparc_impute(self, args):
        train_list, test_list = args
        gene_names = self.RNA_data.index.values
        dge = self.RNA_data.values
        dge = dge.T
        num_cells = dge.shape[0]
        print ('number of cells and genes in the matrix:', dge.shape)
    
        hvg = np.argsort(np.divide(np.var(dge,axis=0),np.mean(dge,axis=0)+0.0001))
        dge_hvg = dge[:,hvg[-2000:]]
        
        num_locations = self.locations.shape[0]
    
        p_location, p_expression = nc.rc.create_space_distributions(num_locations, num_cells)
        cost_expression, cost_locations = nc.rc.setup_for_OT_reconstruction(dge_hvg,self.locations,num_neighbors_source = 5,num_neighbors_target = 5)
        
        insitu_genes = train_list
        insitu_matrix = self.Spatial_data.loc[:,train_list]
        test_genes = test_list
        test_matrix = self.Spatial_data.loc[:,test_list]
        
        markers_in_sc = np.array([], dtype='int')
        for marker in insitu_genes:
            marker_index = np.where(gene_names == marker)[0]
            if len(marker_index) > 0:
                markers_in_sc = np.append(markers_in_sc, marker_index[0])
        print (len(markers_in_sc))
        
        cost_marker_genes = cdist(dge[:, markers_in_sc]/np.amax(dge[:, markers_in_sc]),insitu_matrix/np.amax(insitu_matrix))
        alpha_linear = 0.5
        gw = nc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,alpha_linear, p_expression, p_location,'square_loss', epsilon=5e-3, verbose=True)
        sdge = np.dot(dge.T, gw)
        imputed = pd.DataFrame(sdge,index=self.RNA_data.index)
        print(imputed)
        result = imputed.loc[test_genes]
        result = result.T
        print(result)
        return result
    
    def Spaotsc_impute(self, args):
        train_list, test_list = args
        df_sc = self.RNA_data.T
        df_IS = self.Spatial_data
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
        X_pred = np.matmul(gamma.T, np.array( issc.sc_data.values))

        result = pd.DataFrame(data=X_pred, columns=issc.sc_data.columns.values)
        print(result.shape)
        test_genes = test_list

        result = result.loc[:,test_genes]  
        return result
    
    def Tangram_impute_image(self, args):
        train_list, test_list = args
    
        adata = self.RNA_data_adata
        device = torch.device('cuda:0')
        hyperparm = {'lambda_d' : 1, 'lambda_g1' : 1, 'lambda_g2' : 0, 'lambda_r' : 0,
                'lambda_count' : 1, 'lambda_f_reg' : 1}
        learning_rate = 0.1
        num_epochs = 1000
        
        gene_diff = train_list
        spatial_data=self.Spatial_data[gene_diff]
        space_data=sc.AnnData(spatial_data)
        
        S = np.array(adata[:, gene_diff] .X) 
        G = np.array(space_data.X) 
        d = np.full(G.shape[0], 1/G.shape[0])  
        S = np.log(1+S)
        mapper = mapping.mapping_optimizer.MapperConstrained(
            S=S, G=G, d=d, device=device, **hyperparm, target_count=G.shape[0])
        output, F_out = mapper.train(learning_rate=learning_rate, num_epochs=num_epochs)
        pre_gene = np.dot(adata[:, test_list].X.T, output)
        pre_gene =pd.DataFrame(pre_gene,index=test_list,columns=space_data.obs_names).T
        
        return pre_gene
    
    def Tangram_impute_seq(self, args):
        train_list, test_list = args
        
        adata = self.RNA_data_adata
        device = torch.device('cuda:0')
        hyperparm = {'lambda_d' : 1, 'lambda_g1' : 1, 'lambda_g2' : 0, 'lambda_r' : 0,
                'lambda_count' : 1, 'lambda_f_reg' : 1}
        learning_rate = 0.1
        num_epochs = 6000
        
        gene_diff = train_list
        spatial_data=self.Spatial_data[gene_diff]
        space_data=sc.AnnData(spatial_data)
        space_data.obs['cell_count'] = self.count.cell_counts.values
        
        S = np.array(adata[:, gene_diff].X) 
        G = np.array(space_data.X) 
        d = np.array(space_data.obs.cell_count)/space_data.obs.cell_count.sum() 
        #S = np.log(1+S)
        mapper = mapping.mapping_optimizer.MapperConstrained(
        S=S, G=G, d=d, device=device, **hyperparm, target_count = space_data.obs.cell_count.sum())
        output, F_out = mapper.train(learning_rate=learning_rate, num_epochs=num_epochs)
        pre_gene = np.dot(adata[:, test_list].X.T, output)
        pre_gene =pd.DataFrame(pre_gene,index=test_list,columns=space_data.obs_names).T
        
        return pre_gene

    def pool(self, need_tools):
        #并行
        if "SpaGE" in need_tools:
            
            with multiprocessing.Pool(10) as pool:
                result_SpaGE = pd.concat(pool.map(self.SpaGE_impute, iterable=zip(self.train_list, self.test_list)),axis=1) 
                if not os.path.exists("./impute_output/"):
                    os.mkdir("./impute_output/")
                result_SpaGE.to_csv("./impute_output/result_SpaGE.csv",header=1, index=1)
                
        if "GimVI" in need_tools:
           
            with multiprocessing.Pool(10) as pool:
                result_GimVI = pd.concat(pool.map(self.GimVI_impute, iterable=zip(self.train_list, self.test_list)),axis=1) 
                if not os.path.exists("./impute_output/"):
                    os.mkdir("./impute_output/")
                result_GimVI.to_csv("./impute_output/result_GimVI.csv",header=1, index=1)
                
        if "Novosparc" in need_tools:
            
            with multiprocessing.Pool(10) as pool:
                result_Novosparc = pd.concat(pool.map(self.Novosparc_impute, iterable=zip(self.train_list, self.test_list)),axis=1) 
                if not os.path.exists("./impute_output/"):
                    os.mkdir("./impute_output/")
                result_Novosparc.to_csv("./impute_output/result_Novosparc.csv",header=1, index=1)
                
        if "Spaotsc" in need_tools:
            
            with multiprocessing.Pool(10) as pool:
                result_Spaotsc = pd.concat(pool.map(self.Spaotsc_impute, iterable=zip(self.train_list, self.test_list)),axis=1) 
                if not os.path.exists("./impute_output/"):
                    os.mkdir("./impute_output/")
                result_Spaotsc.to_csv("./impute_output/result_Spaotsc.csv",header=1, index=1)
                
        if "Tangram_image" in need_tools:
            
            with multiprocessing.Pool(10) as pool:
                result_Tangram_image = pd.concat(pool.map(self.Tangram_impute_image, iterable=zip(self.train_list, self.test_list)),axis=1) 
                if not os.path.exists("./impute_output/"):
                    os.mkdir("./impute_output/")
                result_Tangram_image.to_csv("./impute_output/result_Tangram_image.csv",header=1, index=1)
                
        if "Tangram_seq" in need_tools:
            
            with multiprocessing.Pool(10) as pool:
                result_Tangram_seq = pd.concat(pool.map(self.Tangram_impute_seq, iterable=zip(self.train_list, self.test_list)),axis=1) 
                if not os.path.exists("./impute_output/"):
                    os.mkdir("./impute_output/")
                result_Tangram_seq.to_csv("./impute_output/result_Tangram_seq.csv",header=1, index=1)


                
        
        
        






