##import packages
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
import subprocess
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr,ttest_ind,mannwhitneyu

warnings.filterwarnings('ignore')

#SpaOTsc
from spaotsc import SpaOTsc
from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
from scipy import stats

#novoSpaRc
import novosparc as nc
from scipy.spatial.distance import cdist
import h5py

# add Tangram to path
import anndata
import torch
import sys

sys.path.append("Tangram-master") 
import mapping.utils
import mapping.mapping_optimizer
import mapping.plot_utils


class MappingCell:
    def __init__(self, RNA_path, Spatial_path, location_path, count_path = None, device = 'CPU', scrna_meta = None, subclass_mapper = None, gd_result = None, outdir = None):
        self.RNA_path = RNA_path
        self.Spatial_path = Spatial_path
        self.RNA_data =  pd.read_csv(RNA_path, sep='\t', index_col = 0)
        self.Spatial_data = pd.read_csv(Spatial_path, sep='\t', header=0)
        self.locations = np.loadtxt(location_path, skiprows=1)
        self.device = device
        if count_path != None:
            self.count =pd.read_csv(count_path,sep='\t', index_col = 0).astype(int)
        self.scrna_meta = scrna_meta
        self.subclass_mapper = subclass_mapper
        self.gd =  pd.read_csv(gd_result, sep='\t', index_col = 0)
        self.outdir = outdir

    def novoSpaRc(self):
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
        
        insitu_genes = self.Spatial_data.columns & gene_names
        insitu_matrix = self.Spatial_data.loc[:,insitu_genes]
        
        markers_in_sc = np.array([], dtype='int')
        for marker in insitu_genes:
            marker_index = np.where(gene_names == marker)[0]
            if len(marker_index) > 0:
                markers_in_sc = np.append(markers_in_sc, marker_index[0])
        
        cost_marker_genes = cdist(dge[:, markers_in_sc]/np.amax(dge[:, markers_in_sc]),insitu_matrix/np.amax(insitu_matrix))
        alpha_linear = 0.5
        gw = nc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,alpha_linear, p_expression, p_location,'square_loss', epsilon=5e-3, verbose=True)
        print ('we use novoSpaRc to predict')
        np.save(self.outdir + '/novoSpaRc_alignment.npy',gw)
        novoSpaRc_map = gw
        sc_rna_meta = pd.read_csv(self.scrna_meta, sep = '\t', header = 0, index_col = 0)
        novoSpaRc_results=pd.DataFrame(np.zeros((novoSpaRc_map.shape[1],len(np.unique(sc_rna_meta['subclass'])))),columns=np.unique(sc_rna_meta['subclass']))
        for c in np.unique(sc_rna_meta['subclass']):
            novoSpaRc_results.loc[:,c] =  novoSpaRc_map[np.where(sc_rna_meta.subclass == c)[0],:].mean(axis=0)
        if self.subclass_mapper is not None:
            print ('Mapper is using') 
            novoSpaRc_results.columns = [self.subclass_mapper[c] for c in novoSpaRc_results.columns]
            novoSpaRc_pro_results = pd.DataFrame(np.zeros((len(novoSpaRc_results.index), len(np.unique(novoSpaRc_results.columns)))),columns=np.unique(novoSpaRc_results.columns))
            for c in np.unique(novoSpaRc_pro_results.columns):
                if len(novoSpaRc_results.loc[:,c].shape) > 1:
                    print (c)
                    novoSpaRc_pro_results.loc[:,c] = novoSpaRc_results.loc[:,c].sum(axis=1).values
                else:
                    novoSpaRc_pro_results.loc[:,c] = novoSpaRc_results.loc[:,c].values
            novoSpaRc_results = novoSpaRc_pro_results
        CellType = novoSpaRc_results.columns & self.gd.columns
        novoSpaRc_results = novoSpaRc_results[CellType]
        novoSpaRc_results = (novoSpaRc_results.T/novoSpaRc_results.sum(axis=1)).T
        novoSpaRc_results = novoSpaRc_results.fillna(0)
        novoSpaRc_results.to_csv(self.outdir + '/novoSpaRc_CellType_Proportion.txt')     
        
    def SpaOTsc(self):
        df_sc = self.RNA_data.T
        df_IS = self.Spatial_data
        pts = self.locations
        is_dmat = distance_matrix(pts, pts)     
            
        df_is=df_IS    
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
        print ('we use SpaOTsc to predict')
        issc.transport_plan(C**2, alpha=0, rho=1.0, epsilon=1.0, cor_matrix=mcc, scaling=False)
        gamma = issc.gamma_mapping
        for j in range(gamma.shape[1]):
            gamma[:,j] = gamma[:,j]/np.sum(gamma[:,j])
        np.save(self.outdir + '/SpaOTsc_alignment.npy',gamma)
        SpaOTsc_map = gamma
        sc_rna_meta = pd.read_csv(self.scrna_meta, sep = '\t', header = 0, index_col = 0)
        SpaOTsc_results=pd.DataFrame(np.zeros((SpaOTsc_map.shape[1],len(np.unique(sc_rna_meta['subclass'])))),columns=np.unique(sc_rna_meta['subclass']))
        for c in np.unique(sc_rna_meta['subclass']):
            SpaOTsc_results.loc[:,c] =  SpaOTsc_map[np.where(sc_rna_meta.subclass == c)[0],:].mean(axis=0)
        if self.subclass_mapper is not None:
            SpaOTsc_results.columns = [self.subclass_mapper[c] for c in SpaOTsc_results.columns]
            SpaOTsc_pro_results = pd.DataFrame(np.zeros((len(SpaOTsc_results.index), len(np.unique(SpaOTsc_results.columns)))),columns=np.unique(SpaOTsc_results.columns))
            for c in np.unique(SpaOTsc_pro_results.columns):
                if len(SpaOTsc_results.loc[:,c].shape) > 1:
                    print (c)
                    SpaOTsc_pro_results.loc[:,c] = SpaOTsc_results.loc[:,c].sum(axis=1).values
                else:
                    SpaOTsc_pro_results.loc[:,c] = SpaOTsc_results.loc[:,c].values
            SpaOTsc_results = SpaOTsc_pro_results
        CellType = SpaOTsc_results.columns & self.gd.columns
        SpaOTsc_results = SpaOTsc_results[CellType]
        SpaOTsc_results = (SpaOTsc_results.T/SpaOTsc_results.sum(axis=1)).T
        SpaOTsc_results = SpaOTsc_results.fillna(0)
        SpaOTsc_results.to_csv(self.outdir + '/SpaOTsc_CellType_proportion.txt')
    
    def Tangram(self,):
        rna_df = self.RNA_data
        adata = anndata.AnnData(rna_df.T)
        spatial_df = self.Spatial_data
        space_data = anndata.AnnData(spatial_df)
        combined_cell_counts = self.count
        space_data.obs['cell_count'] = combined_cell_counts.values
        space_data = space_data[space_data.obs['cell_count'] > 0]
        mask_prior_indices, mask_adata_indices, selected_genes =  mapping.utils.get_matched_genes(
            space_data.var_names, adata.var_names
        )
        S = np.array(adata[:, mask_adata_indices].X)
        G = np.array(space_data[:, mask_prior_indices].X)
        d = np.array(space_data.obs.cell_count)/space_data.obs.cell_count.sum()
        device = self.device
        if self.device == 'GPU':
            device = torch.device('cuda:0')
        hyperparm = {'lambda_d' : 1, 'lambda_g1' : 1, 'lambda_g2' : 0, 'lambda_r' : 0,
                'lambda_count' : 1, 'lambda_f_reg' : 1}
        learning_rate = 0.1
        num_epochs = 6000
        print ('we use Tangram to predict')
        mapper = mapping.mapping_optimizer.MapperConstrained(
            S=S, G=G, d=d, device=device, **hyperparm, target_count = space_data.obs.cell_count.sum()
        )
        output_all, F_out_all = mapper.train(learning_rate=learning_rate, num_epochs=num_epochs)
        sc_rna_meta = pd.read_csv(scrna_meta, sep = '\t', header=0,index_col = 0)
        adata.obs = sc_rna_meta
    
        df_classes = mapping.utils.one_hot_encoding(adata.obs.subclass)
        prob_assign = mapping.utils.transfer_annotations_prob_filter(output_all, F_out_all, df_classes)
        prob_assign.to_csv(self.outdir + '/Tangram_alignment.txt')
        Tangram_results = pd.read_csv(self.outdir + '/Tangram_alignment.txt',index_col=0)
        if self.subclass_mapper is not None:
            Tangram_results.columns = [self.subclass_mapper[c] for c in Tangram_results.columns]
            Tangram_pro_results = pd.DataFrame(np.zeros((len(Tangram_results.index), len(np.unique(Tangram_results.columns)))),columns=np.unique(Tangram_results.columns))
            for c in np.unique(Tangram_pro_results.columns):
                if len(Tangram_results.loc[:,c].shape) > 1:
                    Tangram_pro_results.loc[:,c] = Tangram_results.loc[:,c].sum(axis=1).values
                else:
                    Tangram_pro_results.loc[:,c] = Tangram_results.loc[:,c].values
            Tangram_results = Tangram_pro_results
        CellType = Tangram_results.columns & self.gd.columns
        Tangram_results = Tangram_results[CellType]
        Tangram_results = (Tangram_results.T/Tangram_results.sum(axis=1)).T
        Tangram_results = Tangram_results.fillna(0)
        Tangram_results.to_csv(self.outdir + '/Tangram_CellType_proportion.txt')
    
    def Seurat(self):
        rna_df = self.RNA_path
        spatial_df = self.Spatial_path
        scrna_meta = self.scrna_meta
        print ('we use seurat to predict')
        os.popen('Rscript Seurat_CellAssigment.r ' + spatial_df + ' ' + rna_df + ' ' + scrna_meta + ' ' + self.outdir)
        print ('Rscript Seurat_CellAssigment.rr ' + spatial_df + ' ' + rna_df + ' ' + scrna_meta + ' ' + self.outdir)
        seurat_results = pd.read_csv(self.outdir + '/Seurat_alignment.txt', index_col=0)
        seurat_results = seurat_results.iloc[:,1:-1]
        Cols = seurat_results.columns
        used_ind = [(x.split('score.')[1]) for x in Cols]
        seurat_results.columns = used_ind
        seurat_results.index = np.arange(len(seurat_results))
        if self.subclass_mapper is not None:
            seurat_results.columns = [self.subclass_mapper[c] for c in seurat_results.columns]
            seurat_pro_results = pd.DataFrame(np.zeros((len(seurat_results.index), len(np.unique(seurat_results.columns)))),columns=np.unique(seurat_results.columns))
            for c in np.unique(seurat_results.columns):
                if len(seurat_pro_results.loc[:,c].shape) > 1:
                    seurat_pro_results.loc[:,c] = seurat_results.loc[:,c].sum(axis=1).values
                else:
                    seurat_pro_results.loc[:,c] = seurat_results.loc[:,c].values
            seurat_results = seurat_pro_results
        CellType = seurat_results.columns & self.gd.columns
        seurat_results = seurat_results[CellType]
        seurat_results = (seurat_results.T/seurat_results.sum(axis=1)).T
        seurat_results = seurat_results.fillna(0)
        seurat_results.to_csv(self.outdir + '/seurat_CellType_proportion.txt')
        
        
    def workstart(self,Tools):
        if "novoSpaRc" in Tools:
            self.novoSpaRc()
        if "Tangram" in Tools:
            self.Tangram()
        if "Seurat" in Tools:
            self.Seurat()
        if "SpaOTsc" in Tools:
            self.SpaOTsc()

