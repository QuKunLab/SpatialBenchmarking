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
import matplotlib

warnings.filterwarnings('ignore')

from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
from scipy import stats

from scipy.spatial.distance import cdist
import h5py

import anndata
import torch
import sys


class MappingCell:
    def __init__(self, RNA_path, Spatial_path, location_path = None, count_path = None, device = None, scrna_annotation = None, annotatetype = None, gd_result = None, outdir = None):
        self.RNA_path = RNA_path
        self.Spatial_path = Spatial_path
        self.RNA_data =  pd.read_csv(RNA_path, sep='\t', index_col = 0)
        self.Spatial_data = pd.read_csv(Spatial_path, sep='\t', header=0)
        self.locations = np.loadtxt(location_path, skiprows=1)
        self.device = device
        if count_path != None:
            self.count =pd.read_csv(count_path,sep='\t', index_col = 0).astype(int)
        self.scrna_annotationfiles = scrna_annotation
        self.scrna_annotation = pd.read_csv(scrna_annotation, sep='\t', header=0)
        if gd_result != None:
            self.gd =  pd.read_csv(gd_result, sep='\t', index_col = 0)
        else:
            self.gd = None
        self.annotatetype = annotatetype
        self.outdir = outdir

    def novoSpaRc(self):
        import novosparc as nc
        from spaotsc import SpaOTsc
        gene_names = self.RNA_data.index.values
        dge = self.RNA_data.values
        dge = dge.T
        num_cells = dge.shape[0]
        print ('number of cells and genes in the matrix:', dge.shape)
    
        hvg = np.argsort(np.divide(np.var(dge,axis=0),np.mean(dge,axis=0)+0.0001))
        dge_hvg = dge[:,hvg[-2000:]]
        #dge_hvg = dge
        
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
        scrna_annotation = self.scrna_annotation[self.annotatetype]
        novoSpaRc_results=pd.DataFrame(np.zeros((novoSpaRc_map.shape[1],len(np.unique(scrna_annotation)))),columns=np.unique(scrna_annotation))
        for c in np.unique(scrna_annotation):
            novoSpaRc_results.loc[:,c] =  novoSpaRc_map[np.where(scrna_annotation == c)[0],:].mean(axis=0)
        mapperdict = dict(zip(self.scrna_annotation[self.annotatetype],self.scrna_annotation['celltype']))
        novoSpaRc_results.columns = [mapperdict[c] for c in novoSpaRc_results.columns]
        novoSpaRc_pro_results = pd.DataFrame(np.zeros((len(novoSpaRc_results.index), len(np.unique(novoSpaRc_results.columns)))),columns=np.unique(novoSpaRc_results.columns))
        for c in np.unique(novoSpaRc_pro_results.columns):
            if len(novoSpaRc_results.loc[:,c].shape) > 1:
                print (c)
                novoSpaRc_pro_results.loc[:,c] = novoSpaRc_results.loc[:,c].sum(axis=1).values
            else:
                novoSpaRc_pro_results.loc[:,c] = novoSpaRc_results.loc[:,c].values
        novoSpaRc_results = novoSpaRc_pro_results
        if self.gd is None:
            CellType = novoSpaRc_results.columns & self.gd.columns
            novoSpaRc_results = novoSpaRc_results[CellType]
        novoSpaRc_results = (novoSpaRc_results.T/novoSpaRc_results.sum(axis=1)).T
        novoSpaRc_results = novoSpaRc_results.fillna(0)
        novoSpaRc_results.to_csv(self.outdir + '/novoSpaRc_CellType_Proportion.txt')     
        
    def SpaOTsc(self):
        from spaotsc import SpaOTsc
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
        scrna_annotation = self.scrna_annotation[self.annotatetype]
        SpaOTsc_results=pd.DataFrame(np.zeros((SpaOTsc_map.shape[1],len(np.unique(scrna_annotation)))),columns=np.unique(scrna_annotation))
        for c in np.unique(scrna_annotation):
            SpaOTsc_results.loc[:,c] =  SpaOTsc_map[np.where(scrna_annotation == c)[0],:].mean(axis=0)
        mapperdict = dict(zip(self.scrna_annotation[self.annotatetype],self.scrna_annotation['celltype']))
        SpaOTsc_results.columns = [mapperdict[c] for c in SpaOTsc_results.columns]
        SpaOTsc_pro_results = pd.DataFrame(np.zeros((len(SpaOTsc_results.index), len(np.unique(SpaOTsc_results.columns)))),columns=np.unique(SpaOTsc_results.columns))
        for c in np.unique(SpaOTsc_pro_results.columns):
            if len(SpaOTsc_results.loc[:,c].shape) > 1:
                print (c)
                SpaOTsc_pro_results.loc[:,c] = SpaOTsc_results.loc[:,c].sum(axis=1).values
            else:
                SpaOTsc_pro_results.loc[:,c] = SpaOTsc_results.loc[:,c].values
        SpaOTsc_results = SpaOTsc_pro_results
        if self.gd is None:
            CellType = SpaOTsc_results.columns & self.gd.columns
            SpaOTsc_results = SpaOTsc_results[CellType]
        SpaOTsc_results = (SpaOTsc_results.T/SpaOTsc_results.sum(axis=1)).T
        SpaOTsc_results = SpaOTsc_results.fillna(0)
        SpaOTsc_results.to_csv(self.outdir + '/SpaOTsc_CellType_proportion.txt')

    def Tangram(self,):
        sys.path.append("FigureData/Tangram-master")
        import mapping.utils
        import mapping.mapping_optimizer
        import mapping.plot_utils
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
        scrna_annotation = self.scrna_annotation
        adata.obs = scrna_annotation
    
        df_classes = mapping.utils.one_hot_encoding(adata.obs[self.annotatetype])
        prob_assign = mapping.utils.transfer_annotations_prob_filter(output_all, F_out_all, df_classes)
        prob_assign.to_csv(self.outdir + '/Tangram_alignment.txt')
        Tangram_results = pd.read_csv(self.outdir + '/Tangram_alignment.txt',index_col=0)
        mapperdict = dict(zip(self.scrna_annotation[self.annotatetype],self.scrna_annotation['celltype']))
        Tangram_results.columns = [mapperdict[c] for c in Tangram_results.columns]
        Tangram_pro_results = pd.DataFrame(np.zeros((len(Tangram_results.index), len(np.unique(Tangram_results.columns)))),columns=np.unique(Tangram_results.columns))
        for c in np.unique(Tangram_pro_results.columns):
            if len(Tangram_results.loc[:,c].shape) > 1:
                print (c)
                Tangram_pro_results.loc[:,c] = Tangram_results.loc[:,c].sum(axis=1).values
            else:
                Tangram_pro_results.loc[:,c] = Tangram_results.loc[:,c].values
        Tangram_results = Tangram_pro_results
        if self.gd is None:
            CellType = Tangram_results.columns & self.gd.columns
            Tangram_results = Tangram_results[CellType]
        Tangram_results = (Tangram_results.T/Tangram_results.sum(axis=1)).T
        Tangram_results = Tangram_results.fillna(0)
        Tangram_results.to_csv(self.outdir + '/Tangram_CellType_proportion.txt')
    
    def Seurat(self):
        rna_df = self.RNA_path
        spatial_df = self.Spatial_path
        scrna_meta = self.scrna_annotation
        print ('we use seurat to predict')
        os.system('Rscript Benchmarking/Seurat_CellAssigment.r ' + spatial_df + ' ' + rna_df + ' ' + self.scrna_annotationfiles + ' ' + self.annotatetype + ' ' + self.outdir)
        print ('Rscript Benchmarking/Seurat_CellAssigment.r ' + spatial_df + ' ' + rna_df + ' ' + self.scrna_annotationfiles + ' ' + self.annotatetype + ' ' + self.outdir)
        #subprocess.run(['Rscript','Benchmarking/Seurat_CellAssigment.r',spatial_df,rna_df,self.scrna_annotationfiles,self.annotatetype,self.outdir])
        seurat_results = pd.read_csv(self.outdir + 'Seurat_alignment.txt', index_col=0)
        seurat_results = seurat_results.iloc[:,1:-1]
        Cols = seurat_results.columns
        used_ind = [(x.split('score.')[1]) for x in Cols]
        seurat_results.columns = used_ind
        seurat_results.index = np.arange(len(seurat_results))
        mapperdict = dict(zip(self.scrna_annotation[self.annotatetype],self.scrna_annotation['celltype']))
        print (mapperdict)
        print (Cols)
        print (seurat_results.columns)
        seurat_results.columns = [mapperdict[c] for c in seurat_results.columns]
        seurat_pro_results = pd.DataFrame(np.zeros((len(seurat_results.index), len(np.unique(seurat_results.columns)))),columns=np.unique(seurat_results.columns))
        for c in np.unique(seurat_pro_results.columns):
            if len(seurat_results.loc[:,c].shape) > 1:
                print (c)
                seurat_pro_results.loc[:,c] = seurat_results.loc[:,c].sum(axis=1).values
            else:
                seurat_pro_results.loc[:,c] = seurat_results.loc[:,c].values
        seurat_results = seurat_pro_results
        if self.gd is None:
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


