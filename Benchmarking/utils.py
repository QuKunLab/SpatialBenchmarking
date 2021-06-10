### please the vefity that you have installed the Seurat,SpaOTsc,Tangram,novoSpaRc
### please  make sure you are in SpatialBenmarking dir and have prepared the data files

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

def Simulated(spatial_rna, spatial_meta, spatial_loc, CoordinateXlable, CoordinateYlable, window, outdir):
    if os.path.exists(outdir):
        print ('The output file is in ' + outdir)
    else:
        os.mkdir(outdir)
    combined_spot = []
    combined_spot_loc = []
    window=window
    c = 0
    for x in np.arange((spatial_loc[CoordinateXlable].min()//window),spatial_loc[CoordinateXlable].max()//window+1):
        for y in np.arange((spatial_loc[CoordinateYlable].min()//window),spatial_loc[CoordinateYlable].max()//window+1):
            tmp_loc = spatial_loc[(x*window < spatial_loc[CoordinateXlable]) & (spatial_loc[CoordinateXlable] < (x+1)*window) & (y*window < spatial_loc[CoordinateYlable]) & (spatial_loc[CoordinateYlable] < (y+1)*window)]
            if len(tmp_loc) > 0:
                c += 1
                combined_spot_loc.append([x,y])
                combined_spot.append(tmp_loc.index.to_list())
            
    combined_cell_counts = pd.DataFrame([len(s) for s in combined_spot],columns=['cell_count'])
    combined_cell_counts.to_csv(outdir + '/combined_cell_counts.txt',sep='\t')
    combined_cell_counts = pd.read_csv(outdir + '/combined_cell_counts.txt',sep='\t',index_col=0)
    print ('The simulated spot has cells with ' + str(combined_cell_counts.min()[0]) + ' to ' + str(combined_cell_counts.max()[0]))
    combined_spot_loc = pd.DataFrame(combined_spot_loc, columns=['x','y'])
    combined_spot_loc.to_csv(outdir + '/combined_Locations.txt',sep='\t',index=False)

    combined_spot_exp = []
    for s in combined_spot:
        combined_spot_exp.append(spatial_rna.loc[s,:].sum(axis=0).values)
    combined_spot_exp = pd.DataFrame(combined_spot_exp, columns=spatial_rna.columns)
    combined_spot_exp.to_csv(outdir + '/combined_spatial_count.txt',sep='\t',index=False)

    combined_spot_clusters = pd.DataFrame(np.zeros((len(combined_spot_loc.index),len(np.unique(spatial_meta['celltype'])))),columns=np.unique(spatial_meta['celltype']))
    for i,c in enumerate(combined_spot):
        for clt in spatial_meta.loc[c,'celltype']:
            combined_spot_clusters.loc[i,clt] += 1
    combined_spot_clusters.to_csv(outdir + '/combined_spot_clusters.txt',sep='\t')
    print ('The simulated spot has size ' + str(combined_spot_clusters.shape[0]))


def SSIM_Calculation(im1,im2,M=1):
    im1, im2 = im1/im1.max(), im2/im2.max()
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim
def rsme(x1,x2):
    x1 = st.zscore(x1)
    x2 = st.zscore(x2)
    return mean_squared_error(x1,x2,squared=False)


def CalculateMetric(outdir,Methods,gd_celltype):
    data = []
    for Method in Methods:
        gd_results = pd.read_csv(gd_celltype,sep = '\t',index_col=0, header = 0)
        Predict_results = pd.read_csv(outdir + '/' + Method + '_CellType_proportion.txt',sep = ',', header = 0, index_col = 0)
        PCC = []
        SSIM = []
        JS = []
        RMSE = []
        CellTypeUse = Predict_results.columns & gd_results.columns
        Predict_results = Predict_results[CellTypeUse]
        gd_results = gd_results[CellTypeUse]
        gd_results = (gd_results.T/gd_results.sum(axis=1)).T
        gd_results = gd_results.fillna(0)
        print ('We Use Celltype Number ' + str(len(CellTypeUse)))
        for i in range(len(gd_results)):
            if np.max(gd_results.loc[i,:]) == 0 or np.max(Predict_results.loc[i,:]) == 0:
                PCC.append(P[0])
                SSIM.append(P[0])
                RMSE.append(1.5)
                JS.append(1)
            else:
                P = pearsonr(Predict_results.loc[i,:],gd_results.loc[i,:])
                PCC.append(P[0])
                SSIM.append(SSIM_Calculation(Predict_results.loc[i,:],gd_results.loc[i,:]))
                RMSE.append(rsme(Predict_results.loc[i,:],gd_results.loc[i,:]))
                JSD = jensenshannon(Predict_results.loc[i,:],gd_results.loc[i,:])
                JS.append(JSD**2)
        
        PCC = np.nan_to_num(PCC, nan=0)
        SSIM = np.nan_to_num(SSIM, nan=0)
        RMSE = np.nan_to_num(RMSE, nan = 1.5)
        JS = np.nan_to_num(JS, nan=1)
        
        Metric = pd.DataFrame(PCC)
        Metric.columns = ['PCC']
        Metric['SSIM'] = SSIM
        Metric['RMSE'] = RMSE
        Metric['JS'] = JS
        Metric.to_csv(outdir + '/' + Method + '_Cellmapping_metric.txt',sep = '\t') 
