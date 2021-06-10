#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats as st
import seaborn as sns
import matplotlib as mpl 
import matplotlib.pyplot as plt
import os



def cal_ssim(im1,im2,M):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
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




def scale_max(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content/content.max()
        result = pd.concat([result, content],axis=1)
    return result



def scale_z_score(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = stats.zscore(content)
        content = pd.DataFrame(content,columns=[label])
        result = pd.concat([result, content],axis=1)
    return result




def scale_plus(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content/content.sum()
        result = pd.concat([result,content],axis=1)
    return result



class count:
    def __init__(self, raw_count_path, impute_count_path, tool, outdir, metric):
        self.raw_count = pd.read_csv(raw_count_path, header=0, sep="\t")
        self.impute_count = pd.read_csv(impute_count_path, header=0, index_col=0)
        self.impute_count = self.impute_count.fillna(1e-20)
        self.tool = tool
        self.outdir = outdir
        self.metric = metric
        
    def ssim(self, raw, impute, scale = 'scale_max'):
        if scale == 'scale_max':
            raw = scale_max(raw)
            impute = scale_max(impute)
        else:
            print ('Please note you do not scale data by max')
        if raw.shape[1] == impute.shape[1]:
            result = pd.DataFrame()
            for label in raw.columns:
                raw_col =  raw.loc[:,label]
                impute_col = impute.loc[:,label]
                
                M = [raw_col.max(),impute_col.max()][raw_col.max()>impute_col.max()]
                raw_col_2 = np.array(raw_col)
                raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0],1)
                
                impute_col_2 = np.array(impute_col)
                impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0],1)
                
                ssim = cal_ssim(raw_col_2,impute_col_2,M)
                
                ssim_df = pd.DataFrame(ssim, index=["SSIM"],columns=[label])
                result = pd.concat([result, ssim_df],axis=1)
            return result
        else:
            print("columns error")
            
    def pearsonr(self, raw, impute, scale = None):
        if raw.shape[1] == impute.shape[1]:
            result = pd.DataFrame()
            for label in raw.columns:
                raw_col =  raw.loc[:,label]
                impute_col = impute.loc[:,label]
                pearsonr, _ = st.pearsonr(raw_col,impute_col)
                pearson_df = pd.DataFrame(pearsonr, index=["Pearson"],columns=[label])
                result = pd.concat([result, pearson_df],axis=1)
            return result
        
    def JS(self, raw, impute, scale = 'scale_plus'):
        if scale == 'scale_plus':
            raw = scale_plus(raw)
            impute = scale_plus(impute)
        else:
            print ('Please note you do not scale data by plus')    
        if raw.shape[1] == impute.shape[1]:
            result = pd.DataFrame()
            for label in raw.columns:
                raw_col =  raw.loc[:,label]
                impute_col = impute.loc[:,label]
                
                M = (raw_col + impute_col)/2
                KL = 0.5*st.entropy(raw_col,M)+0.5*st.entropy(impute_col,M)
                KL_df = pd.DataFrame(KL, index=["JS"],columns=[label])
                
                
                result = pd.concat([result, KL_df],axis=1)
            return result
        
    def RMSE(self, raw, impute, scale = 'zscore'):
        if scale == 'zscore':
            raw = scale_z_score(raw)
            impute = scale_plus(impute)
        else:
            print ('Please note you do not scale data by zscore')
        if raw.shape[1] == impute.shape[1]:
            result = pd.DataFrame()
            for label in raw.columns:
                raw_col =  raw.loc[:,label]
                impute_col = impute.loc[:,label]
                
                RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())
                RMSE_df = pd.DataFrame(RMSE, index=["RMSE"],columns=[label])
                
                result = pd.concat([result, RMSE_df],axis=1)
            return result
                
        
    def compute_all(self):
        raw = self.raw_count
        impute = self.impute_count
        tool = self.tool
        outdir = self.outdir
        SSIM = self.ssim(raw,impute)
        Pearson = self.pearsonr(raw, impute)
        JS = self.JS(raw, impute)
        RMSE = self.RMSE(raw, impute)
        
        result_all = pd.concat([Pearson, SSIM, RMSE, JS],axis=0)
        
        if not os.path.exists(outdir):
            print ('This is an Error : No impute file folder')
        result_all.T.to_csv(outdir + "metrics_"+tool+".csv",header=1, index=1)
        self.accuracy = result_all
        return result_all
        
    def plot_boxplot(self,tools,OutPdf):
        font = {'family':'DejaVu Sans','weight':'normal','size':15}
        plt.figure(figsize=(18,16), dpi= 80)
        result = pd.DataFrame()
        Method = self.metric
        for tool in tools:
            result_metrics = self.accuracy.T
            result_metrics['tool'] = tool
            result = pd.concat([result, result_metrics],axis=0)
        
        n = 221
        for method in Method:
            ax1 = plt.subplot(n)
            ax1 = sns.boxplot(x=method, y='tool', data=result 
                         ,fliersize=1,showcaps = True,whis = 0.5 ,showfliers = False)
            ax1.set_xlabel(method)
            ax1.set_ylabel(tool)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            n = n + 1 
            plt.yticks([])
        if not os.path.exists(OutPdf):
            os.mkdir(OutPdf)
        plt.savefig(OutPdf + "/Accuracy_metrics.pdf")
        plt.show()
            
        

