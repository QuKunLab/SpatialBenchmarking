import scanpy as sc
import numpy as np
import pandas as pd

import scvi
from scvi.model import CondSCVI, DestVI

import sys
import os


scrna_path = sys.argv[1]
spatial_path = sys.argv[2]
celltype_key = sys.argv[3]
output_path = sys.argv[4]

sc_adata = sc.read_h5ad(scrna_path)
st_adata = sc.read_h5ad(spatial_path)

# filter genes to be the same on the spatial data
intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
st_adata = st_adata[:, intersect].copy()
sc_adata = sc_adata[:, intersect].copy()
G = len(intersect)

# let us filter some genes
G = 7000
sc.pp.filter_genes(sc_adata, min_counts=10)

sc_adata.layers["counts"] = sc_adata.X.copy()

sc.pp.highly_variable_genes(
    sc_adata,
    n_top_genes=G,
    subset=True,
    layer="counts",
    flavor="seurat_v3"
)

sc.pp.normalize_total(sc_adata, target_sum=10e4)
sc.pp.log1p(sc_adata)
sc_adata.raw = sc_adata
st_adata.layers["counts"] = st_adata.X.copy()
sc.pp.normalize_total(st_adata, target_sum=10e4)
sc.pp.log1p(st_adata)
st_adata.raw = st_adata
# filter genes to be the same on the spatial data
intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
st_adata = st_adata[:, intersect].copy()
sc_adata = sc_adata[:, intersect].copy()
G = len(intersect)
scvi.data.setup_anndata(sc_adata, layer="counts", labels_key=celltype_key)
sc_model = CondSCVI(sc_adata, weight_obs=True)
sc_model.train(max_epochs=250,lr=0.001)
sc_model.history["elbo_train"].plot()
scvi.data.setup_anndata(st_adata, layer="counts")
st_model = DestVI.from_rna_model(st_adata, sc_model)
st_model.train(max_epochs=2500)
st_model.history["elbo_train"].plot()
st_model.get_proportions().to_csv(output_path + '/DestVI_result.txt')
