# basic imports
import pandas as pd
import sys
import numpy as np
import pandas as pd
import scanpy as sc

# add `Tangram` to path
import tangram as tg

sc_file_path = sys.argv[1]
spatial_file_path = sys.argv[2]
celltype_key = sys.argv[3]
output_file_path = sys.argv[4]

ad_sc = sc.read_h5ad(sc_file_path)
ad_sp = sc.read_h5ad(spatial_file_path)

# use raw count both of scrna and spatial
sc.pp.normalize_total(ad_sc)
celltype_counts = ad_sc.obs[celltype_key].value_counts()
celltype_drop = celltype_counts.index[celltype_counts < 2]
print(f'Drop celltype {list(celltype_drop)} contain less 2 sample')
ad_sc = ad_sc[~ad_sc.obs[celltype_key].isin(celltype_drop),].copy()
sc.tl.rank_genes_groups(ad_sc, groupby=celltype_key, use_raw=False)
markers_df = pd.DataFrame(ad_sc.uns["rank_genes_groups"]["names"]).iloc[0:200, :]
print(markers_df)
genes_sc = np.unique(markers_df.melt().value.values)
print(genes_sc)
genes_st = ad_sp.var_names.values
genes = list(set(genes_sc).intersection(set(genes_st)))

tg.pp_adatas(ad_sc, ad_sp, genes=genes)

ad_map = tg.map_cells_to_space(
                   ad_sc,
                   ad_sp,
                   mode='clusters',
                   cluster_label=celltype_key)

tg.project_cell_annotations(ad_map, ad_sp, annotation=celltype_key)

celltype_density = ad_sp.obsm['tangram_ct_pred']
celltype_density = (celltype_density.T/celltype_density.sum(axis=1)).T

celltype_density.to_csv(output_file_path)
