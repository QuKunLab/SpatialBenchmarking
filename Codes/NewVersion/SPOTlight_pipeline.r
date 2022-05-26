library(Matrix)
library(data.table)
library(Seurat)
library(dplyr)
library(SPOTlight)
library(igraph)
library(RColorBrewer)
library(SeuratDisk)
library(future)
# check the current active plan
plan("multiprocess", workers = 8)
options(future.globals.maxSize= 3*1024*1024^2)

args<-commandArgs(T)
scrna_path = args[1]
spatial_path = args[2]
celltype_final = args[3]
output_path = args[4]

sc <- LoadH5Seurat(scrna_path)
st <- LoadH5Seurat(spatial_path)

set.seed(123)
sc <- Seurat::SCTransform(sc, verbose = FALSE)

Idents(sc) <- sc@meta.data[,celltype_final]

cluster_markers_all <- FindAllMarkers(object = sc, 
                                              assay = "SCT",
                                              slot = "data",
                                              verbose = TRUE, 
                                              only.pos = TRUE)

spotlight_ls <- spotlight_deconvolution(
  se_sc = sc,
  counts_spatial = st@assays$RNA@counts,
  clust_vr = celltype_final, # Variable in sc_seu containing the cell-type annotation
  cluster_markers = cluster_markers_all, # Dataframe with the marker genes
  cl_n = 100, # number of cells per cell type to use
  hvg = 3000, # Number of HVG to use
  ntop = NULL, # How many of the marker genes to use (by default all)
  transf = "uv", # Perform unit-variance scaling per cell and spot prior to factorzation and NLS
  method = "nsNMF", # Factorization method
  min_cont = 0 # Remove those cells contributing to a spot below a certain threshold 
  )
decon_mtrx <- spotlight_ls[[2]]
write.csv(decon_mtrx[,which(colnames(decon_mtrx) != "res_ss")], paste0(output_path, '/SPOTlight_result.txt'))
