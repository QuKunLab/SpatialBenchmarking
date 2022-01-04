library(Seurat)
library(dplyr)
library(SeuratDisk)

args<-commandArgs(T)
scrna_path = args[1]
spatial_path = args[2]
celltype_final = args[3]
output_path = args[4]

sc_rna <- LoadH5Seurat(scrna_path)
sc_rna <- SCTransform(sc_rna)
spatial <- LoadH5Seurat(spatial_path)
spatial <- SCTransform(spatial)
anchors <- FindTransferAnchors(reference=sc_rna, query = spatial, dims = 1:30, normalization.method = 'SCT')
predictions <- TransferData(anchorset = anchors, refdata = sc_rna@meta.data[,celltype_final], dims = 1:30)
write.csv(predictions, output_path)