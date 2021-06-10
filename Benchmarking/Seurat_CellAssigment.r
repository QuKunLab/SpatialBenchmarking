library(RcppCNPy)
library(Seurat)
library(dplyr)
Args<-commandArgs()
Spatialfile = Args[6]
RNAfile = Args[7]
scrna_meta = Args[8]
annotatetype = Args[9]
outdir = Args[10]
combined_data = read.csv(Spatialfile,sep='\t')
rownames(combined_data) <- seq_len(nrow(combined_data))
combined_obj <- CreateSeuratObject(t(combined_data))
print ('We are running R script')
rna <- read.csv(RNAfile,sep='\t',row.names=1)
rna_obj <- CreateSeuratObject(rna)

features <- intersect(rownames(rna_obj),rownames(combined_obj))
rna_celltype = read.csv(scrna_meta,sep='\t',row.names=1)
rna_obj <- AddMetaData(rna_obj,metadata = rna_celltype)

anchors <- FindTransferAnchors(reference=rna_obj, query = combined_obj, features = features, dims = 1:30, k.filter = NA,reduction='cca',reference.assay = 'RNA',query.assay = 'RNA')
predictions <- TransferData(anchorset = anchors, refdata = rna_obj$annotatetype, outdir = outdir, dims = 1:30,weight.reduction = 'cca')
print ('predictions result will be loaded')
write.csv(predictions, paste0(outdir,'/Seurat_alignment.txt'))
