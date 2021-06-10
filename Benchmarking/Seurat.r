library(Seurat)
library(ggplot2)
library('Matrix')
Args<-commandArgs()
RNAfile = Args[6]

RNA <- read.table(RNAfile,sep = '\t',header = TRUE,row.names = 1,quote = "")
RNA <- CreateSeuratObject(counts=RNA, project='RNA', min.cells=0, min.features=0)
Norm = Args[9]
if (Norm == 'Norm'){
    RNA <- NormalizeData(RNA)
    RNA <- FindVariableFeatures(RNA,nfeatures = 2000)
    RNA <- ScaleData(RNA)
}


Spatialfile = Args[7]
Spatial_orig <- t(read.table(Spatialfile,sep = '\t',header = TRUE,,quote = ""))
Genes = scan(Spatialfile,what = 'character',nlines=1)
rownames(Spatial_orig) = Genes
colnames(Spatial_orig) <- paste0(colnames(Spatial_orig), 1:ncol(Spatial_orig))


feature.remove = Args[9]
features <- Args[8]
Spatial = Spatial_orig[features,]
print (dim(Spatial))
Spatial <- CreateSeuratObject(counts=Spatial, project='Spatial', min.cells=0, min.features=0)
if (Norm == 'Norm'){
    Spatial <- NormalizeData(Spatial)
    Spatial <- FindVariableFeatures(Spatial,nfeatures = 2000)
    Spatial <- ScaleData(Spatial)
}
DN = 30
if ((legth(features)-1)<30){
        DN = (length(features))
}
anchors <- FindTransferAnchors(reference = RNA,query = Spatial,features = features,reduction = 'cca',reference.assay = 'RNA',query.assay = 'RNA', k.filter = NA, dims = 1:DN)
refdata <- GetAssayData(object = RNA,assay = 'RNA',slot = 'data')
imputation <- TransferData(anchorset = anchors,refdata = refdata,weight.reduction = 'pca',dims = 1:DN)
options(warn = -1)
Imp_New_genes = as.data.frame(imputation@data)[feature.remove,]
return Imp_New_genes
warnings('off')
