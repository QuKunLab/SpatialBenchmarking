options (warn = -1)
library(Seurat)
library(ggplot2)
library('Matrix')
Args<-commandArgs()
RNAfile = Args[6]

RNA <- read.table(RNAfile,sep = '\t',header = TRUE,row.names = 1,quote = "")
RNA <- CreateSeuratObject(counts=RNA, project='RNA', min.cells=0, min.features=0)

Spatialfile = Args[7]
Spatial_orig <- t(read.table(Spatialfile,sep = '\t',header = TRUE,,quote = ""))
Genes = scan(Spatialfile,what = 'character',nlines=1)
rownames(Spatial_orig) = Genes
colnames(Spatial_orig) <- paste0(colnames(Spatial_orig), 1:ncol(Spatial_orig))


test_genes = Args[9]
features <- Args[8]
OutFile = Args[10]

features <- strsplit(features,split=',')[[1]]
test_genes <- strsplit(test_genes,split=',')[[1]]

Spatial = Spatial_orig[features,]
print(dim(Spatial))

Spatial <- CreateSeuratObject(counts=Spatial, project='Spatial', min.cells=0, min.features=0)

DN = 30
if ((length(features)-1)<30){
        DN = (length(features) -1)
}

options (warn = -1)
anchors <- FindTransferAnchors(reference = RNA,query = Spatial,features = features,reduction = 'cca',reference.assay = 'RNA',query.assay = 'RNA', k.filter = NA, dims = 1:DN)

refdata <- GetAssayData(object = RNA,assay = 'RNA',slot = 'data')

imputation <- TransferData(anchorset = anchors,refdata = refdata,weight.reduction = 'pca',dims = 1:DN)

options(warn = -1)
print (test_genes)

Imp_New_genes = as.data.frame(imputation@data)[test_genes,]

write.table(Imp_New_genes,file = paste0(OutFile),sep='\t',quote=F,col.names = TRUE)


warnings('off')
