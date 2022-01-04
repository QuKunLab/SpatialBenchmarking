library(Seurat)
library(ggplot2)
library(reticulate)
np <- import("numpy")
Args<-commandArgs()
PATHDIR = Args[6]
RNAfile = paste0(Args[6],'scRNA_count.txt')

RNA <- read.table(RNAfile,sep = '\t',header = TRUE,row.names = 1,quote = "")
RNA <- CreateSeuratObject(counts=RNA, project='RNA', min.cells=0, min.features=0)

Spatialfile = paste0(Args[6],'Insitu_count.txt')
Spatial_orig <- t(read.table(Spatialfile,sep = '\t',header = TRUE,,quote = ""))
Genes = scan(Spatialfile,what = 'character',nlines=1)
rownames(Spatial_orig) = Genes
colnames(Spatial_orig) <- paste0(colnames(Spatial_orig), 1:ncol(Spatial_orig))

Result = as.data.frame(array(,dim=c(dim(Spatial_orig)[2],dim(Spatial_orig)[1])))
colnames(Result) = Genes
rownames(Result) <- 1:ncol(Spatial_orig)
Result = t(Result)

train_list <- np$load(paste0(PATHDIR,'train_list.npy'), allow_pickle = TRUE)
test_list <- np$load(paste0(PATHDIR,'test_list.npy'), allow_pickle = TRUE)
trainshape = dim(train_list)
testshape = dim(test_list)
if (length(trainshape) == 2){
    train_list <- array(train_list, dim = c(trainshape[2],trainshape[1]))
    test_list <- array(test_list, dim = c(testshape[2],testshape[1]))
}
if (length(trainshape) == 1){
    train_list <- array(train_list, dim = c(1,trainshape))
    test_list <- array(test_list, dim = c(1,testshape))
}

run_imputation <- function(i) {
    genes.leaveout = unlist(train_list[,i])
    feature.remove = unlist(test_list[,i])
    print ('We Used Test Genes : ')
    print (feature.remove)
    features <- unlist(train_list[,i])
    print (length(features))
    Spatial = Spatial_orig[features,]
    print (dim(Spatial))
    Spatial <- CreateSeuratObject(counts=Spatial, project='Spatial', min.cells=0, min.features=0)
    DN = 30
    if ((length(features)-1)<30){
        DN = (length(features)-1)
    }
    anchors <- FindTransferAnchors(reference = RNA,query = Spatial,features = features,reduction = 'cca',reference.assay = 'RNA',query.assay = 'RNA', k.filter = NA, dims = 1:DN)
    refdata <- GetAssayData(object = RNA,assay = 'RNA',slot = 'data')
    print ('run Transfer')
    imputation <- TransferData(anchorset = anchors,refdata = refdata,weight.reduction = 'pca',dims = 1:DN)
    options(warn = -1)
    Imp_New_genes = as.data.frame(imputation@data)[feature.remove,]
    return (Imp_New_genes)    
}
dir.create(Args[7])
for(i in 1:10) {
    Result[unlist(test_list[,i]), ] = as.matrix(run_imputation(i))
}
write.table(t(Result), paste0(Args[7], 'Seurat_impute.csv'), sep = ',', quote = FALSE)
warnings('off')
