options (warn = -1)
library(liger)
library("plyr")
library('Matrix')
Args<-commandArgs()
RNAfile = Args[6]
rna = read.csv(RNAfile,sep = '\t',header = TRUE,row.names = 1,quote = "")
spatial_origal = as.data.frame(t(read.csv(Args[7],quote = "",sep = '\t',header = TRUE)))
Genes = scan(Args[7],what = 'character',nlines=1)
rownames(spatial_origal) = Genes

rnaUse = rna
predict <- Args[9]
features <- Args[8]
features <- strsplit(features,split=',')[[1]]
predict <- strsplit(predict,split=',')[[1]]
OutFile = Args[10]

Exp = as.data.frame(colSums(rna[features,]))
colnames(Exp) = c('SumGene')
RemoveCells = subset(Exp,SumGene==0)
UseCells = setdiff(colnames(rna),rownames(RemoveCells))
rnaUse = rna[UseCells]
spatialUse = spatial_origal[features,]
print(dim(spatialUse))
Ligerex.leaveout <- createLiger(list(SMSC_RNA = rnaUse,SMSC_FISH = spatialUse))
Ligerex.leaveout <- normalize(Ligerex.leaveout)
Ligerex.leaveout@norm.data <- Ligerex.leaveout@raw.data
Ligerex.leaveout <- selectGenes(Ligerex.leaveout)
Ligerex.leaveout <- scaleNotCenter(Ligerex.leaveout)
k = (length(Ligerex.leaveout@var.genes)-3)
if (k>20){
    k = 20
}
options (warn = -1)
Ligerex.leaveout <- optimizeALS(Ligerex.leaveout,k = k, lambda = 20)
Ligerex.leaveout <- quantile_norm(Ligerex.leaveout)
options (warn = -1)
Imputation <- imputeKNN(Ligerex.leaveout,reference = 'SMSC_RNA', queries = list('SMSC_FISH'), norm = FALSE, scale = FALSE, knn_k = 30)
Result = as.matrix(Imputation@raw.data$SMSC_FISH)[unlist(predict),]
write.table(Result,file = paste0(OutFile),sep='\t',quote=F,col.names = TRUE)


