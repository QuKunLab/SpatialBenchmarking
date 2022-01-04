library(liger)
library(Seurat)
library("plyr")
library(reticulate)
np <- import("numpy")
Args<-commandArgs()
PATHDIR = Args[6]
RNAfile = paste0(Args[6],'scRNA_count.txt')
rna = read.csv(RNAfile,sep = '\t',header = TRUE,row.names = 1,quote = "")

spatial_origal = as.data.frame(t(read.csv(paste0(Args[6],'Insitu_count.txt'),quote = "",sep = '\t',header = TRUE)))
Genes = scan(paste0(Args[6],'Insitu_count.txt'),what = 'character',nlines=1)
rownames(spatial_origal) = Genes

Result = as.data.frame(array(,dim=c(dim(spatial_origal)[2],dim(spatial_origal)[1])))
colnames(Result) = Genes
rownames(Result) <- paste0('V',1:ncol(spatial_origal))
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

dir.create(Args[7])
run_imputation <- function(n){
	rnaUse = rna
	predict <- unlist(test_list[,n])
	predict <- unlist(unlist(predict))
	predict <- as.character(predict)
	print (length(predict))
	features <- unlist(train_list[,n])
	Exp = as.data.frame(colSums(rna[features,]))
	colnames(Exp) = c('SumGene')
	RemoveCells = subset(Exp,SumGene==0)
	UseCells = setdiff(colnames(rna),rownames(RemoveCells))
	rnaUse = rna[UseCells]
	spatialUse = spatial_origal[features,]
	print (dim(spatialUse))
	Ligerex.leaveout <- createLiger(list(SMSC_RNA = rnaUse,SMSC_FISH = spatialUse))
	Ligerex.leaveout <- normalize(Ligerex.leaveout)
	Ligerex.leaveout@norm.data <- Ligerex.leaveout@raw.data
	Ligerex.leaveout <- selectGenes(Ligerex.leaveout)
	Ligerex.leaveout <- scaleNotCenter(Ligerex.leaveout)
	k = 20
	if ((length(Ligerex.leaveout@var.genes))<20){
	        k = (length(Ligerex.leaveout@var.genes) - 3)
	}
	Ligerex.leaveout <- optimizeALS(Ligerex.leaveout,k = k, lambda = 20)
	Ligerex.leaveout <- quantile_norm(Ligerex.leaveout)
	a = (colnames(Ligerex.leaveout@raw.data[['SMSC_RNA']])) 
	b = (rownames(Ligerex.leaveout@H.norm)) 
	c = intersect(x=a,y=b)
	Ligerex.leaveout@raw.data[['SMSC_RNA']] = Ligerex.leaveout@raw.data[['SMSC_RNA']][,c]
	Ligerex.leaveout@norm.data[['SMSC_RNA']] = Ligerex.leaveout@norm.data[['SMSC_RNA']][,c]
	Ligerex.leaveout@scale.data[['SMSC_RNA']] = Ligerex.leaveout@scale.data[['SMSC_RNA']][c,]
	a = (colnames(Ligerex.leaveout@raw.data[['SMSC_FISH']])) 
	b = (rownames(Ligerex.leaveout@H.norm)) 
	c = intersect(x=a,y=b)
	Ligerex.leaveout@raw.data[['SMSC_FISH']] = Ligerex.leaveout@raw.data[['SMSC_FISH']][,c]
	Ligerex.leaveout@norm.data[['SMSC_FISH']] = Ligerex.leaveout@norm.data[['SMSC_FISH']][,c]
	Ligerex.leaveout@scale.data[['SMSC_FISH']] = Ligerex.leaveout@scale.data[['SMSC_FISH']][c,]

	Imputation <- imputeKNN(Ligerex.leaveout,reference = 'SMSC_RNA', queries = list('SMSC_FISH'), norm = FALSE, scale = FALSE, knn_k = 30)
	Imp_New_genes = as.data.frame(Imputation@raw.data$SMSC_FISH)[unlist(predict),]
	return (Imp_New_genes)
}
for(i in 1:10) {
        Result[unlist(test_list[,i]), colnames(run_imputation(i))] = as.matrix(run_imputation(i))
}
warnings('off')

write.table(t(Result), paste0(Args[7],'LIGER_impute.csv'), sep = ',', quote = FALSE)

