library(RCTD)
library(Matrix)
library(data.table)
library(Seurat)
library(SeuratDisk)

args<-commandArgs(T)
snrna_path = args[1]
spatial_path = args[2]
celltype_final = args[3]
output_path = args[4]

sc_obj <- LoadH5Seurat(snrna_path)
diff_list <- list()
c = 0
for (i in seq_along(unique(sc_obj@meta.data[,celltype_final]))){
    if(sum(sc_obj@meta.data[,celltype_final] == unique(sc_obj@meta.data[,celltype_final])[i]) < 25){
        c = c+1
        diff_list[[c]] <- unique(sc_obj@meta.data[,celltype_final])[i]
        print(unique(sc_obj@meta.data[,celltype_final])[i])
    }
}
for (i in diff_list){
        sc_obj = sc_obj[,sc_obj@meta.data[,celltype_final]!=i]
    }
sc_obj@meta.data[,celltype_final] <- as.factor(as.character(sc_obj@meta.data[,celltype_final]))
### Load in/preprocess your data, this might vary based on your file type
print('prepare data')
counts <- data.frame(sc_obj@assays$RNA@counts)
colnames(counts) <- colnames(sc_obj)
meta_data <- data.frame(sc_obj@meta.data[,celltype_final])
cell_types <- meta_data[,1]
names(cell_types) <- rownames(sc_obj@meta.data)
cell_types <- as.factor(cell_types)
nUMI_df <- data.frame(colSums(sc_obj@assays$RNA@counts))
nUMI <- nUMI_df$colSums.sc_obj.assays.RNA
names(nUMI) <- rownames(nUMI_df)

### Create the Reference object
reference <- Reference(counts, cell_types, nUMI)
#> Warning in Reference(counts, cell_types, nUMI): Reference: nUMI does not match colSums of counts. If this is unintended, please correct this discrepancy. If this
#>             is intended, there is no problem.
spatial_obj <- LoadH5Seurat(spatial_path)
coords <- data.frame(colnames(spatial_obj))
colnames(coords) <- 'barcodes'
coords$xcoord <- seq_along(colnames(spatial_obj))
coords$ycoord <- seq_along(colnames(spatial_obj))
counts <- data.frame(spatial_obj@assays$RNA@counts) # load in counts matrix
colnames(counts) <- colnames(spatial_obj)
coords <- data.frame(colnames(spatial_obj))
colnames(coords) <- 'barcodes'
coords$xcoord <- seq_along(colnames(spatial_obj))
coords$ycoord <- seq_along(colnames(spatial_obj))
rownames(coords) <- coords$barcodes; coords$barcodes <- NULL # Move barcodes to rownames
nUMI <- colSums(counts) # In this case, total counts per pixel is nUMI

### Create SpatialRNA object
puck <- SpatialRNA(coords, counts, nUMI)
myRCTD <- create.RCTD(puck, reference, max_cores = 8)
myRCTD <- run.RCTD(myRCTD, doublet_mode = 'full')
results <- myRCTD@results
# normalize the cell type proportions to sum to 1.
norm_weights = sweep(results$weights, 1, rowSums(results$weights), '/') 
cell_type_names <- myRCTD@cell_type_info$info[[2]] #list of cell type names
spatialRNA <- myRCTD@spatialRNA
write.csv(norm_weights, output_path)
