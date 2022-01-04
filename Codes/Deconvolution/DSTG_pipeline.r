library(SeuratDisk)

args = commandArgs(T)
scrna_path <- args[1]
spatial_path <- args[2]
celltype_key <- args[3]
output_path <- args[4]

scrna_name <- strsplit(basename(scrna_path),".h5seurat")
spatial_name <- strsplit(basename(spatial_path),".h5seurat")
print(paste0(output_path,'/',scrna_name,'_counts.rds'))

scrna <- LoadH5Seurat(scrna_path)
spatial <- LoadH5Seurat(spatial_path)

sc_counts_df <- data.frame(scrna@assays$RNA@counts)
rownames(sc_counts_df) <- rownames(scrna@assays$RNA@counts)
colnames(sc_counts_df) <- colnames(scrna@assays$RNA@counts)
sc_counts_df <- sc_counts_df[rowSums(sc_counts_df)!=0,]
sc_counts_df <- sc_counts_df[,colSums(sc_counts_df)!=0]

st_counts_df <- data.frame(spatial@assays$RNA@counts)
rownames(st_counts_df) <- rownames(spatial@assays$RNA@counts)
colnames(st_counts_df) <- colnames(spatial@assays$RNA@counts)
st_counts_df <- st_counts_df[rowSums(st_counts_df)!=0,]
st_counts_df <- st_counts_df[,colSums(st_counts_df)!=0]

overlap_genes <- intersect(rownames(st_counts_df),rownames(sc_counts_df))
st_counts_df <- st_counts_df[overlap_genes,]
sc_counts_df <- sc_counts_df[overlap_genes,]

df <- data.frame(as.character(scrna@meta.data[,celltype_key]),row.names = rownames(scrna@meta.data))
colnames(df) <- c(celltype_key)

saveRDS(sc_counts_df,paste0(output_path,'/',scrna_name,'_counts.rds'))
saveRDS(st_counts_df,paste0(output_path,'/',spatial_name,'_counts.rds'))
saveRDS(df, paste0(output_path,'/',scrna_name,'_meta.rds'))
