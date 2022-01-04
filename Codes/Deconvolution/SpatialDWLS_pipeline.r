library(Giotto)
library(SeuratDisk)

args<-commandArgs(T)
scrna_path = args[1]
spatial_path = args[2]
celltype_final = args[3]
output_path = args[4]

my_python_path = "~/miniconda3/envs/cellpymc/bin/python"
instrs = createGiottoInstructions(python_path = my_python_path)
sc <- LoadH5Seurat(scrna_path)
st <- LoadH5Seurat(spatial_path)
st_data <- createGiottoObject(
    raw_exprs = st@assays$RNA@counts,
    instructions = instrs
)
# st_data <- filterGiotto(gobject = st_data,
#                              expression_threshold = 1,
#                              gene_det_in_min_cells = 50,
#                              min_det_genes_per_cell = 1000,
#                              expression_values = c('raw'),
#                              verbose = T)
st_data <- normalizeGiotto(gobject = st_data)
st_data <- calculateHVG(gobject = st_data)
gene_metadata = fDataDT(st_data)
featgenes = gene_metadata[hvg == 'yes']$gene_ID
st_data <- runPCA(gobject = st_data, genes_to_use = featgenes, scale_unit = F)
signPCA(st_data, genes_to_use = featgenes, scale_unit = F)
st_data <- runUMAP(st_data, dimensions_to_use = 1:10)
st_data <- createNearestNetwork(gobject = st_data, dimensions_to_use = 1:10, k = 15)
st_data <- doLeidenCluster(gobject = st_data, resolution = 0.4, n_iterations = 1000)
sc_data <- createGiottoObject(
    raw_exprs = sc@assays$RNA@counts,
    instructions = instrs
)
sc_data <- normalizeGiotto(gobject = sc_data)
sc_data <- calculateHVG(gobject = sc_data)
gene_metadata = fDataDT(sc_data)
featgenes = gene_metadata[hvg == 'yes']$gene_ID
sc_data <- runPCA(gobject = sc_data, genes_to_use = featgenes, scale_unit = F)
signPCA(sc_data, genes_to_use = featgenes, scale_unit = F)
sc_data@cell_metadata$leiden_clus <- as.character(sc@meta.data[,celltype_final])
scran_markers_subclusters = findMarkers_one_vs_all(gobject = sc_data,
                                                   method = 'scran',
                                                   expression_values = 'normalized',
                                                   cluster_column = 'leiden_clus')
Sig_scran <- unique(scran_markers_subclusters$genes[which(scran_markers_subclusters$ranking <= 100)])
norm_exp<-2^(sc_data@norm_expr)-1
id<-sc_data@cell_metadata$leiden_clus
ExprSubset<-norm_exp[Sig_scran,]
Sig_exp<-NULL
for (i in unique(id)){
  Sig_exp<-cbind(Sig_exp,(apply(ExprSubset,1,function(y) mean(y[which(id==i)]))))
}
colnames(Sig_exp)<-unique(id)
st_data <- runDWLSDeconv(st_data,sign_matrix = Sig_exp, n_cell = 20)
write.csv(st_data@spatial_enrichment$DWLS, paste0(output_path, '/Cell2locations_result.txt'))
