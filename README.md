# SpatialBenchmarking
Benchmarking methods for integrating spatial and single-cell transcriptomics data
![image](https://user-images.githubusercontent.com/44384930/121382531-46252f80-c979-11eb-853f-7d8d19f3dc8d.png)

Implementation description
We collected 14 all fourteen paired datasetsdesigned a pipeline to systematically evaluate the accuracy of these integration methods for predicting the RNA spatial distribution. Then we down-sampled the spatial transcriptomics data of five datasets to test the performance of the integration methods for datasets with sparse expression matrices. Beyond assessment of the spatial distribution of RNA transcripts, we also tested the performance of Tangram, Seurat, SpaOTsc, and novoSpaRc for assigning cell locations.

Before you run the pipeline, please make sure that you have installed all the seven packages. (gimVI, SpaGE, Tangram, Seurat, SpaOTsc, LIGER, novoSpaRc) and you can run the jupyter notebook to reproduce the results and figures in our paper.

Tutorial
The tutorial notebook is a step-by-step example showing how to assess these integration methods on the spatially measured genes, and how to use them to predict new spatial gene patterns and cell locations.

Datasets
All datasets used are publicly available data, for convenience datasets can be downloaded from: 
https://drive.google.com/drive/folders/1pHmE9cg_tMcouV1LFJFtbyBJNp7oQo9J?usp=sharing.


