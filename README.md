# Benchmarking methods for integrating spatial and single-cell transcriptomics data

<img width="703" alt="Pipeline" src="https://user-images.githubusercontent.com/44384930/121383040-ba5fd300-c979-11eb-91ec-af017486f3c0.png">

Implementation description

We collected 14 paired datasetsdesigned a pipeline to 1) systematically evaluate the accuracy of these integration methods for predicting the RNA spatial distribution. 2) Then we down-sampled the spatial transcriptomics data of five datasets to test the performance of the integration methods for datasets with sparse expression matrices. 3) Beyond assessment of the spatial distribution of RNA transcripts, we also tested the performance of Tangram, Seurat, SpaOTsc, and novoSpaRc for assigning cell locations.

Dependencies and requirements

Before you run the pipeline, please make sure that you have installed all the seven packages. (gimVI, SpaGE, Tangram, Seurat, SpaOTsc, LIGER, novoSpaRc) and you can run the jupyter notebook of CellAssignment.ipynb and PredictGenes.ipynb to reproduce the results and figures in our paper.

Tutorial

The tutorial notebook is a step-by-step example showing how to use them to predict new spatial gene patterns and cell locations.

For more details, please see the SpatialGenes.py & CellAssigment.py in Benchmarking direction.

Datasets

All datasets used are publicly available data, for convenience datasets can be downloaded from: 
https://drive.google.com/drive/folders/1pHmE9cg_tMcouV1LFJFtbyBJNp7oQo9J?usp=sharing.


