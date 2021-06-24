# Benchmarking methods for integrating spatial and single-cell transcriptomics data

<img width="703" alt="Pipeline" src="https://user-images.githubusercontent.com/44384930/121383040-ba5fd300-c979-11eb-91ec-af017486f3c0.png">

<font size="6"><strong>Implementation description</strong></font size="6">

We collected 14 paired datasetsdesigned a pipeline to 1) systematically evaluate the accuracy of these integration methods for predicting the RNA spatial distribution. 2) Then we down-sampled the spatial transcriptomics data of five datasets to test the performance of the integration methods for datasets with sparse expression matrices. 3) Beyond assessment of the spatial distribution of RNA transcripts, we also tested the performance of Tangram, Seurat, SpaOTsc, and novoSpaRc for assigning cell locations.

Dependencies and requirements

Before you run the pipeline, please make sure that you have installed and python3, R(3.6.1) and all the seven packages(gimVI, SpaGE, Tangram, Seurat, SpaOTsc, LIGER, novoSpaRc) :
1. Before the installation of these packages, please install Miniconda to manage all needed software and dependencies. You can download Miniconda from https://conda.io/miniconda.html.
2. Download SpatialBenchmarking.zip from https://github.com/QuKunLab/SpatialBenchmarking. Unzipping this package and you will see Benchmarkingenvironment.yml and Config.env.sh located in its folder.
3. Build isolated environment for SpatialBenchmarking: 
conda env create -f Benchmarkingenvironment.yml
4. Activate Benchmarking environment:
conda activate Benchmarking
5. sh Config.env.sh
6. Enter R and install required packages by command : install.packages(c('vctrs','rlang','htmlwidgets'))

The package has been tested on Linux system (CentOS) and should work in any valid python environment. Installation of Benchmarking may take about 7-15 minutes to install the dependencies.

Tutorial

You can run the jupyter notebook of CellAssignment.ipynb and PredictGenes.ipynb to reproduce the results of figure2&4 in our paper.

If you want to analysis your own data, the doc/Tutorial.ipynb is a example showing how to use them to predict new spatial gene patterns and cell locations.

For more details, please see the SpatialGenes.py & CellAssigment.py in Benchmarking directory.

Datasets

All datasets used are publicly available data, for convenience datasets can be downloaded from: 
https://drive.google.com/drive/folders/1pHmE9cg_tMcouV1LFJFtbyBJNp7oQo9J?usp=sharing.


