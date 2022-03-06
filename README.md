# Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type deconvolution
![Figure1](https://user-images.githubusercontent.com/100823826/156927073-de21af45-8eee-476a-a11d-3b67c968eb70.jpg)

__Implementation description__

  We collected 45 paired datasets and 32 simulated datasets and designed a pipeline to 1) systematically evaluate the accuracy of eight integration methods for predicting the RNA spatial distribution. 2) Then we down-sampled the spatial transcriptomics data of five datasets to test the performance of the integration methods for datasets with sparse expression matrices. 3) Beyond assessment of the spatial distribution of RNA transcripts, we also tested the performance of ten integration methods for celltypes deconvolution.

  We provide example guidance to help researchers select optimal integration methods for working with their datasets:
  the [doc/Tutorial.pdf](https://github.com/QuKunLab/SpatialBenchmarking/blob/main/doc/Tutorial.pdf) is an example showing how to use them to predict new spatial gene patterns
and cell locations.


__Dependencies and requirements for Predicting undetected transcripts__

 Before you run the pipeline, please make sure that you have installed and python3, R(3.6.1) and all the eight packages(gimVI, SpaGE, Tangram, Seurat, SpaOTsc, LIGER, novoSpaRc, stPlus) :
1. Before the installation of these packages, please install Miniconda to manage all needed software and dependencies. You can download Miniconda from https://conda.io/miniconda.html.
2. Download SpatialBenchmarking.zip from https://github.com/QuKunLab/SpatialBenchmarking. Unzipping this package and you will see Benchmarkingenvironment.yml and Config.env.sh located in its folder.
3. Build isolated environment for SpatialBenchmarking: 
`conda env create -f Benchmarkingenvironment.yml`
4. Activate Benchmarking environment:
`conda activate Benchmarking`
5. `sh Config.env.sh`
6. Enter R and install required packages by command : `install.packages(c('vctrs','rlang','htmlwidgets'))`

__Dependencies and requirements for Predicting celltypes deconvolution__

 Before you run the pipeline, please make sure that you have installed and python3, R and all the ten packages(Cell2location, RCTD, SpatialDWLS, Stereoscope, SPOTlight, Tangram, Seurat, STRIDE, DestVI, DSTG)

The package has been tested on Linux system (CentOS) and should work in any valid python environment. Installation of Benchmarking may take about 7-15 minutes to install the dependencies.

__Tutorial__

  If you want to analysis your own data, the [doc/Tutorial.ipynb](https://github.com/QuKunLab/SpatialBenchmarking/blob/main/doc/Tutorial.ipynb) is an example showing how to use them to predict new spatial gene patterns and cell locations.

  You also can run the jupyter notebook of `BLAST_GenePrediction.ipynb` and `BLAST_CelltypeDeconvolution.ipynb` to reproduce the results of figure2&4 in our paper.
  
  For more details, please see the `SpatialGenes.py` & `Deconvolution.py` in Benchmarking directory.

__Datasets__

  All datasets used are publicly available data, for convenience datasets can be downloaded from: 
https://drive.google.com/drive/folders/1pHmE9cg_tMcouV1LFJFtbyBJNp7oQo9J?usp=sharing.


