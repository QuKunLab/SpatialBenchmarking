import sys
import os

class Deconvolutions:
    def __init__(self, RNA_file = None, RNA_h5ad = None, RNA_h5Seurat = None, Spatial_file = None, Spatial_h5ad = None, Spatial_h5Seurat = None, celltype_key = None, celltype_file = None, my_python_path = None, output_path = None):
        """
            @author: wen zhang
            This function integrates spatial and scRNA-seq data to predictes the celltype deconvolution of the spots.
            
            A minimal example usage:
            Assume we have (1) scRNA-seq data file named RNA_h5ad or RNA_h5Seurat
            (2) spatial transcriptomics data file named Spatial_h5ad or Spatial_h5Seurat
            (3) celltype annotataion title in scRNA-seq data file
            
            >>> import Benchmarking.DeconvolutionSpot as DeconvolutionSpot
            >>> test = DeconvolutionSpot.Deconvolutions(RNA_file, RNA_h5ad, RNA_h5Seurat, Spatial_file, Spatial_h5ad, Spatial_h5Seurat, celltype_key, celltype_file, output_path)
            >>> Methods = ['Cell2location','SpatialDWLS','RCTD','STRIDE','Stereoscope','Tangram','DestVI', 'Seurat', 'SPOTlight', 'DSTG']
            >>> Result = test.Dencon(Methods)
            
            Parameters
            -------
            
            RNA_file : str
            scRNA-seq data count file.
            
            RNA_h5ad : str
            scRNA-seq data file with h5ad format.
            
            RNA_h5Seurat : str
            scRNA-seq data file with h5Seurat format.
            
            Spatial_file : str
            Spatial data count file.
            
            Spatial_h5ad : str
            Spatial data file with h5ad format.
            
            Spatial_h5Seurat : str
            Spatial data file with h5Seurat format.
            
            celltype_key : str
            celltype annotataion title in scRNA-seq data h5ad file or h5Seurat file
            
            celltype_file : str
            celltype annotataion file
            
            my_python_path : str
            which python path used for Cell2location
            
            output_path : str
            Outfile path
            
            """
        
        self.RNA_file = RNA_file
        self.RNA_h5ad = RNA_h5ad
        self.RNA_h5Seurat = RNA_h5Seurat
        self.Spatial_file = Spatial_file
        self.Spatial_h5ad = Spatial_h5ad
        self.Spatial_h5Seurat = Spatial_h5Seurat
        self.celltype_key = celltype_key
        self.celltype_file = celltype_file
        self.my_python_path = my_python_path
        self.output_path = output_path
    
    def Dencon(self, need_tools):
        if "Cell2location" in need_tools:
            RNA_h5ad = self.RNA_h5ad
            Spatial_h5ad = self.Spatial_h5ad
            celltype_key = self.celltype_key
            output_path = self.output_path
            python_path = self.python_path
            os.system('python Codes/Deconvolution/Cell2location_pipeline.py ' + RNA_h5ad + ' ' + Spatial_h5ad + ' ' + celltype_key + ' ' + output_path + ' ' + python_path)

        if "SpatialDWLS" in need_tools:
            RNA_h5Seurat = self.RNA_h5Seurat
            Spatial_h5Seurat = self.Spatial_h5Seurat
            celltype_key = self.celltype_key
            output_path = self.output_path
            my_python_path = self.my_python_path
            os.system('Rscript Codes/Deconvolution/SpatialDWLS_pipeline.r ' + RNA_h5Seurat + ' ' + Spatial_h5Seurat + ' ' + celltype_key + ' ' + output_path + ' ' + my_python_path)

        if "RCTD" in need_tools:
            RNA_h5Seurat = self.RNA_h5Seurat
            Spatial_h5Seurat = self.Spatial_h5Seurat
            celltype_key = self.celltype_key
            output_path = self.output_path
            os.system('Rscript Codes/Deconvolution/RCTD_pipeline.r ' + RNA_h5Seurat + ' ' + Spatial_h5Seurat + ' ' + celltype_key + ' ' + output_path)

        if "STRIDE" in need_tools:
            RNA_file = self.RNA_file
            Spatial_file = self.Spatial_file
            celltype_file = self.celltype_file
            output_path = self.output_path
            os.system('sh Codes/Deconvolution/STRIDE_pipeline.sh ' + RNA_file + ' ' + Spatial_file + ' ' + celltype_file + ' ' + output_path)

        if "Stereoscope" in need_tools:
            RNA_h5ad = self.RNA_h5ad
            Spatial_h5ad = self.Spatial_h5ad
            celltype_key = self.celltype_key
            output_path = self.output_path
            os.system('python Codes/Deconvolution/Stereoscope_pipeline.py ' + RNA_h5ad + ' ' + Spatial_h5ad + ' ' + celltype_key + ' ' + output_path)

        if "Tangram" in need_tools:
            RNA_h5ad = self.RNA_h5ad
            Spatial_h5ad = self.Spatial_h5ad
            celltype_key = self.celltype_key
            output_path = self.output_path
            os.system('python Codes/Deconvolution/Tangram_pipeline.py ' + RNA_h5ad + ' ' + Spatial_h5ad + ' ' + celltype_key + ' ' + output_path)

        if "DestVI" in need_tools:
            RNA_h5ad = self.RNA_h5ad
            Spatial_h5ad = self.Spatial_h5ad
            celltype_key = self.celltype_key
            output_path = self.output_path
            os.system('python Codes/Deconvolution/DestVI_pipeline.py ' + RNA_h5ad + ' ' + Spatial_h5ad + ' ' + celltype_key + ' ' + output_path)

        if "Seurat" in need_tools:
            RNA_h5Seurat = self.RNA_h5Seurat
            Spatial_h5Seurat = self.Spatial_h5Seurat
            celltype_key = self.celltype_key
            output_path = self.output_path
            os.system('Rscript Codes/Deconvolution/Seurat_pipeline.r ' + RNA_h5Seurat + ' ' + Spatial_h5Seurat + ' ' + celltype_key + ' ' + output_path)

        if "SPOTlight" in need_tools:
            RNA_h5Seurat = self.RNA_h5Seurat
            Spatial_h5Seurat = self.Spatial_h5Seurat
            celltype_key = self.celltype_key
            output_path = self.output_path
            os.system('Rscript Codes/Deconvolution/SPOTlight_pipeline.r ' + RNA_h5Seurat + ' ' + Spatial_h5Seurat + ' ' + celltype_key + ' ' + output_path)

        if "DSTG" in need_tools:
            RNA_h5Seurat = self.RNA_h5Seurat
            Spatial_h5Seurat = self.Spatial_h5Seurat
            celltype_key = self.celltype_key
            output_path = self.output_path
            os.system('Rscript Codes/Deconvolution/DSTG_pipeline.r ' + RNA_h5Seurat + ' ' + Spatial_h5Seurat + ' ' + celltype_key + ' ' + output_path)



