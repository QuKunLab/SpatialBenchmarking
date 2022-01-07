scrna_path=$1
spatial_path=$2
celltype_key=$3
output_path=$4

dir_path=`dirname $scrna_path`
dataset=`echo $dir_path | rev | cut -d/ -f1 | rev`
celltype_path=`echo $celltype_key`
prefix='STRIDE'
echo $dataset
echo $celltype_path
STRIDE deconvolve --sc-count $scrna_path \
--sc-celltype $celltype_path \
--st-count $spatial_path \
--outdir $output_path --outprefix $prefix --normalize
