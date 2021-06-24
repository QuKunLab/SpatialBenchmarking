conda env create -f Benchmarkingenvironment.yml
source activate Benchmarking
cd Extenrnal/SpaOTsc
pip install --user --requirement requirements.txt
pip install --user .
