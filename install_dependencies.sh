#Install conda env
conda env create -f environment.yml

conda activate rarefold2
pip install numpy==1.26.4
pip install jax==0.4.29 jaxlib==0.4.29+cuda12.cudnn91 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax-cuda12-plugin==0.4.29

conda deactivate

### Get network parameters for RareFold (a few minutes)
##RareFold
#ZENODO=https://zenodo.org/records/14892196/files
#wget $ZENODO/params20000.npy
#mkdir data/params
#mv params20000.npy  data/params/
##EvoBindRare
#wget $ZENODO/finetuned_params25000.npy
#mv finetuned_params25000.npy data/params/
#
### Get Uniclust30 (10-20 minutes depending on bandwidth)
## 25 Gb download, 87 Gb extracted
#wget http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz --no-check-certificate
#mkdir data/uniclust30
#mv uniclust30_2018_08_hhsuite.tar.gz data
#cd data
#tar -zxvf uniclust30_2018_08_hhsuite.tar.gz
#cd ..

### Install HHblits (a few minutes)
#git clone https://github.com/soedinglab/hh-suite.git
#mkdir -p hh-suite/build && cd hh-suite/build
#cmake -DCMAKE_INSTALL_PREFIX=. ..
#make -j 4 && make install
#cd ../..

