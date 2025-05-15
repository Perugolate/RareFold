# RareFold
Structure prediction and design of proteins with noncanonical amino acids.

RareFold predicts single-chain protein structures containing noncanonical amino acids (NCAAs) and enables the design of novel peptide binders through the EvoBindRare framework. \
EvoBindRare designs both linear and cyclic binders directly from a protein target sequence—no prior knowledge of binding sites is required (though optional). This allows rapid, flexible design of binders with expanded chemical diversity.

<img src="./RareFold.svg"/>

# LICENSE
RareFold is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).  \
The RareFold parameters for prediction are made available under the terms of the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode). \
The design protocol EvoBindRare and the parameters for design are made available under the terms of the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).


# Installation
## (several minutes)
The entire installation takes <1 hour on a standard computer. \
We assume you have CUDA12. For CUDA11, you will have to change the installation of some packages. \
The runtime will depend on the GPU you have available and the size of the protein-ligand complex you are predicting. \
On an NVIDIA A100 GPU, the prediction time is a few minutes on average.

First install miniconda, see: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html or https://docs.conda.io/projects/miniconda/en/latest/miniconda-other-installer-links.html


```
bash install_dependencies.sh
```

1. Get the RareFold parameters for single-chain structure prediction
2. Get the EvoBindRare parameters for binder design
3. Get Uniclust for MSA search
4. Install the RareFold environment

# Predict using the forward model
## Run the test case (a few minutes)
```
conda activate rarefold
bash predict.sh
```


# EvoBindRare
## Design linear or cyclic peptide binders incorporating noncanonical amino acids
<p align="center">
  <img alt="Linear" src="./linear.gif" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Cyclic" src="./cyclic.gif" width="45%">
</p>

```
conda activate rarefold
bash design.sh
```
