# RareFold
Structure prediction and design of proteins with noncanonical amino acids.

RareFold predicts single-chain protein structures with noncanonical amino acids, but also designs
novel peptide binders incorporating noncanonical amino acids through the framework EvoBindRare (below). \
EvoBindRare designs binders based only on a protein target sequence. It is not necessary to specify any target residues within the protein sequence (although this is possible). Cyclic binder design is also possible.

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
