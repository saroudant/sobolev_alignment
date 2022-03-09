# Sobolev Alignment

Alignment of deep probabilistic models for comparing cell line and tumor single cell profiles.

## Requirements
### scVI-tools
Sobolev Alignment runs on scvi-tools version 0.10.0. More details available on <a href="https://scvi-tools.org/">scvi-tools homepage</a>. Installation through setup.py or pip will install the correct version.

### Falkon
Sobolev Alignment requires Falkon to run. Installation procedure is thoroughly explained on <a href="https://falkonml.github.io/falkon/install.html">Falkon homepage</a>. You need to install Falkon <b>before installing Sobolev Alignment</b> as this requirement is not added to the setup.

### CUDA and GPU compliance
Our implementation has been tested for Cuda 11.2 and exploits the multi-GPU implementation of Falkon. However, compatibility problems are frequent.

