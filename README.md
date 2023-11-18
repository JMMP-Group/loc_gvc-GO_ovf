# loc_gvc-GO_ovf
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10149529.svg)](https://doi.org/10.5281/zenodo.10149529)

Repository to support the study presented in:

Localised general vertical coordinates for quasi-Eulerian ocean models: the Nordic over ows test-case

The code of this repository uses and reproduces the data archived in [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8055023.svg)](https://doi.org/10.5281/zenodo.8055023) 

## Installing dependencies

1) First, the pyogcm conda environment needs to be installed:

```shell
git clone git@github.com:JMMP-Group/loc_gvc-GO_ovf.git
cd loc_gvc-GO_ovf
conda env create -f pyogcm.yml
conda activate pyogcm
```
2) Then, the nordic-seas-validation repository can be installed

```shell
git clone git@github.com:JMMP-Group/nordic-seas-validation.git
cd nordic-seas-validation
pip install -e .
python -c "import nsv"
```

