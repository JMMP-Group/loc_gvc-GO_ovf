# loc_gvc-GO_ovf
Repository to support the study presented in:

Localised general vertical coordinates for quasi-Eulerian ocean models: the Nordic over ows test-case

The code of this repository uses and reproduces the data archived at https://zenodo.org/records/8055023 

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

