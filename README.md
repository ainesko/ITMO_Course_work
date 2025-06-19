# Evaluation performance for methods for somatic mutational signatures deconvolution
## Authors
K.Mardanova¹, M.Artomov² 
1. ITMO University
2. Institute for Genomic Medicine Nationwide Children’s Hospital
## Project description

Comparative evaluation of DualSimplex, SigProfiler, and MuSiCal for de novo signature extraction and exposure estimation.


## MuSiCal installation

```bash
git clone https://github.com/parklab/MuSiCal

source ~/miniconda3/bin/activate root
conda create -n python37_musical python=3.7
conda install numpy scipy scikit-learn matplotlib pandas seaborn

cd  /Path/To/MuSiCal
pip install ./MuSiCal
```

## SigProfilerExtractor installation
```bash
# Create a clean conda environment (recommended)
conda create -n sigprofiler python=3.9
conda activate sigprofiler

# Install SigProfilerExtractor and dependencies
pip install SigProfilerExtractor
```
