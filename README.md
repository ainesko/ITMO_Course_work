# Evaluation performance for methods for somatic mutational signatures deconvolution
## Authors
K.Mardanova¹, M.Artomov² 
1. ITMO University
2. Institute for Genomic Medicine Nationwide Children’s Hospital
## Project description

Comparative evaluation of DualSimplex, SigProfiler, and MuSiCal for de novo signature extraction and exposure estimation.


## MuSiCal installation

```bash
# Clone github repository
git clone https://github.com/parklab/MuSiCal

# Create a clean conda environment (recommended)
source ~/miniconda3/bin/activate root
conda create -n python37_musical python=3.7
conda install numpy scipy scikit-learn matplotlib pandas seaborn

# Install MuSiCal and dependencies
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

# Install Reference Genome
from SigProfilerMatrixGenerator import install as gen_install
gen_install.install('GRCh38') 
```

## Comment on DualSimplex
The DualSimplex framework was developed by Kleverov et al. (2023) at Washington University School of Medicine and ITMO University. This study applies the published components of their non-negative matrix factorization approach for mutational signature analysis. Certain algorithmic enhancements remain unpublished at this time; all analyses were performed in accordance with the authors' documented methodology.

## References

Alexandrov, L. B. et al. The repertoire of mutational signatures in human cancer. Nature 578, 94–101 (2020).
https://github.com/AlexandrovLab/SigProfilerExtractor

Jin, H. et al. Accurate and sensitive mutational signature analysis with MuSiCal. Nature Genetics, 56(3), 541–552 (2024). 
https://github.com/parklab/MuSiCal

Kleverov, D. et al. Non-negative matrix factorization and deconvolution as dual simplex problem. bioRxiv 2024.04.09.588652 (2024).
https://github.com/artyomovlab/dualsimplex


