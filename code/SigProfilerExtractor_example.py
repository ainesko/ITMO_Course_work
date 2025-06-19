import SigProfilerExtractor
import pandas as pd
import numpy as np
import os 

# File should be in tsv format
path_to_skin_table = '/Users/ksenia/SigProfilerExtractor/SigProfilerExtractor/data/TextInput/simulated_example.Skin.Melanoma.X.txt'
sig.sigProfilerExtractor("matrix", "SKCM_musical_final_results_SigProfiler", path_to_skin_table, opportunity_genome="GRCh38", minimum_signatures=1, maximum_signatures=15)
