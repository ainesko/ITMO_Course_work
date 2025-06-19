import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import re


H_Musical = pd.read_csv('/Users/ksenia/Desktop/ITMO/H_s_musical_SkinMelanoma.csv', index_col=0)
H_SigProfiler = pd.read_csv('/Users/ksenia/Desktop/ITMO/SkinMel_results_SigProfiler/SBS96/Suggested_Solution/COSMIC_SBS96_Decomposed_Solution/Activities/COSMIC_SBS96_Activities.txt', sep='\t', index_col=0).T
H_DualSimplex = pd.read_csv('/Users/ksenia/dualsimplex_paper/figures/H_matrix_coefficients_SkinMel_Musical.csv', index_col=0)
H_true = pd.read_csv('/Users/ksenia/Desktop/ITMO/simulated_example.Skin.Melanoma.H_true.csv', index_col=0)
H_DualSimplex_check = pd.read_csv('/Users/ksenia/dualsimplex_paper/figures/H_matrix_for_known_signatures.csv', index_col=0)
X = pd.read_csv("/Users/ksenia/MuSiCal/examples/data/simulated_example.Skin.Melanoma.X.csv", index_col=0)


# Function to extract numerical and alphabetical parts from signature names
def extract_number(sig_name):
    """
    Parses signature names into sortable components.
    Example:
    'SBS1' → (1, '')
    'SBS7a' → (7, 'a')
    'SBS13' → (13, '')
    """
    match = re.match(r"SBS(\d+)([a-z]*)", sig_name)
    if match:
        num = int(match.group(1))  # Extract numeric part (1, 2, 7, etc.)
        suffix = match.group(2)    # Extract letter suffix ('a', 'b', etc.)
        return (num, suffix)       # Tuple for hierarchical sorting
    return (float('inf'), '')      # Fallback for non-matching formats


# Set modern plotting style with white grid background
plt.style.use('seaborn-v0_8-whitegrid')

# Configure font sizes for better readability
plt.rcParams.update({
    'font.size': 14,        # Base font size
    'axes.labelsize': 14,   # Axis labels
    'xtick.labelsize': 13,  # X-axis ticks
    'ytick.labelsize': 13,  # Y-axis ticks
    'legend.fontsize': 12   # Legend
})

# Create numeric sorting key from signature names (e.g., 'SBS1' → 1)
filtered_df['sort_key'] = filtered_df.index.map(extract_number)

# Sort signatures numerically and remove temporary column
filtered_df = filtered_df.sort_values(by='sort_key').drop(columns='sort_key')

# Initialize figure with high DPI for publication quality
plt.figure(figsize=(12, 6), dpi=100)

# Create bar plot with clean styling
ax = filtered_df.plot(
    kind='bar',
    width=0.8,
    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],  # Colorblind-friendly palette
    edgecolor='k',  # Black borders for definition
    linewidth=0.5
)

# Add essential labels
plt.title("Mutational Signatures (SP82399)", pad=20)
plt.ylabel("Contribution", labelpad=10)
plt.xlabel("Signature", labelpad=10)

# Axis formatting
plt.xticks(rotation=45, ha='right')  # Diagonal labels with right alignment
plt.grid(axis='y', linestyle=':', alpha=0.7)  # Subtle grid lines

# Clean legend styling
plt.legend(title="Method", frameon=False)

# Export high-resolution PNG
plt.savefig('signatures.png', dpi=300, bbox_inches='tight')
plt.show()


def sort_signatures(signatures):
    """Sort SBS signatures numerically and alphabetically.
    
    Args:
        signatures: List of signature names (e.g., ['SBS1', 'SBS2a', 'SBS13'])
    
    Returns:
        List of signatures sorted by numeric then alphabetic components
    """
    def extract_key(sig):
        match = re.match(r"SBS(\d+)([a-z]*)", sig)
        if match:
            return (int(match.group(1)), match.group(2)  # (numeric, alphabetic)
        return (float('inf'), '')  # Push invalid formats to end
    return sorted(signatures, key=extract_key)

# Sort ground truth signatures
true_sigs = sort_signatures(H_true.index)

# Initialize comparison DataFrame
mae_comparison = pd.DataFrame(index=true_sigs)

# Methods to evaluate (name: signature matrix)
methods = {
    'SigProfiler': H_SigProfiler,
    'DS_NNLS': H_DualSimplex,
    'MuSiCal': H_Musical,
    'DS_NMF': H_DS_NMF
}

# Calculate MAE for each method
for method_name, method_data in methods.items():
    aligned = method_data.reindex(true_sigs, fill_value=0)  # Align to ground truth
    mae_comparison[method_name] = np.abs(H_true - aligned).mean(axis=1)

# Visualization
plt.figure(figsize=(18, 6))
ax = sns.heatmap(
    mae_comparison.T,  # Transpose for methods on Y-axis
    annot=True,
    fmt=".3f",
    cmap="YlOrRd",
    cbar_kws={'label': 'Mean Absolute Error (MAE)'},
    linewidths=0.5,
    annot_kws={"size": 9},
    vmin=0,  # Ensure consistent color scale
    vmax=0.5  # Adjust based on your data range
)

# Formatting
plt.title("MAE Comparison Across Signature Extraction Methods", pad=20)
plt.xlabel("COSMIC SBS Signatures")
plt.ylabel("Extraction Method")
plt.xticks(rotation=45, ha='right')

# Annotate missing signatures
for j, method in enumerate(methods.keys()):
    missing = set(true_sigs) - set(methods[method].index)
    for i, sig in enumerate(true_sigs):
        if sig in missing:
            ax.text(
                i + 0.5, j + 0.5,  # Center in cell
                "MISS", 
                ha='center', 
                va='center', 
                color='blue', 
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, lw=0)  # White background
            )

plt.tight_layout()
plt.savefig('MAE_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
