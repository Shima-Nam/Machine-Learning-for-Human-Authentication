import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from collections import defaultdict
import matplotlib.pyplot as plt  # For plotting dendrogram and correlation matrix

# -----------------------------------------------------------------------------
# Function: select_features_dendrogram
# Selects representative features using hierarchical clustering on the feature
# correlation matrix. Optionally plots the dendrogram and reordered correlation
# matrix.
#
# Args:
#   df (pd.DataFrame): Input DataFrame with features and 'Subject_ID'.
#   scaler (sklearn Scaler, optional): Scaler to use for standardization.
#       If None, uses StandardScaler.
#   plot (bool): If True, plots the dendrogram and reordered correlation matrix.
#
# Returns:
#   selected_features_names (list): List of selected feature names.
#   scaler: The fitted scaler.
# -----------------------------------------------------------------------------
def select_features_dendrogram(
    df: 'pd.DataFrame', 
    scaler: 'StandardScaler' = None, 
    plot: bool = False
) -> tuple[list, StandardScaler]:
    # Data validation: check for 'Subject_ID' column
    if 'Subject_ID' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Subject_ID' column.")
    # Data validation: check for at least two feature columns
    feature_cols = [col for col in df.columns if col != 'Subject_ID']
    if len(feature_cols) < 2:
        raise ValueError("Input DataFrame must contain at least two feature columns (excluding 'Subject_ID').")
    # Data validation: check for missing values
    if df[feature_cols].isnull().any().any():
        raise ValueError("Input DataFrame contains missing values in feature columns.")

    if scaler is None:
        scaler = StandardScaler().set_output(transform="pandas")
    # Scale the data (excluding 'Subject_ID')
    df_scaled = scaler.fit_transform(df)
    X = df_scaled.drop('Subject_ID', axis=1)
    # Compute Spearman correlation matrix
    corr = spearmanr(X).correlation
    
    # Handle case where spearmanr returns a scalar instead of a matrix
    if not isinstance(corr, np.ndarray) or corr.ndim != 2:
        raise ValueError("Spearman correlation failed — likely due to constant or identical features.")
    
    # Ensure symmetry and set diagonal to 1
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # Convert correlation to distance matrix for clustering
    distance_matrix = 1 - np.abs(corr)
    # Perform hierarchical clustering (Ward linkage)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    # Assign cluster IDs based on a distance threshold
    cluster_ids = hierarchy.fcluster(dist_linkage, 0.66, criterion="distance")
    # Group feature indices by cluster
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    # Select the first feature from each cluster as representative
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_features_names = X.columns[selected_features]

    if plot:
        # Plot dendrogram and reordered correlation matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        dendro = hierarchy.dendrogram(dist_linkage, labels=X.columns, ax=ax1, leaf_rotation=90)
        dendro_idx = np.arange(0, len(dendro["ivl"]))
        # Show the correlation matrix reordered according to the dendrogram leaves
        ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
        ax2.set_xticks(dendro_idx)
        ax2.set_yticks(dendro_idx)
        ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
        ax2.set_yticklabels(dendro["ivl"])
        _ = fig.tight_layout()
        plt.show()

    if np.isnan(corr).any():
        raise ValueError("Correlation matrix contains NaN values. Check for constant features.")

    return selected_features_names, scaler
