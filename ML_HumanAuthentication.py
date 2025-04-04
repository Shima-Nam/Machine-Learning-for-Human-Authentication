# %% loading-preparing dataset 

import pandas as pd
import numpy as np
np.random.seed(108)

import os
print(os.getcwd())  # Check current working directory

df = pd.read_csv("...\\features.csv", header=None)

# Assign column names
num_features = 170  # Number of features per beat
df.columns = [f"F_{i+1}" for i in range(num_features)]

print(df.head())

# Number of subjects and beats per subject
num_subjects = 33
beats_per_subject = 160

# Create Subject_IDs: 1,1,...,1 (160 times), ..., 32,32,...,32 (160 times)
subject_ids = np.repeat(np.arange(num_subjects)+1, beats_per_subject)

# Verify the length
print(len(subject_ids))  # 33 * 160 = 5280

df["Subject_ID"] = subject_ids
df.head()

df = df.sample(frac=1, random_state=108).reset_index(drop=True)

del beats_per_subject, num_features, num_subjects

#%%
# Reducing feature space using dendrogram 
# first scaling the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform="pandas")
df_for_dendro = scaler.fit_transform(df)

#%%
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

X = df_for_dendro.drop('Subject_ID', axis = 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=X.columns.to_list(), ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
_ = fig.tight_layout()

plt.show()

from collections import defaultdict

cluster_ids = hierarchy.fcluster(dist_linkage, 0.66, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
selected_features_names = X.columns[selected_features]

del ax1, ax2, cluster_id, fig, idx, cluster_id_to_feature_ids, cluster_ids, corr, dendro,
del dendro_idx, dist_linkage, distance_matrix
