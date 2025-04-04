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

#%%
#developing CC1 and CC2 functions

def CC(X, Supervisor):
    CC1 = []  # Stores mean difference (CC1)
    CC2_min = []  # Stores min difference (CC2_min)
    CC2_max = []  # Stores max difference (CC2_max)
    subject_ids = []  # To track Subject_IDs

    # Ensure data is sorted by Subject_ID
    X = X.sort_values(by='Subject_ID').reset_index(drop=True)
    Supervisor = Supervisor.sort_values(by='Subject_ID').reset_index(drop=True)

    # Group data by Subject_ID
    for subject, X_group in X.groupby("Subject_ID"):
        # Get supervisor values for this subject (drop Subject_ID for subtraction)
        Supervisor_values = Supervisor[Supervisor["Subject_ID"] == subject].drop(columns=["Subject_ID"]).values
        
        # Drop Subject_ID from X_group as well
        X_group = X_group.drop(columns=["Subject_ID"]).reset_index(drop=True)

        # Iterate over beats in chunks of 10
        for i in range(0, len(X_group), 10):
            if i + 10 > len(X_group):  # Avoid exceeding available beats
                break

            # Select 10 beats
            X_chunk = X_group.iloc[i:i+10]

            # Compute absolute difference with Supervisor
            diff = np.abs(X_chunk - Supervisor_values)

            # Compute mean, min, and max across the 10 beats
            CC1.append(diff.mean(axis=0))  # Mean difference (CC1)
            CC2_min.append(diff.min(axis=0))  # Min difference (CC2_min)
            CC2_max.append(diff.max(axis=0))  # Max difference (CC2_max)

            # Store Subject_ID
            subject_ids.append(subject)

    # Convert lists to DataFrame
    CC1_df = pd.DataFrame(CC1, columns=X_group.columns)
    CC2_min_df = pd.DataFrame(CC2_min, columns=X_group.columns)
    CC2_max_df = pd.DataFrame(CC2_max, columns=X_group.columns)

    # Rename the columns    
    CC1_df = CC1_df.add_suffix('_mean')
    CC2_min_df = CC2_min_df.add_suffix('_min')
    CC2_max_df = CC2_max_df.add_suffix('_max')

    # Add Subject_IDs
    CC1_df.insert(0, "Subject_ID", subject_ids)
    CC2_min_df.insert(0, "Subject_ID", subject_ids)
    CC2_max_df.insert(0, "Subject_ID", subject_ids)
    
    return CC1_df, CC2_min_df, CC2_max_df

X_sel = df[selected_features_names] #total raw data with selected features
X_sel['Subject_ID'] = df.Subject_ID

from sklearn.model_selection import train_test_split

# looping over the test size
Test = np.linspace(0.6, 0.2, num=3)

CC1_X_list, CC2_min_X_list, CC2_max_X_list = [], [], []
CC1_Xt_list, CC2_min_Xt_list, CC2_max_Xt_list = [], [], []
CC_X_dict = {}

#we can estimate the number of iteration on selecting the supervisor beats through simulation (Monte Carlo method).
#From simulations, the expected number of iterations typically falls between 25 and 35 iterations.
#The actual value may slightly vary depending on randomness.

for i,test_sz in enumerate(Test):
  
    for _ in range(10):
        
        #first selecting 10 supervisor_beats and 100 beats for split
        num_beats = 110
        data = (
            X_sel.groupby("Subject_ID")
            .apply(lambda x: x.sample(n=num_beats, random_state=108))
            .reset_index(drop=True)
        )    
    
        num_supervisor_beats = 10
        supervisor_beats = (
            data.groupby("Subject_ID")
            .apply(lambda x: x.sample(n=num_supervisor_beats)) 
        )
    
        supervisor_beats_index = supervisor_beats.index.get_level_values(1)
        supervisor_beats = supervisor_beats.reset_index(drop=True)
        X_sel_split = data.drop(index=supervisor_beats_index).reset_index(drop = True)
        
        #--------------------
        #spliting train and test sets and scaling
        X = X_sel_split.drop('Subject_ID', axis=1)
        y = X_sel_split.Subject_ID
        X_Supervisor = supervisor_beats.drop('Subject_ID', axis=1)
        y_Supervisor = supervisor_beats.Subject_ID
        
        #split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 108, stratify=y)
        #scale
        X_train = scaler.fit_transform(X_train)
        X_Supervisor = scaler.fit_transform(X_Supervisor)
        X_test = scaler.transform(X_test)    
        
        #---------
        # Calculation of CC1 and CC2
        
        X_y_train = X_train  
        X_y_train['Subject_ID'] = y_train
        X_y_Supervisor = X_Supervisor 
        X_y_Supervisor['Subject_ID'] = y_Supervisor
        X_y_test = X_test
        X_y_test['Subject_ID'] = y_test
        CC1_X, CC2_min_X, CC2_max_X = CC(X_y_train, X_y_Supervisor)
        CC1_Xt, CC2_min_Xt, CC2_max_Xt = CC(X_y_test, X_y_Supervisor )
        
        # Append results to the lists
        CC1_X_list.append(CC1_X)
        CC2_min_X_list.append(CC2_min_X)
        CC2_max_X_list.append(CC2_max_X)
        
        CC1_Xt_list.append(CC1_Xt)
        CC2_min_Xt_list.append(CC2_min_Xt)
        CC2_max_Xt_list.append(CC2_max_Xt)
  
    # Concatenate results supervisor iteration for each test set
    CC1_X_final = pd.concat(CC1_X_list, ignore_index=True)
    CC2_min_X_final = pd.concat(CC2_min_X_list, ignore_index=True)
    CC2_max_X_final = pd.concat(CC2_max_X_list, ignore_index=True)
    
    CC1_Xt_final = pd.concat(CC1_Xt_list, ignore_index=True)
    CC2_min_Xt_final = pd.concat(CC2_min_Xt_list, ignore_index=True)
    CC2_max_Xt_final = pd.concat(CC2_max_Xt_list, ignore_index=True)

    CC_X_dict[i] = {'CC1_X_final': CC1_X_final,
                     'CC2_min_X_final': CC2_min_X_final,
                     'CC2_max_X_final': CC2_max_X_final,
                     'CC1_Xt_final': CC1_Xt_final,
                     'CC2_min_Xt_final': CC2_min_Xt_final,
                     'CC2_max_Xt_final': CC2_max_Xt_final}

del CC2_min_X, CC2_min_X_final, CC2_min_X_list, CC2_min_Xt, CC2_min_Xt_final, CC2_min_Xt_list, 
del CC2_max_X, CC2_max_X_final, CC2_max_X_list, CC2_max_Xt, CC2_max_Xt_final, CC2_max_Xt_list
del CC1_X, CC1_X_final, CC1_X_list, CC1_Xt, CC1_Xt_final, CC1_Xt_list
del Test, i, test_sz, X_sel, X_sel_split, X_Supervisor, X_test, X_train, X_y_Supervisor, X_y_test
del X_y_train, y_Supervisor, y_test, y_train, X, y 

#%%
#Classification

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

# Define hyperparameter grids for the model
param_grid = [
    {
        'model': [CatBoostClassifier(verbose=0)],  # Suppress training logs
        'model__iterations': [100, 200, 300],
        'model__learning_rate': [0.01 , 0.3, 0.5]
    }    
    ]

dict_nb = [0, 1, 2]
feature_nb = [10, 20, 31]
# Dictionary to store results
results = {}

for i in dict_nb:
    #break the dictionary to each train_test split
    train_test_set = CC_X_dict[i]
    Xtrain1_mean = train_test_set['CC1_X_final'].drop('Subject_ID',axis = 1)
    ytrain1    = train_test_set['CC1_X_final'].Subject_ID
    Xtrain2_max  = train_test_set['CC2_max_X_final'].drop('Subject_ID',axis = 1)
    #ytrain2    = train_test_set['CC2_max_X_final'].Subject_ID
    Xtest1_mean  = train_test_set['CC1_Xt_final'].drop('Subject_ID',axis = 1)
    ytest1     = train_test_set['CC1_Xt_final'].Subject_ID
    Xtest2_max   = train_test_set['CC2_max_Xt_final'].drop('Subject_ID',axis = 1)
   # ytest2     = train_test_set['CC2_max_Xt_final'].Subject_ID

    results[i] = {}  # Store results per train-test split
    
    for subset_idx, col in enumerate(feature_nb):
      
        X_train_subset = pd.concat([Xtrain1_mean.iloc[:, :col], Xtrain2_max.iloc[:, :col]], axis = 1)
        y_train = ytrain1 #same as ytrain2
        X_test_subset = pd.concat([Xtest1_mean.iloc[:, :col], Xtest2_max.iloc[:, :col]], axis = 1)      
        y_test = ytest1 
        
        # Define a new pipeline for each iteration
        pipeline = Pipeline([
            ('model', RandomForestClassifier(random_state=108))])

        # Define a new GridSearchCV instance over multiple estimators
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        # Fit GridSearchCV on the current training data
        grid_search.fit(X_train_subset, y_train)

       # i = split_idx
        results[i][subset_idx] = {
            'cv_results' : grid_search.cv_results_,
            'best_model' : grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'test_score' : grid_search.score(X_test_subset, y_test),
            'best_score' : grid_search.best_score_
        }

import pickle

# Save results
with open("...\\results.pkl", "wb") as f:
    pickle.dump(results, f)

# Load results later
with open("...\\results.pkl", "rb") as f:
    loaded_results = pickle.load(f)

#%%                
#Visualiztion
import matplotlib.pyplot as plt

Test = [0.6, 0.4, 0.2]  # Define the actual test sizes

for idx, (test_size, subsets) in enumerate(loaded_results.items()):
    feature_counts = []
    accuracies = []

    for feature_idx, data in subsets.items():
        feature_counts.append(feature_nb[feature_idx])  # Number of features
        accuracies.append(data['test_score'])  # Accuracy

    plt.plot(feature_counts, accuracies, marker='o', label=f"Test size: {Test[idx]}")  # Use actual test size

plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Features")
plt.legend()
plt.grid(True)
plt.show()

test_sizes = []
best_accuracies = []

for test_size, subsets in loaded_results.items():
    test_sizes.append(test_size)
    
    # Get the best accuracy for this test size
    best_acc = max(subsets[feature_idx]['test_score'] for feature_idx in subsets)
    best_accuracies.append(best_acc)

plt.plot(test_sizes, best_accuracies, marker='s', linestyle='-', color='r')
plt.xticks([0, 1 , 2], labels = [0.6, 0.4, 0.2])
plt.xlabel("Test Size")
plt.ylabel("Best Accuracy")
plt.title("Accuracy vs Test Size")
plt.grid(True)
plt.show()