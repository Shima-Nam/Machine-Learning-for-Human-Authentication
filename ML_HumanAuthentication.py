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