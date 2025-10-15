import pandas as pd
import numpy as np
import os


# -----------------------------------------------------------------------------
# Function: load_and_prepare_data
# Loads a CSV file containing feature data, assigns column names, adds a Subject_ID
# column, shuffles the data, and performs basic validation checks.
#
# Args:
#   csv_path (str): Path to the CSV file.
#   num_features (int): Number of feature columns expected in the CSV.
#   num_subjects (int): Number of unique subjects in the dataset.
#   beats_per_subject (int): Number of samples (beats) per subject.
#   random_seed (int): Seed for reproducibility.
#
# Returns:
#   pd.DataFrame: Prepared and validated DataFrame with features and Subject_ID.
# -----------------------------------------------------------------------------
def load_and_prepare_data(csv_path, num_features, num_subjects, beats_per_subject, random_seed):
    # Check if the file exists
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    np.random.seed(random_seed)
    # Load the CSV file without headers
    df = pd.read_csv(csv_path, header=None)
    # Validate the number of columns
    if df.shape[1] != num_features:
        raise ValueError(f"CSV has {df.shape[1]} columns, but num_features={num_features}.")
    
    print(df.head())
    
    # Validate the number of rows
    expected_rows = num_subjects * beats_per_subject
    if df.shape[0] != expected_rows:
        raise ValueError(f"CSV has {df.shape[0]} rows, but expected {expected_rows} (num_subjects * beats_per_subject).")
    # Assign feature column names
    df.columns = [f"F_{i+1}" for i in range(num_features)]
    # Create Subject_IDs: repeated for each subject
    subject_ids = np.repeat(np.arange(num_subjects)+1, beats_per_subject)
    df["Subject_ID"] = subject_ids

    print(len(subject_ids))  # 33 * 160 = 5280

    # Shuffle the DataFrame rows
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df