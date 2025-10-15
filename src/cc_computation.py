import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Function: CC
# Computes CC1 (mean), CC2_min (min), and CC2_max (max) differences between X and
# Supervisor for each subject. For each subject, the function processes the data
# in chunks of 10 samples, computes the absolute difference between each chunk and
# the supervisor values, and aggregates the results.
#
# Args:
#   X (pd.DataFrame): DataFrame with features and 'Subject_ID' for the main data.
#   Supervisor (pd.DataFrame): DataFrame with features and 'Subject_ID' for the supervisor.
#
# Returns:
#   CC1_df (pd.DataFrame): DataFrame of mean differences per chunk (CC1).
#   CC2_min_df (pd.DataFrame): DataFrame of min differences per chunk (CC2_min).
#   CC2_max_df (pd.DataFrame): DataFrame of max differences per chunk (CC2_max).
# -----------------------------------------------------------------------------
def CC(X, Supervisor):
    # Data validation: check for required columns and missing values
    for df, name in zip([X, Supervisor], ['X', 'Supervisor']):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"{name} must be a pandas DataFrame.")
        if 'Subject_ID' not in df.columns:
            raise ValueError(f"{name} must contain a 'Subject_ID' column.")
        if df.drop('Subject_ID', axis=1).isnull().any().any():
            raise ValueError(f"{name} contains missing values in feature columns.")

    CC1 = []
    CC2_min = []
    CC2_max = []
    subject_ids = []

    # Sort both DataFrames by Subject_ID for consistency
    X = X.sort_values(by='Subject_ID').reset_index(drop=True)
    Supervisor = Supervisor.sort_values(by='Subject_ID').reset_index(drop=True)

    # Process each subject separately
    for subject, X_group in X.groupby("Subject_ID"):
        # Data validation: check if supervisor data exists for this subject
        Supervisor_values = Supervisor[Supervisor["Subject_ID"] == subject].drop(columns=["Subject_ID"]).values
        if Supervisor_values.shape[0] == 0:
            raise ValueError(f"No supervisor data found for subject {subject}.")
        X_group = X_group.drop(columns=["Subject_ID"]).reset_index(drop=True)
        # Process in chunks of 10 samples
        for i in range(0, len(X_group), 10):
            if i + 10 > len(X_group):
                break  # Skip incomplete chunk
            X_chunk = X_group.iloc[i:i+10]
            # Compute absolute difference between chunk and supervisor values
            diff = np.abs(X_chunk - Supervisor_values)
            # Aggregate statistics
            CC1.append(diff.mean(axis=0))
            CC2_min.append(diff.min(axis=0))
            CC2_max.append(diff.max(axis=0))
            subject_ids.append(subject)

    # Prepare output DataFrames with appropriate column names
    CC1_df = pd.DataFrame(CC1, columns=X_group.columns).add_suffix('_mean')
    CC2_min_df = pd.DataFrame(CC2_min, columns=X_group.columns).add_suffix('_min')
    CC2_max_df = pd.DataFrame(CC2_max, columns=X_group.columns).add_suffix('_max')
    for df in [CC1_df, CC2_min_df, CC2_max_df]:
        df.insert(0, "Subject_ID", subject_ids)
    return CC1_df, CC2_min_df, CC2_max_df