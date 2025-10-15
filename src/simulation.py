import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .cc_computation import CC

# -----------------------------------------------------------------------------
# Function: run_simulation
# Runs repeated sampling, splitting, scaling, and CC feature computation for
# different test sizes. Returns a dictionary of concatenated CC features for
# each test size.
#
# Args:
#   X_sel (pd.DataFrame): DataFrame containing features and 'Subject_ID'.
#   test_sizes (list): List of test set sizes (fractions) to simulate.
#   iterations (int): Number of repetitions per test size.
#   random_state (int): Seed for reproducibility.
#
# Returns:
#   results_dict (dict): Dictionary with CC features for each test size.
# -----------------------------------------------------------------------------
def run_simulation(X_sel, test_sizes, iterations=10, random_state=108):
    # Data validation: check for required columns and missing values
    if not isinstance(X_sel, pd.DataFrame):
        raise ValueError("X_sel must be a pandas DataFrame.")
    if 'Subject_ID' not in X_sel.columns:
        raise ValueError("Input DataFrame must contain a 'Subject_ID' column.")
    if X_sel.drop('Subject_ID', axis=1).isnull().any().any():
        raise ValueError("Input DataFrame contains missing values in feature columns.")
    if not isinstance(test_sizes, list) or not test_sizes:
        raise ValueError("test_sizes must be a non-empty list.")
    if iterations < 1:
        raise ValueError("iterations must be >= 1.")

    scaler = StandardScaler()
    results_dict = {}

    for i, test_sz in enumerate(test_sizes):
        CC1_X_list, CC2_min_X_list, CC2_max_X_list = [], [], []
        CC1_Xt_list, CC2_min_Xt_list, CC2_max_Xt_list = [], [], []

        for it in range(iterations):
            # Sample up to 110 beats per subject
            data = (
                X_sel.loc[:, X_sel.columns]  # ensures we're not operating on the group column itself
                .groupby("Subject_ID", group_keys=False)
                .apply(lambda x: x.sample(n=min(len(x), 110), random_state=random_state))
                .reset_index(drop=True)
            )
            
            # Sample up to 10 supervisor beats per subject
            supervisor = (
                data.loc[:, data.columns]
                .groupby("Subject_ID", group_keys=False)
                .apply(lambda x: x.sample(n=min(len(x), 10), random_state=random_state))
                .reset_index(drop=True)
            )
            
            # Remove supervisor beats from data
            supervisor_idx = supervisor.index
            X_split = data.drop(supervisor_idx).reset_index(drop=True)

            # Data validation: check if enough samples remain for splitting
            if X_split.shape[0] == 0:
                raise ValueError("No samples left after removing supervisor beats.")

            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_split.drop("Subject_ID", axis=1),
                X_split["Subject_ID"],
                test_size=test_sz,
                random_state=random_state,
                stratify=X_split["Subject_ID"]
            )

            # Prepare supervisor features and labels
            X_Supervisor = supervisor.drop("Subject_ID", axis=1)
            y_Supervisor = supervisor["Subject_ID"]

            # Scale features
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
            X_Supervisor = pd.DataFrame(scaler.fit_transform(X_Supervisor), columns=X_Supervisor.columns)

            # Reconstruct DataFrames with Subject_ID
            train_df = X_train.copy()
            train_df["Subject_ID"] = y_train.reset_index(drop=True)
            supervisor_df = X_Supervisor.copy()
            supervisor_df["Subject_ID"] = y_Supervisor.reset_index(drop=True)
            test_df = X_test.copy()
            test_df["Subject_ID"] = y_test.reset_index(drop=True)

            # Compute CC features for train and test sets
            CC1_X, CC2_min_X, CC2_max_X = CC(train_df, supervisor_df)
            CC1_Xt, CC2_min_Xt, CC2_max_Xt = CC(test_df, supervisor_df)

            # Collect results for this iteration
            CC1_X_list.append(CC1_X)
            CC2_min_X_list.append(CC2_min_X)
            CC2_max_X_list.append(CC2_max_X)
            CC1_Xt_list.append(CC1_Xt)
            CC2_min_Xt_list.append(CC2_min_Xt)
            CC2_max_Xt_list.append(CC2_max_Xt)

        # Concatenate results from all iterations for this test size
        results_dict[i] = {
            'CC1_X_final': pd.concat(CC1_X_list, ignore_index=True),
            'CC2_min_X_final': pd.concat(CC2_min_X_list, ignore_index=True),
            'CC2_max_X_final': pd.concat(CC2_max_X_list, ignore_index=True),
            'CC1_Xt_final': pd.concat(CC1_Xt_list, ignore_index=True),
            'CC2_min_Xt_final': pd.concat(CC2_min_Xt_list, ignore_index=True),
            'CC2_max_Xt_final': pd.concat(CC2_max_Xt_list, ignore_index=True),
        }

    return results_dict
