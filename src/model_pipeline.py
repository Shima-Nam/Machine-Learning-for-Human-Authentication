import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from .cc_computation import CC
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Function: sample_subject_data
# Samples a specified number of beats and supervisor beats for each subject.
#
# Args:
#   X_sel (pd.DataFrame): DataFrame containing features and 'Subject_ID'.
#   num_beats (int): Number of beats to sample per subject.
#   num_supervisor_beats (int): Number of supervisor beats to sample per subject.
#   random_seed (int): Seed for reproducibility.
#
# Returns:
#   X_sel_split (pd.DataFrame): DataFrame with sampled beats (excluding supervisor beats).
#   supervisor_beats (pd.DataFrame): DataFrame with sampled supervisor beats.
# -----------------------------------------------------------------------------
def sample_subject_data(X_sel, num_beats, num_supervisor_beats, random_seed):
    """Sample beats and supervisor beats for each subject."""
    # Data validation: check for required columns and enough samples per subject
    if 'Subject_ID' not in X_sel.columns:
        raise ValueError("Input DataFrame must contain a 'Subject_ID' column.")
    subjects = X_sel['Subject_ID'].unique()
    for subject in subjects:
        n_samples = (X_sel['Subject_ID'] == subject).sum()
        if n_samples < num_beats:
            raise ValueError(f"Subject {subject} has only {n_samples} samples, but num_beats={num_beats}.")
        if num_beats < num_supervisor_beats:
            raise ValueError(f"num_beats ({num_beats}) must be >= num_supervisor_beats ({num_supervisor_beats}).")
    # Sample num_beats for each subject
    data = (
        X_sel.groupby("Subject_ID", group_keys=False)
        .apply(lambda x: x.sample(n=num_beats, random_state=random_seed))
        .reset_index(drop=True)
    )
    # Sample supervisor beats from the previously sampled data
    supervisor_beats = (
        data.groupby("Subject_ID", group_keys=False)
        .apply(lambda x: x.sample(n=num_supervisor_beats))
    )
    supervisor_beats_index = supervisor_beats.index
    supervisor_beats = supervisor_beats.reset_index(drop=True)
    # Remove supervisor beats from the main data
    X_sel_split = data.drop(index=supervisor_beats_index).reset_index(drop=True)
    return X_sel_split, supervisor_beats

# -----------------------------------------------------------------------------
# Function: compute_cc_features
# Splits the data into train/test sets, scales them, and computes CC features
# for both sets using the supervisor beats.
#
# Args:
#   X_sel_split (pd.DataFrame): DataFrame with sampled beats (excluding supervisor beats).
#   supervisor_beats (pd.DataFrame): DataFrame with supervisor beats.
#   scaler (sklearn Scaler): Scaler for standardization.
#   test_sz (float): Test set size (fraction).
#   random_seed (int): Seed for reproducibility.
#
# Returns:
#   Tuple of DataFrames: (CC1_X, CC2_min_X, CC2_max_X, CC1_Xt, CC2_min_Xt, CC2_max_Xt)
# -----------------------------------------------------------------------------
def compute_cc_features(X_sel_split, supervisor_beats, scaler, test_sz, random_seed):
    """Split, scale, and compute CC features for train/test sets."""
    # Data validation: check for missing values and required columns
    for df, name in zip([X_sel_split, supervisor_beats], ['X_sel_split', 'supervisor_beats']):
        if 'Subject_ID' not in df.columns:
            raise ValueError(f"{name} must contain a 'Subject_ID' column.")
        if df.drop('Subject_ID', axis=1).isnull().any().any():
            raise ValueError(f"{name} contains missing values in feature columns.")
    # Prepare features and labels
    X = X_sel_split.drop('Subject_ID', axis=1)
    y = X_sel_split.Subject_ID
    X_Supervisor = supervisor_beats.drop('Subject_ID', axis=1)
    y_Supervisor = supervisor_beats.Subject_ID
    # Data validation: check for enough samples for stratified split
    if len(np.unique(y)) < 2:
        raise ValueError("Not enough classes in y for stratified split.")
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sz, random_state=random_seed, stratify=y
    )
    # Scale the data
    X_train = scaler.fit_transform(X_train)
    X_Supervisor = scaler.fit_transform(X_Supervisor)
    X_test = scaler.transform(X_test)
    # Reconstruct DataFrames with Subject_ID
    X_y_train = pd.DataFrame(X_train, columns=X.columns)
    X_y_train['Subject_ID'] = y_train.values
    X_y_Supervisor = pd.DataFrame(X_Supervisor, columns=X.columns)
    X_y_Supervisor['Subject_ID'] = y_Supervisor.values
    X_y_test = pd.DataFrame(X_test, columns=X.columns)
    X_y_test['Subject_ID'] = y_test.values
    # Compute CC features for train and test sets
    CC1_X, CC2_min_X, CC2_max_X = CC(X_y_train, X_y_Supervisor)
    CC1_Xt, CC2_min_Xt, CC2_max_Xt = CC(X_y_test, X_y_Supervisor)
    return CC1_X, CC2_min_X, CC2_max_X, CC1_Xt, CC2_min_Xt, CC2_max_Xt

# -----------------------------------------------------------------------------
# Function: run_classification
# Runs classification and hyperparameter search for each test size and feature subset.
#
# Args:
#   CC_X_dict (dict): Dictionary containing CC features for each test size.
#   feature_nb (list): List of feature counts to use for subsets.
#   param_grid (list): List of hyperparameter grids for GridSearchCV.
#   random_seed (int): Seed for reproducibility.
#
# Returns:
#   results (dict): Dictionary containing classification results for each test size and feature subset.
# -----------------------------------------------------------------------------
def run_classification(CC_X_dict, feature_nb, param_grid, random_seed):
    """Run classification and hyperparameter search for each test size and feature subset."""
    results = {}
    for i in CC_X_dict:
        train_test_set = CC_X_dict[i]
        # Prepare train and test feature sets
        Xtrain1_mean = train_test_set['CC1_X_final'].drop('Subject_ID', axis=1)
        ytrain1 = train_test_set['CC1_X_final'].Subject_ID
        Xtrain2_max = train_test_set['CC2_max_X_final'].drop('Subject_ID', axis=1)
        Xtest1_mean = train_test_set['CC1_Xt_final'].drop('Subject_ID', axis=1)
        ytest1 = train_test_set['CC1_Xt_final'].Subject_ID
        Xtest2_max = train_test_set['CC2_max_Xt_final'].drop('Subject_ID', axis=1)
        results[i] = {}
        print(f"Starting classification for test size index {i+1}/{len(CC_X_dict)}")
        # Iterate over feature subset sizes
        for subset_idx, col in tqdm(list(enumerate(feature_nb)), desc=f"Feature subsets for test size {i+1}"):
            # Data validation: check if enough features are available
            if col > Xtrain1_mean.shape[1] or col > Xtrain2_max.shape[1]:
                raise ValueError(f"Requested {col} features, but only {Xtrain1_mean.shape[1]} or {Xtrain2_max.shape[1]} available.")
            # Concatenate selected feature subsets for train and test
            X_train_subset = pd.concat([Xtrain1_mean.iloc[:, :col], Xtrain2_max.iloc[:, :col]], axis=1)
            y_train = ytrain1
            X_test_subset = pd.concat([Xtest1_mean.iloc[:, :col], Xtest2_max.iloc[:, :col]], axis=1)
            y_test = ytest1
            # Data validation: check for missing values
            if X_train_subset.isnull().any().any() or X_test_subset.isnull().any().any():
                raise ValueError("Train or test feature subset contains missing values.")
            # Define pipeline and grid search
            pipeline = Pipeline([('model', RandomForestClassifier(random_state=random_seed))])
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            print(f"Starting GridSearchCV for {col} features...")
            grid_search.fit(X_train_subset, y_train)
            print(f"Finished GridSearchCV for {col} features.")
            # Store results
            results[i][subset_idx] = {
                'cv_results': grid_search.cv_results_,
                'best_model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'test_score': grid_search.score(X_test_subset, y_test),
                'best_score': grid_search.best_score_
            }
    return results

# -----------------------------------------------------------------------------
# Function: run_pipeline
# Main pipeline: performs sampling, CC computation, and classification.
#
# Args:
#   X_sel (pd.DataFrame): DataFrame with selected features and 'Subject_ID'.
#   scaler (sklearn Scaler): Scaler for standardization.
#   feature_nb (list): List of feature counts to use for subsets.
#   random_seed (int): Seed for reproducibility.
#
# Returns:
#   results (dict): Dictionary containing classification results for each test size and feature subset.
# -----------------------------------------------------------------------------
def run_pipeline(X_sel, scaler, feature_nb, random_seed):
    """
    Main pipeline: sampling, CC computation, and classification.
    """
    # Data validation: check for required columns and missing values
    if 'Subject_ID' not in X_sel.columns:
        raise ValueError("Input DataFrame must contain a 'Subject_ID' column.")
    if X_sel.drop('Subject_ID', axis=1).isnull().any().any():
        raise ValueError("Input DataFrame contains missing values in feature columns.")
    if len(feature_nb) == 0:
        raise ValueError("feature_nb list must not be empty.")
    Test = np.linspace(0.6, 0.2, num=3)  # Different test sizes
    CC_X_dict = {}
    for i, test_sz in enumerate(Test):
        print(f"Test size {test_sz} ({i+1}/{len(Test)})")
        CC1_X_list, CC2_min_X_list, CC2_max_X_list = [], [], []
        CC1_Xt_list, CC2_min_Xt_list, CC2_max_Xt_list = [], [], []
        # Repeat sampling and CC computation for robustness
        for iteration in tqdm(range(10), desc=f"Test size {test_sz}"):
            X_sel_split, supervisor_beats = sample_subject_data(
                X_sel, num_beats=110, num_supervisor_beats=10, random_seed=random_seed
            )
            CC1_X, CC2_min_X, CC2_max_X, CC1_Xt, CC2_min_Xt, CC2_max_Xt = compute_cc_features(
                X_sel_split, supervisor_beats, scaler, test_sz, random_seed
            )
            CC1_X_list.append(CC1_X)
            CC2_min_X_list.append(CC2_min_X)
            CC2_max_X_list.append(CC2_max_X)
            CC1_Xt_list.append(CC1_Xt)
            CC2_min_Xt_list.append(CC2_min_Xt)
            CC2_max_Xt_list.append(CC2_max_Xt)
        # Concatenate results from all iterations
        CC1_X_final = pd.concat(CC1_X_list, ignore_index=True)
        CC2_min_X_final = pd.concat(CC2_min_X_list, ignore_index=True)
        CC2_max_X_final = pd.concat(CC2_max_X_list, ignore_index=True)
        CC1_Xt_final = pd.concat(CC1_Xt_list, ignore_index=True)
        CC2_min_Xt_final = pd.concat(CC2_min_Xt_list, ignore_index=True)
        CC2_max_Xt_final = pd.concat(CC2_max_Xt_list, ignore_index=True)
        # Store all CC features for this test size
        CC_X_dict[i] = {
            'CC1_X_final': CC1_X_final,
            'CC2_min_X_final': CC2_min_X_final,
            'CC2_max_X_final': CC2_max_X_final,
            'CC1_Xt_final': CC1_Xt_final,
            'CC2_min_Xt_final': CC2_min_Xt_final,
            'CC2_max_Xt_final': CC2_max_Xt_final
        }
    # Define hyperparameter grid for CatBoost
    param_grid = [{
        'model': [CatBoostClassifier(verbose=0)],
        'model__iterations': [100, 200, 300],
        'model__learning_rate': [0.01 , 0.3, 0.5]
    }]
    # Run classification for all test sizes and feature subsets
    results = run_classification(CC_X_dict, feature_nb, param_grid, random_seed)
    return results