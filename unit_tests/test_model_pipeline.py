import pandas as pd
import pytest
from src.model_pipeline import sample_subject_data, compute_cc_features
from sklearn.preprocessing import StandardScaler

def test_sample_subject_data_insufficient_samples(small_df):
    X_sel = small_df.copy()
    # request more beats than available -> should raise
    with pytest.raises(ValueError):
        sample_subject_data(X_sel, num_beats=1000, num_supervisor_beats=10, random_seed=0)

def test_compute_cc_features_stratify_error(small_df):
    # create X_sel_split with only one subject -> stratify will fail
    single = small_df[small_df["Subject_ID"] == 1].reset_index(drop=True)
    supervisor = single.iloc[:2].reset_index(drop=True)
    scaler = StandardScaler()
    with pytest.raises(ValueError):
        compute_cc_features(single, supervisor, scaler, test_sz=0.5, random_seed=0)

def test_sample_and_compute_cc(small_df):
    X_sel = small_df.copy()
    # use small numbers for speed
    X_split, supervisor = sample_subject_data(X_sel, num_beats=10, num_supervisor_beats=2, random_seed=0)
    assert "Subject_ID" in X_split.columns and "Subject_ID" in supervisor.columns
    scaler = StandardScaler()
    CCs = compute_cc_features(X_split, supervisor, scaler, test_sz=0.3, random_seed=0)
    assert len(CCs) == 6
    for df in CCs:
        assert "Subject_ID" in df.columns