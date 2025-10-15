import pandas as pd
import numpy as np
import pytest
from src.feature_selection import select_features_dendrogram

def test_constant_feature_causes_error(small_df):
    df = small_df.copy()
    # make one feature constant -> spearman may produce NaNs
    df["F_1"] = 1.0
    with pytest.raises(ValueError):
        select_features_dendrogram(df, scaler=None, plot=False)

def test_returns_scaler_and_features(small_df):
    selected, scaler = select_features_dendrogram(small_df, scaler=None, plot=False)
    assert hasattr(scaler, "transform")
    assert len(selected) >= 1

def test_select_features_dendrogram_basic(small_df):
    selected_features, scaler = select_features_dendrogram(small_df, scaler=None, plot=False)
    assert len(selected_features) >= 1
    feature_cols = [c for c in small_df.columns if c != "Subject_ID"]
    for f in selected_features:
        assert f in feature_cols