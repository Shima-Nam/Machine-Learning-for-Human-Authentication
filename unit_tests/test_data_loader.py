import os
import pandas as pd
import numpy as np
import pytest
from src.data_loader import load_and_prepare_data

def test_csv_column_count_mismatch(tmp_path):
    p = tmp_path / "bad.csv"
    # write CSV with 3 columns but we expect 5
    pd.DataFrame(np.zeros((6, 3))).to_csv(p, index=False, header=False)
    with pytest.raises(ValueError):
        load_and_prepare_data(str(p), num_features=5, num_subjects=2, beats_per_subject=3, random_seed=0)

def test_row_count_mismatch(tmp_path):
    # 2 subjects * 3 beats expected => 6 rows, give 5 rows
    p = tmp_path / "bad_rows.csv"
    pd.DataFrame(np.zeros((5, 4))).to_csv(p, index=False, header=False)
    with pytest.raises(ValueError):
        load_and_prepare_data(str(p), num_features=4, num_subjects=2, beats_per_subject=3, random_seed=0)

def test_shuffle_reproducible(tmp_csv):
    # same seed should produce same shuffle
    df1 = load_and_prepare_data(csv_path=tmp_csv, num_features=5, num_subjects=3, beats_per_subject=20, random_seed=1)
    df2 = load_and_prepare_data(csv_path=tmp_csv, num_features=5, num_subjects=3, beats_per_subject=20, random_seed=1)
    assert df1.equals(df2)

def test_load_and_prepare_data_basic(tmp_csv):
    df = load_and_prepare_data(
        csv_path=tmp_csv,
        num_features=5,
        num_subjects=3,
        beats_per_subject=20,
        random_seed=42,
    )
    assert "Subject_ID" in df.columns
    assert df.shape == (3 * 20, 6)  # 5 features + Subject_ID
    assert set(df["Subject_ID"].unique()) == {1, 2, 3}

def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_and_prepare_data("nonexistent.csv", num_features=1, num_subjects=1, beats_per_subject=1, random_seed=0)