import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

# Ensure project root is on sys.path so "import src.*" works when running tests individually
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture
def small_df():
    # small synthetic dataset: 3 subjects, 20 beats each, 5 features
    num_subjects = 3
    beats_per_subject = 20
    num_features = 5
    data = np.arange(num_subjects * beats_per_subject * num_features).reshape(
        num_subjects * beats_per_subject, num_features
    ).astype(float)
    df = pd.DataFrame(data, columns=[f"F_{i+1}" for i in range(num_features)])
    df["Subject_ID"] = np.repeat(np.arange(1, num_subjects+1), beats_per_subject)
    return df

@pytest.fixture
def tmp_csv(tmp_path: Path, small_df):
    p = tmp_path / "features.csv"
    # write csv without header (match load_and_prepare_data expectation)
    small_df.drop(columns=["Subject_ID"]).to_csv(p, index=False, header=False)
    return str(p)

@pytest.fixture
def scaler():
    return StandardScaler()