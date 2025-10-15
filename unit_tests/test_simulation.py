import pytest
import pandas as pd
import numpy as np
from src.simulation import run_simulation

def test_run_simulation_basic(small_df):
    X_sel = small_df.copy()
    # use reduced iterations and small test_sizes for speed
    res = run_simulation(X_sel, test_sizes=[0.5], iterations=2, random_state=0)
    assert isinstance(res, dict)
    assert 0 in res
    keys = res[0].keys()
    expected = {'CC1_X_final','CC2_min_X_final','CC2_max_X_final','CC1_Xt_final','CC2_min_Xt_final','CC2_max_Xt_final'}
    assert expected.issubset(set(keys))