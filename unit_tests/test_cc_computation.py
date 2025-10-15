import pandas as pd
import numpy as np
import pytest
from src.cc_computation import CC

def test_CC_basic(small_df):
    # build simple supervisor: first sample of each subject
    sup_list = []
    for s, g in small_df.groupby("Subject_ID"):
        vals = g.drop(columns="Subject_ID").iloc[:1].values
        sup = pd.DataFrame(vals, columns=g.drop(columns="Subject_ID").columns)
        sup["Subject_ID"] = s
        sup_list.append(sup)
    supervisor = pd.concat(sup_list, ignore_index=True)
    CC1, CC2_min, CC2_max = CC(small_df, supervisor)
    assert "Subject_ID" in CC1.columns
    assert CC1.shape[0] > 0
    assert CC1.select_dtypes(include=[np.number]).notnull().all().all()

def test_CC_missing_supervisor_raises(small_df):
    # supervisor with no matching subject
    sup = pd.DataFrame({"F_1":[0.0], "F_2":[0.0], "F_3":[0.0], "F_4":[0.0], "F_5":[0.0], "Subject_ID":[999]})
    with pytest.raises(ValueError):
        CC(small_df, sup)

def test_CC_input_type_validation(small_df):
    with pytest.raises(ValueError):
        CC("not a df", small_df)
    with pytest.raises(ValueError):
        CC(small_df, "not a df")