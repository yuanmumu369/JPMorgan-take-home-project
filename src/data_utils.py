from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

TARGET_COL = "bad_flag"
ID_COL = "id"

def _smart_read_csv(path: str) -> pd.DataFrame:
    # Some files include a notice line before the real header.
    # We'll peek the first two lines and decide whether to skip the first.
    import io
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline()
        second = f.readline()
    if 'bad_flag' in second and 'bad_flag' not in first:
        return pd.read_csv(path, skiprows=1)
    else:
        return pd.read_csv(path)


def load_training_data(path: str) -> pd.DataFrame:
    df = _smart_read_csv(path)
    return df

def load_test_data(path: str) -> pd.DataFrame:
    df = _smart_read_csv(path)
    return df

def split_train_valid(df: pd.DataFrame, test_size: float=0.2, random_state: int=42):
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_valid, y_train, y_valid

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    # Identify feature types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Common loan metadata columns that are IDs/descriptions -> drop from modeling if present
    for drop_col in ["member_id", "desc"]:
        if drop_col in cat_cols:
            cat_cols.remove(drop_col)
        if drop_col in num_cols:
            num_cols.remove(drop_col)

    # Some columns may be effectively categorical but stored as object/str, we already capture.
    # Build transformers
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ]
    )
    return preprocessor, num_cols, cat_cols

def fit_preprocessor_save(preprocessor, X_train, path: str):
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, path)

def load_preprocessor(path: str):
    return joblib.load(path)