"""
Step 4: Variant classification.

Trains a Random Forest baseline and an XGBoost classifier to distinguish
pathogenic from benign variants, using:

  - GENE-LEVEL train/test split (GroupKFold on gene_symbol) rather than a
    random row-level split, so the test set contains genes unseen during
    training — a more realistic estimate of generalization (see
    methodology Section 5.4 / literature on CardioBoost).
  - Class weighting to address ClinVar's typical pathogenic/benign
    imbalance (an alternative SMOTE path is included but commented out —
    prefer class weighting first since it doesn't fabricate synthetic
    variants).

Usage:
    python -m src.modeling.train_classifier
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

from src.config import FEATURES_DIR, OUTPUTS_DIR, ensure_dirs
from src.modeling.evaluate import evaluate_classifier

NUMERIC_FEATURES = [
    "position",
    "delta_hydrophobicity",
    "delta_molecular_weight",
    "delta_volume",
    "delta_charge",
    "conservation_score",
]
BOOLEAN_FEATURES = [
    "polarity_change",
    "wt_is_proline",
    "mut_is_proline",
    "wt_is_glycine",
    "mut_is_glycine",
    "in_annotated_domain",
]
CATEGORICAL_FEATURES = ["panel"]
# conservation_score is included only once a real conservation source is
# wired in (see conservation_and_domain.py); it's all-null placeholder
# data until then, so it's left out of NUMERIC_FEATURES for now.


def load_feature_matrix() -> pd.DataFrame:
    path = FEATURES_DIR / "feature_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.features.build_feature_matrix` first."
        )
    return pd.read_csv(path)


def gene_level_split(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["gene_symbol"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES + BOOLEAN_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def train_random_forest(X_train, y_train) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    class_weight="balanced",  # addresses class imbalance
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train) -> Pipeline:
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)  # addresses class imbalance

    pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "clf",
                xgb.XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def main():
    ensure_dirs()
    df = load_feature_matrix()
    df = df.dropna(
    subset=[
        f for f in NUMERIC_FEATURES + BOOLEAN_FEATURES
        if f != "conservation_score"
    ] + ["label"]
)
    df["y"] = (df["label"] == "pathogenic").astype(int)

    train_df, test_df = gene_level_split(df)
    print(f"Train genes: {train_df['gene_symbol'].nunique()}, rows: {len(train_df)}")
    print(f"Test genes:  {test_df['gene_symbol'].nunique()}, rows: {len(test_df)}")

    feature_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    X_train, y_train = train_df[feature_cols], train_df["y"]
    X_test, y_test = test_df[feature_cols], test_df["y"]

    print("\n--- Random Forest ---")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_classifier(rf_model, X_test, y_test, name="RandomForest")

    print("\n--- XGBoost ---")
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_classifier(xgb_model, X_test, y_test, name="XGBoost")

    joblib.dump(rf_model, OUTPUTS_DIR / "model_random_forest.joblib")
    joblib.dump(xgb_model, OUTPUTS_DIR / "model_xgboost.joblib")
    print(f"\nModels saved to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
