"""
Step 5: Explainable AI via SHAP.

Produces:
  1. A global summary plot (which features matter most across the dataset).
  2. A per-variant force/waterfall-style explanation (which features drove
     one specific prediction) — this is what feeds into the final
     integrated report (Step 8).

Usage:
    python -m src.explainability.shap_explain
"""
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from src.config import OUTPUTS_DIR, FEATURES_DIR
from src.modeling.train_classifier import (
    NUMERIC_FEATURES,
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
)


def load_model_and_data(model_name: str = "model_xgboost.joblib"):
    model = joblib.load(OUTPUTS_DIR / model_name)
    df = pd.read_csv(FEATURES_DIR / "feature_matrix.csv")
    feature_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    df = df.dropna(subset=NUMERIC_FEATURES + BOOLEAN_FEATURES + ["label"])
    return model, df, feature_cols


def compute_shap_values(model, X: pd.DataFrame):
    """
    model is a sklearn Pipeline(preprocess, clf). SHAP's TreeExplainer needs
    the fitted estimator + the *transformed* feature matrix, and we keep
    the transformed feature names around for readable plots/labels.
    """
    preprocess = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    X_transformed = preprocess.transform(X)
    feature_names = preprocess.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_transformed)
    shap_values.feature_names = list(feature_names)
    return explainer, shap_values, X_transformed, feature_names


def global_summary_plot(shap_values, out_path):
    plt.figure()
    shap.summary_plot(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved global SHAP summary plot to {out_path}")


def explain_single_variant(shap_values, feature_names, X_row, index: int, top_k: int = 5) -> list[dict]:
    """
    Return the top_k features (by absolute SHAP value) driving the
    prediction for one variant — this is the structured explanation that
    feeds into the final integrated report (Step 8 / generate_report.py).
    """
    row_values = shap_values.values[index]
    contributions = sorted(
        zip(feature_names, row_values, X_row),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:top_k]
    return [
        {"feature": name, "shap_value": float(val), "feature_value": raw_val}
        for name, val, raw_val in contributions
    ]


def main():
    model, df, feature_cols = load_model_and_data()
    X = df[feature_cols]

    explainer, shap_values, X_transformed, feature_names = compute_shap_values(model, X)

    global_summary_plot(shap_values, OUTPUTS_DIR / "shap_global_summary.png")

    # Example: explain the first variant in the dataset
    explanation = explain_single_variant(shap_values, feature_names, X_transformed[0], index=0)
    print("\nExample per-variant explanation (row 0):")
    for item in explanation:
        print(f"  {item['feature']}: SHAP={item['shap_value']:.3f} (value={item['feature_value']})")


if __name__ == "__main__":
    main()
