"""
Step 5: Explainable AI via SHAP.

Produces, for the best model under leave-one-gene-out CV (see metrics.json):
  1. A global beeswarm summary plot and a mean |SHAP| bar chart.
  2. Dependence plots for the most important features.
  3. Waterfall explanations for selected individual variants, which feed
     the integrated variant report (Step 8).

SHAP values are computed with TreeExplainer on the final model trained on
all labelled variants, so they describe what that model learned.

Usage:
    python -m src.explainability.shap_explain
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import OUTPUTS_DIR, FEATURES_DIR  # noqa: E402
from src.modeling.train_classifier import FEATURE_SETS, FULL_SET, columns  # noqa: E402

MODEL_FILES = {"RandomForest": "model_random_forest.joblib", "XGBoost": "model_xgboost.joblib"}
READABLE = {
    "relative_position": "Relative position",
    "delta_hydrophobicity": "Δ hydrophobicity",
    "delta_molecular_weight": "Δ molecular weight",
    "delta_volume": "Δ side-chain volume",
    "delta_charge": "Δ charge",
    "blosum62": "BLOSUM62 score",
    "conservation_score": "Conservation",
    "plddt": "AlphaFold pLDDT",
    "rsa": "Relative solvent accessibility",
    "contact_count": "Cα contacts (10 Å)",
    "polarity_change": "Polarity change",
    "wt_is_proline": "WT is proline",
    "mut_is_proline": "Mutant is proline",
    "wt_is_glycine": "WT is glycine",
    "mut_is_glycine": "Mutant is glycine",
    "in_annotated_domain": "In annotated domain",
    "panel_arrhythmia": "Panel: arrhythmia",
    "panel_cardiomyopathy": "Panel: cardiomyopathy",
}
# Variants explained individually, if present in the labelled set.
EXAMPLE_VARIANTS = [("MYH7", "R", 403, "Q"), ("KCNQ1", "R", 243, "H"), ("ACTC1", "E", 101, "K"), ("ACTC1", "R", 39, "H")]


def best_model_name() -> str:
    path = OUTPUTS_DIR / "metrics.json"
    if path.exists():
        return json.load(open(path)).get("best_model", "XGBoost")
    return "XGBoost"


def readable_names(preprocess) -> list[str]:
    names = []
    for raw in preprocess.get_feature_names_out():
        name = raw.split("__", 1)[-1]
        names.append(READABLE.get(name, name))
    return names


def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    preprocess = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]
    X_t = preprocess.transform(X)
    explanation = shap.TreeExplainer(clf)(X_t)
    if explanation.values.ndim == 3:  # RandomForest: one set of values per class
        explanation = explanation[:, :, 1]
    explanation.feature_names = readable_names(preprocess)
    return explanation


def explain_single_variant(explanation: shap.Explanation, index: int, top_k: int = 5) -> list[dict]:
    """Top-k features (by |SHAP|) behind one prediction, for the report."""
    order = np.argsort(-np.abs(explanation.values[index]))[:top_k]
    return [
        {
            "feature": explanation.feature_names[i],
            "shap_value": float(explanation.values[index, i]),
            "feature_value": float(explanation.data[index, i]),
        }
        for i in order
    ]


def main():
    name = best_model_name()
    model = joblib.load(OUTPUTS_DIR / MODEL_FILES[name])
    df = pd.read_csv(FEATURES_DIR / "feature_matrix.csv")
    df = df[df["label"].isin(["pathogenic", "benign"])].reset_index(drop=True)
    X = df[columns(FEATURE_SETS[FULL_SET])]
    explanation = compute_shap_values(model, X)

    plt.figure()
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title(f"SHAP summary — {name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_global_summary.png", dpi=200)
    plt.close()

    plt.figure()
    shap.plots.bar(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_bar.png", dpi=200)
    plt.close()

    importance = pd.Series(np.abs(explanation.values).mean(0), index=explanation.feature_names)
    importance.sort_values(ascending=False).to_csv(OUTPUTS_DIR / "shap_importance.csv", header=["mean_abs_shap"])
    print("Mean |SHAP| (top 10):")
    print(importance.sort_values(ascending=False).head(10).round(4).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, feature in zip(axes, importance.sort_values(ascending=False).index[:3]):
        shap.plots.scatter(explanation[:, feature], ax=ax, show=False, color="#3E6E8E")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_dependence.png", dpi=200)
    plt.close()

    proba = model.predict_proba(X)[:, 1]
    examples = []
    for gene, wt, pos, mut in EXAMPLE_VARIANTS:
        hit = df.index[(df.gene_symbol == gene) & (df.wt_aa == wt) & (df.position == pos) & (df.mut_aa == mut)]
        if len(hit) == 0:
            print(f"[info] example {gene} {wt}{pos}{mut} not in labelled set")
            continue
        i = int(hit[0])
        label = f"{gene}_{wt}{pos}{mut}"
        plt.figure()
        shap.plots.waterfall(explanation[i], max_display=10, show=False)
        plt.title(f"{gene} p.{wt}{pos}{mut} (ClinVar: {df.loc[i, 'label']}, model: {proba[i]:.2f})", fontsize=10)
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / f"shap_waterfall_{label}.png", dpi=200)
        plt.close()
        examples.append({
            "variant": label,
            "clinvar_label": df.loc[i, "label"],
            "model_score": float(proba[i]),
            "top_features": explain_single_variant(explanation, i),
        })
    with open(OUTPUTS_DIR / "shap_examples.json", "w") as f:
        json.dump({"model": name, "base_value": float(np.mean(explanation.base_values)), "examples": examples}, f, indent=2)
    print(f"Saved SHAP outputs for {name} to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
