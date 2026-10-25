"""
Step 4: Variant classification and evaluation.

Trains a Random Forest and an XGBoost classifier to separate pathogenic from
benign missense variants and evaluates them in ways that expose gene-level
leakage:

  1. Leave-one-gene-out (LOGO) cross-validation — the primary evaluation.
     Each gene is predicted by a model that never saw any variant from it.
  2. A single gene-level hold-out split (GroupShuffleSplit, seed 42), kept
     for continuity with the first version of the project.
  3. Random stratified 5-fold CV, which lets variants from the same gene
     sit on both sides of the split. Reported only to show how much a
     leaky split inflates performance.

Also reported: feature-set ablations, a sensitivity analysis restricted to
ClinVar records with at least one review star, single-feature baselines,
and AlphaMissense on exactly the same variants as an external comparator.
Class imbalance is handled with class weighting rather than SMOTE so that no
synthetic variants are created.

Outputs (outputs/):
  metrics.json, logo_predictions.csv, model_random_forest.joblib,
  model_xgboost.joblib, vus_scores.csv

Usage:
    python -m src.modeling.train_classifier
"""
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import FEATURES_DIR, OUTPUTS_DIR, ensure_dirs
from src.modeling.evaluate import classification_metrics, bootstrap_ci

PHYSCHEM_FEATURES = [
    "delta_hydrophobicity",
    "delta_molecular_weight",
    "delta_volume",
    "delta_charge",
    "blosum62",
]
CONTEXT_FEATURES = ["relative_position", "conservation_score"]
STRUCTURE_FEATURES = ["plddt", "rsa", "contact_count"]
NUMERIC_FEATURES = PHYSCHEM_FEATURES + CONTEXT_FEATURES + STRUCTURE_FEATURES
BOOLEAN_FEATURES = [
    "polarity_change",
    "wt_is_proline",
    "mut_is_proline",
    "wt_is_glycine",
    "mut_is_glycine",
    "in_annotated_domain",
]
CATEGORICAL_FEATURES = ["panel"]

# Feature sets compared in the ablation study. "v1" reproduces the inputs of
# the first version of the pipeline (raw residue number, no BLOSUM62 or
# structural context).
FEATURE_SETS = {
    "v1 (raw position)": (
        ["position", "delta_hydrophobicity", "delta_molecular_weight", "delta_volume",
         "delta_charge", "conservation_score"],
        BOOLEAN_FEATURES,
        CATEGORICAL_FEATURES,
    ),
    "Physicochemical only": (PHYSCHEM_FEATURES, BOOLEAN_FEATURES[:5], []),
    "+ conservation & position": (PHYSCHEM_FEATURES + CONTEXT_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES),
    "Full (+ structure)": (NUMERIC_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES),
}
FULL_SET = "Full (+ structure)"
SEED = 42


def load_feature_matrix(name: str = "feature_matrix.csv") -> pd.DataFrame:
    path = FEATURES_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.features.build_feature_matrix` first."
        )
    return pd.read_csv(path)


def build_preprocessor(numeric, boolean, categorical) -> ColumnTransformer:
    transformers = [("num", SimpleImputer(strategy="median"), list(numeric) + list(boolean))]
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)))
    return ColumnTransformer(transformers=transformers)


def make_model(kind: str, feature_set: tuple, y_train: pd.Series) -> Pipeline:
    if kind == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",  # addresses class imbalance
            random_state=SEED,
            n_jobs=-1,
        )
    else:
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        clf = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=n_neg / max(n_pos, 1),  # addresses class imbalance
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        )
    return Pipeline([("preprocess", build_preprocessor(*feature_set)), ("clf", clf)])


def columns(feature_set: tuple) -> list[str]:
    return [c for group in feature_set for c in group]


def cross_val_predict(df, kind, feature_set, splitter, groups=None) -> np.ndarray:
    X, y = df[columns(feature_set)], df["y"]
    proba = np.full(len(df), np.nan)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = make_model(kind, feature_set, y.iloc[train_idx])
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return proba


def summarise(y, proba, with_ci=True) -> dict:
    out = classification_metrics(y, proba)
    if with_ci:
        out["roc_auc_ci"] = bootstrap_ci(y, proba, roc_auc_score)
        out["pr_auc_ci"] = bootstrap_ci(y, proba, average_precision_score)
    return out


def per_gene_auc(df, proba) -> dict:
    out = {}
    for gene, idx in df.groupby("gene_symbol").groups.items():
        y = df.loc[idx, "y"]
        if y.nunique() == 2 and min((y == 0).sum(), (y == 1).sum()) >= 3:
            out[gene] = {
                "n": int(len(y)),
                "n_pathogenic": int(y.sum()),
                "roc_auc": float(roc_auc_score(y, proba[idx])),
            }
    return out


def main():
    ensure_dirs()
    df = load_feature_matrix()
    df = df[df["label"].isin(["pathogenic", "benign"])].reset_index(drop=True)
    df["y"] = (df["label"] == "pathogenic").astype(int)
    groups = df["gene_symbol"]
    full = FEATURE_SETS[FULL_SET]
    results = {
        "n_variants": int(len(df)),
        "n_pathogenic": int(df["y"].sum()),
        "n_benign": int((df["y"] == 0).sum()),
        "n_genes": int(groups.nunique()),
    }
    predictions = df[["clinvar_uid", "gene_symbol", "wt_aa", "position", "mut_aa", "label", "y"]].copy()

    # 1. Leave-one-gene-out CV (primary)
    print("\n=== Leave-one-gene-out CV ===")
    results["logo"] = {}
    for kind in ("RandomForest", "XGBoost"):
        proba = cross_val_predict(df, kind, full, LeaveOneGroupOut(), groups)
        predictions[f"logo_{kind}"] = proba
        results["logo"][kind] = summarise(df["y"], proba)
        results["logo"][kind]["per_gene"] = per_gene_auc(df, proba)
        m = results["logo"][kind]
        print(f"{kind:12s} ROC-AUC {m['roc_auc']:.3f} {m['roc_auc_ci']}  PR-AUC {m['pr_auc']:.3f}  "
              f"bal.acc {m['balanced_accuracy']:.3f}")

    # External comparator and single-feature baselines on the same variants
    results["comparators"] = {}
    am_mask = df["alphamissense"].notna()
    results["comparators"]["AlphaMissense"] = {
        "n": int(am_mask.sum()),
        **summarise(df.loc[am_mask, "y"], df.loc[am_mask, "alphamissense"].to_numpy()),
    }
    for kind in ("RandomForest", "XGBoost"):
        results["comparators"][f"{kind} (same variants)"] = summarise(
            df.loc[am_mask, "y"], predictions.loc[am_mask, f"logo_{kind}"].to_numpy()
        )
    baselines = {
        "BLOSUM62 (lower = more severe)": -df["blosum62"],
        "Conservation score": df["conservation_score"],
        "|Δ charge|": df["delta_charge"].abs(),
        "Burial (1 - RSA)": 1 - df["rsa"],
    }
    for name, score in baselines.items():
        mask = score.notna()
        results["comparators"][name] = {
            "n": int(mask.sum()),
            "roc_auc": float(roc_auc_score(df.loc[mask, "y"], score[mask])),
            "roc_auc_ci": bootstrap_ci(df.loc[mask, "y"], score[mask].to_numpy(), roc_auc_score),
        }
    predictions["alphamissense"] = df["alphamissense"]

    # 2. Ablation of feature sets under LOGO
    print("\n=== Feature-set ablation (LOGO) ===")
    results["ablation"] = {}
    for name, fs in FEATURE_SETS.items():
        results["ablation"][name] = {}
        for kind in ("RandomForest", "XGBoost"):
            proba = cross_val_predict(df, kind, fs, LeaveOneGroupOut(), groups)
            results["ablation"][name][kind] = summarise(df["y"], proba, with_ci=False)
            print(f"{name:28s} {kind:12s} ROC-AUC {results['ablation'][name][kind]['roc_auc']:.3f}")

    # 3. Random (leaky) 5-fold CV for contrast
    print("\n=== Random stratified 5-fold CV (leaky) ===")
    results["random_cv"] = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name in ("v1 (raw position)", FULL_SET):
        results["random_cv"][name] = {}
        for kind in ("RandomForest", "XGBoost"):
            proba = cross_val_predict(df, kind, FEATURE_SETS[name], skf)
            results["random_cv"][name][kind] = summarise(df["y"], proba, with_ci=False)
            print(f"{name:28s} {kind:12s} ROC-AUC {results['random_cv'][name][kind]['roc_auc']:.3f}")

    # 4. Single gene-level hold-out split (continuity with v1)
    print("\n=== Gene-level hold-out split ===")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr, te = next(splitter.split(df, groups=groups))
    results["holdout"] = {
        "test_genes": sorted(df.iloc[te]["gene_symbol"].unique().tolist()),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "test_pathogenic": int(df.iloc[te]["y"].sum()),
    }
    for kind in ("RandomForest", "XGBoost"):
        model = make_model(kind, full, df.iloc[tr]["y"])
        model.fit(df.iloc[tr][columns(full)], df.iloc[tr]["y"])
        proba = model.predict_proba(df.iloc[te][columns(full)])[:, 1]
        results["holdout"][kind] = summarise(df.iloc[te]["y"], proba)
        print(f"{kind:12s} ROC-AUC {results['holdout'][kind]['roc_auc']:.3f}")

    # 5. Sensitivity: labels with >= 1 review star only
    starred = df[df["review_stars"] >= 1].reset_index(drop=True)
    results["sensitivity_1star"] = {"n": int(len(starred)), "n_pathogenic": int(starred["y"].sum())}
    for kind in ("RandomForest", "XGBoost"):
        proba = cross_val_predict(starred, kind, full, LeaveOneGroupOut(), starred["gene_symbol"])
        results["sensitivity_1star"][kind] = summarise(starred["y"], proba)

    # Final models on all labelled data -> SHAP and VUS scoring
    best = max(("RandomForest", "XGBoost"), key=lambda k: results["logo"][k]["roc_auc"])
    results["best_model"] = best
    for kind, fname in (("RandomForest", "model_random_forest.joblib"), ("XGBoost", "model_xgboost.joblib")):
        model = make_model(kind, full, df["y"])
        model.fit(df[columns(full)], df["y"])
        joblib.dump(model, OUTPUTS_DIR / fname)

    vus_path = FEATURES_DIR / "feature_matrix_vus.csv"
    if vus_path.exists():
        vus = pd.read_csv(vus_path)
        model = joblib.load(OUTPUTS_DIR / ("model_xgboost.joblib" if best == "XGBoost" else "model_random_forest.joblib"))
        vus["model_score"] = model.predict_proba(vus[columns(full)])[:, 1]
        vus.to_csv(OUTPUTS_DIR / "vus_scores.csv", index=False)
        both = vus["alphamissense"].notna()
        results["vus"] = {
            "n": int(len(vus)),
            "model_score_ge_0.5": int((vus["model_score"] >= 0.5).sum()),
            "spearman_with_alphamissense": float(
                vus.loc[both, ["model_score", "alphamissense"]].corr(method="spearman").iloc[0, 1]
            ),
        }

    predictions.to_csv(OUTPUTS_DIR / "logo_predictions.csv", index=False)
    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBest model under LOGO: {best}. Metrics saved to {OUTPUTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
