"""
Step 3 (part C): Assemble the full feature matrix used for classification.

Joins the cleaned ClinVar variant table (Step 2 output) with:
  - physicochemical delta features (fully implemented, offline)
  - conservation + domain-membership features (domain membership works
    once UniProt data is fetched; conservation score is a placeholder
    until a real source is wired in — see conservation_and_domain.py)

Usage:
    python -m src.features.build_feature_matrix
"""
import json
import dataclasses
import pandas as pd

from src.config import PROCESSED_DIR, FEATURES_DIR, RAW_DIR, ensure_dirs
from src.features.physicochemical import compute_physchem_features
from src.features.conservation_and_domain import compute_conservation_and_domain_features


def load_uniprot_features_by_gene() -> dict:
    path = RAW_DIR / "uniprot_entries.json"
    if not path.exists():
        print(f"[warn] {path} not found — domain-membership features will be empty. "
              f"Run `python -m src.data_collection.fetch_uniprot` first.")
        return {}
    with open(path) as f:
        entries = json.load(f)
    return {e["gene_symbol"]: e["features"] for e in entries}


def build_feature_row(row: pd.Series, uniprot_features_by_gene: dict) -> dict:
    physchem = compute_physchem_features(row["wt_aa"], row["mut_aa"])
    gene_features = uniprot_features_by_gene.get(row["gene_symbol"], [])
    cons = compute_conservation_and_domain_features(
    row["gene_symbol"],
    row["position"],
    gene_features
)

    features = {
        "gene_symbol": row["gene_symbol"],
        "panel": row.get("panel"),
        "position": row["position"],
        "wt_aa": row["wt_aa"],
        "mut_aa": row["mut_aa"],
        "label": row["label"],
        **dataclasses.asdict(physchem),
        **dataclasses.asdict(cons),
    }
    return features


def main():
    ensure_dirs()
    labelled_path = PROCESSED_DIR / "variants_labelled.csv"
    if not labelled_path.exists():
        raise FileNotFoundError(
            f"{labelled_path} not found — run `python -m src.preprocessing.clean_variants` first."
        )

    variants = pd.read_csv(labelled_path)
    uniprot_features_by_gene = load_uniprot_features_by_gene()

    rows = [build_feature_row(r, uniprot_features_by_gene) for _, r in variants.iterrows()]
    feature_df = pd.DataFrame(rows)

    out_path = FEATURES_DIR / "feature_matrix.csv"
    feature_df.to_csv(out_path, index=False)
    print(f"Saved feature matrix with {len(feature_df)} rows, {feature_df.shape[1]} columns to {out_path}")
    print("\nColumns:", list(feature_df.columns))


if __name__ == "__main__":
    main()
