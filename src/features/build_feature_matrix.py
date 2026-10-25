"""
Step 3 (part C): Assemble the feature matrices used for classification.

Joins the cleaned ClinVar variant tables (Step 2 output) with:
  - physicochemical delta features and the BLOSUM62 substitution score
  - relative position along the protein (position / protein length)
  - conservation score (build_conservation_scores.py) and domain membership
  - AlphaFold structural context: pLDDT, relative solvent accessibility and
    local contact count (structural_features.py)
  - the AlphaMissense score, kept as a comparator column only

Two matrices are written: the labelled set used for training/evaluation and
the VUS set, which is scored by the final model.

Usage:
    python -m src.features.build_feature_matrix
"""
import json
import dataclasses
import pandas as pd
from Bio.Align import substitution_matrices

from src.config import PROCESSED_DIR, FEATURES_DIR, RAW_DIR, ensure_dirs
from src.features.physicochemical import compute_physchem_features
from src.features.conservation_and_domain import (
    compute_conservation_and_domain_features,
    load_conservation_scores,
)

BLOSUM62 = substitution_matrices.load("BLOSUM62")
PASSTHROUGH_COLUMNS = ["clinvar_uid", "clinical_significance", "review_status", "review_stars", "position_verified"]


def load_uniprot_entries() -> dict:
    path = RAW_DIR / "uniprot_entries.json"
    if not path.exists():
        print(f"[warn] {path} not found — domain-membership features will be empty. "
              f"Run `python -m src.data_collection.fetch_uniprot` first.")
        return {}
    with open(path) as f:
        return {e["gene_symbol"]: e for e in json.load(f)}


def load_json(name: str) -> dict:
    path = RAW_DIR / name
    if not path.exists():
        print(f"[warn] {path} not found — related features will be empty.")
        return {}
    with open(path) as f:
        return json.load(f)


def load_alphamissense() -> dict:
    path = RAW_DIR / "alphamissense_panel.tsv"
    if not path.exists():
        print(f"[warn] {path} not found — AlphaMissense comparator column will be empty.")
        return {}
    am = pd.read_csv(path, sep="\t")
    return dict(zip(am["uniprot_id"] + ":" + am["protein_variant"], am["am_pathogenicity"]))


def build_feature_row(row: pd.Series, ctx: dict) -> dict:
    gene = row["gene_symbol"]
    position = int(row["position"])
    entry = ctx["uniprot"].get(gene, {})
    physchem = compute_physchem_features(row["wt_aa"], row["mut_aa"])
    cons = compute_conservation_and_domain_features(
        gene, position, entry.get("features", []), ctx["conservation"]
    )
    struct = ctx["structure"].get(gene, {}).get(str(position), {})
    seq_len = entry.get("sequence_length")
    am_key = f"{entry.get('uniprot_id')}:{row['wt_aa']}{position}{row['mut_aa']}"

    features = {
        "gene_symbol": gene,
        "panel": row.get("panel"),
        "position": position,
        "relative_position": round(position / seq_len, 4) if seq_len else None,
        "wt_aa": row["wt_aa"],
        "mut_aa": row["mut_aa"],
        "label": row["label"],
        **dataclasses.asdict(physchem),
        "blosum62": float(BLOSUM62[row["wt_aa"]][row["mut_aa"]]),
        **dataclasses.asdict(cons),
        "plddt": struct.get("plddt"),
        "rsa": struct.get("rsa"),
        "contact_count": struct.get("contact_count"),
        "alphamissense": ctx["alphamissense"].get(am_key),
    }
    for col in PASSTHROUGH_COLUMNS:
        if col in row:
            features[col] = row[col]
    return features


def build(variants: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    return pd.DataFrame([build_feature_row(r, ctx) for _, r in variants.iterrows()])


def main():
    ensure_dirs()
    labelled_path = PROCESSED_DIR / "variants_labelled.csv"
    if not labelled_path.exists():
        raise FileNotFoundError(
            f"{labelled_path} not found — run `python -m src.preprocessing.clean_variants` first."
        )

    ctx = {
        "uniprot": load_uniprot_entries(),
        "conservation": load_conservation_scores(),
        "structure": load_json("structure_features.json"),
        "alphamissense": load_alphamissense(),
    }

    for name, src in [("feature_matrix.csv", labelled_path), ("feature_matrix_vus.csv", PROCESSED_DIR / "variants_vus.csv")]:
        if not src.exists():
            continue
        feature_df = build(pd.read_csv(src), ctx)
        out_path = FEATURES_DIR / name
        feature_df.to_csv(out_path, index=False)
        print(f"Saved {out_path.name}: {len(feature_df)} rows, {feature_df.shape[1]} columns")
        coverage = feature_df[["conservation_score", "plddt", "alphamissense"]].notna().mean().round(3)
        print("  coverage:", coverage.to_dict())


if __name__ == "__main__":
    main()
