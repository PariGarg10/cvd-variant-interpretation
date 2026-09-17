"""
Step 2: Data Preprocessing.

- Removes duplicate / conflicting ClinVar entries.
- Splits variants into a confidently-labelled set (Pathogenic/Likely
  Pathogenic vs Benign/Likely Benign) for training, and a separate VUS
  (Variant of Uncertain Significance) set held out for later qualitative
  evaluation (per methodology Section 5.2 — VUS are not discarded, since
  they're the primary real-world use case for the tool).
- Extracts a clean amino-acid substitution (e.g. "p.Arg403Gln") from the
  ClinVar protein_change / title fields into structured wt/mut/position
  columns used downstream by feature engineering.

Usage:
    python -m src.preprocessing.clean_variants
"""
import re
import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR, ensure_dirs

PATHOGENIC_LABELS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN_LABELS = {"Benign", "Likely benign", "Benign/Likely benign"}
VUS_LABELS = {"Uncertain significance"}

# 3-letter -> 1-letter amino acid code map, used to parse ClinVar's
# p.Xxx###Yyy notation into a compact form.
AA_3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*",
}

PROTEIN_CHANGE_RE = re.compile(
    r"(?:p\.)?(?:(?P<wt3>[A-Za-z]{3})(?P<pos>\d+)(?P<mut3>[A-Za-z]{3})|(?P<wt1>[A-Z])(?P<pos1>\d+)(?P<mut1>[A-Z]))"
)


def parse_protein_change(text: str):
    """Extract (wt_aa, position, mut_aa) from ClinVar protein-change strings.

    Supports both:
        p.Arg403Gln
        Arg403Gln
        R403Q
    """
    if not isinstance(text, str):
        return None, None, None

    m = PROTEIN_CHANGE_RE.search(text)
    if not m:
        return None, None, None

    # Three-letter amino-acid notation
    if m.group("wt3"):
        wt1 = AA_3TO1.get(m.group("wt3"))
        mut1 = AA_3TO1.get(m.group("mut3"))
        pos = int(m.group("pos"))
        return wt1, pos, mut1

    # One-letter amino-acid notation
    wt1 = m.group("wt1")
    mut1 = m.group("mut1")
    pos = int(m.group("pos1"))

    return wt1, pos, mut1


def normalize_label(raw_significance: str) -> str:
    if not isinstance(raw_significance, str):
        return "unknown"
    sig = raw_significance.strip()
    if sig in PATHOGENIC_LABELS:
        return "pathogenic"
    if sig in BENIGN_LABELS:
        return "benign"
    if sig in VUS_LABELS:
        return "vus"
    return "other"  # conflicting interpretations, risk factor, drug response, etc.


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Deduplicate on gene + protein change (same substitution reported
    # multiple times across submissions collapses to one row).
    search_col = df["protein_change"].fillna(df["title"])
    df["wt_aa"], df["position"], df["mut_aa"] = zip(*search_col.map(parse_protein_change))
    df = df.dropna(subset=["wt_aa", "position", "mut_aa"])
    df["position"] = df["position"].astype(int)

    df["label"] = df["clinical_significance"].map(normalize_label)

    dedup_keys = ["gene_symbol", "wt_aa", "position", "mut_aa"]
    df = df.sort_values("review_status").drop_duplicates(subset=dedup_keys, keep="first")

    return df.reset_index(drop=True)


def main():
    ensure_dirs()
    raw_path = RAW_DIR / "clinvar_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found — run `python -m src.data_collection.fetch_clinvar` first."
        )

    df = pd.read_csv(raw_path)
    cleaned = clean(df)

    train_eval_set = cleaned[cleaned["label"].isin(["pathogenic", "benign"])].reset_index(drop=True)
    vus_set = cleaned[cleaned["label"] == "vus"].reset_index(drop=True)

    train_eval_set.to_csv(PROCESSED_DIR / "variants_labelled.csv", index=False)
    vus_set.to_csv(PROCESSED_DIR / "variants_vus.csv", index=False)

    print(f"Labelled (pathogenic/benign) variants: {len(train_eval_set)}")
    print(f"  pathogenic: {(train_eval_set['label'] == 'pathogenic').sum()}")
    print(f"  benign:     {(train_eval_set['label'] == 'benign').sum()}")
    print(f"VUS held out for qualitative evaluation: {len(vus_set)}")


if __name__ == "__main__":
    main()
