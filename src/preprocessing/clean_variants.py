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
import json
import re
import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR, ensure_dirs

PATHOGENIC_LABELS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN_LABELS = {"Benign", "Likely benign", "Benign/Likely benign"}
VUS_LABELS = {"Uncertain significance"}

# ClinVar review status -> "gold stars". Used to keep the best-supported
# record when the same substitution appears more than once, and exposed as a
# column so training can be restricted to higher-confidence labels.
REVIEW_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, single submitter": 1,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no classification provided": 0,
}

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


def parse_all_protein_changes(text: str) -> list[tuple[str, int, str]]:
    """Return every (wt, position, mut) substitution mentioned in `text`.

    ClinVar lists one protein change per transcript, e.g. "R69C, R147C, R105C",
    so a single record can carry several candidate positions.
    """
    if not isinstance(text, str):
        return []
    out = []
    for m in PROTEIN_CHANGE_RE.finditer(text):
        if m.group("wt3"):
            wt, pos, mut = AA_3TO1.get(m.group("wt3")), int(m.group("pos")), AA_3TO1.get(m.group("mut3"))
        else:
            wt, pos, mut = m.group("wt1"), int(m.group("pos1")), m.group("mut1")
        if wt and mut and wt != "*" and mut != "*":
            out.append((wt, pos, mut))
    return out


def choose_canonical_change(title: str, protein_change: str, sequence: str | None):
    """Pick the substitution whose wild-type residue matches the UniProt
    canonical sequence.

    The HGVS title (ClinVar's preferred transcript, normally MANE Select) is
    tried first, then every isoform listed in protein_change. Returns
    (wt, position, mut, verified). If no sequence is available the first
    candidate is returned unverified; if a sequence is available and nothing
    matches, (None, None, None, False) is returned so the record is dropped
    rather than annotated at the wrong residue.
    """
    candidates = parse_all_protein_changes(title) + parse_all_protein_changes(protein_change)
    if not candidates:
        return None, None, None, False
    if not sequence:
        wt, pos, mut = candidates[0]
        return wt, pos, mut, False
    for wt, pos, mut in candidates:
        if 1 <= pos <= len(sequence) and sequence[pos - 1] == wt:
            return wt, pos, mut, True
    return None, None, None, False


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


def load_reference_sequences() -> dict:
    path = RAW_DIR / "uniprot_entries.json"
    if not path.exists():
        print(f"[warn] {path} not found; protein positions cannot be checked against UniProt.")
        return {}
    with open(path) as f:
        return {e["gene_symbol"]: e["sequence"] for e in json.load(f)}


def belongs_to_gene(row: pd.Series) -> bool:
    """False for records ClinVar does not link to the queried gene (they come
    back from esearch because they overlap it, e.g. an antisense locus)."""
    genes = row.get("record_genes")
    if isinstance(genes, str) and genes:
        return row["gene_symbol"] in genes.split(";")
    title_gene = re.search(r"\(([A-Za-z0-9-]+)\):c\.", str(row.get("title", "")))
    return title_gene is None or title_gene.group(1) == row["gene_symbol"]


def clean(df: pd.DataFrame, sequences: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    sequences = sequences or {}

    n0 = len(df)
    df = df[df.apply(belongs_to_gene, axis=1)]
    print(f"Dropped {n0 - len(df)} records linked to a different gene")

    parsed = [
        choose_canonical_change(t, pc, sequences.get(g))
        for t, pc, g in zip(df["title"], df["protein_change"], df["gene_symbol"])
    ]
    df["wt_aa"], df["position"], df["mut_aa"], df["position_verified"] = zip(*parsed) if parsed else ([], [], [], [])
    n1 = len(df)
    df = df.dropna(subset=["wt_aa", "position", "mut_aa"])
    print(f"Dropped {n1 - len(df)} records with no substitution matching the reference sequence")
    df["position"] = df["position"].astype(int)

    df["label"] = df["clinical_significance"].map(normalize_label)
    df["review_stars"] = df["review_status"].map(lambda s: REVIEW_STARS.get(str(s).strip(), 0))

    # Same substitution reported more than once: keep the best-reviewed record.
    dedup_keys = ["gene_symbol", "wt_aa", "position", "mut_aa"]
    df = df.sort_values("review_stars", ascending=False, kind="stable").drop_duplicates(subset=dedup_keys, keep="first")

    return df.reset_index(drop=True)


def main():
    ensure_dirs()
    raw_path = RAW_DIR / "clinvar_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found — run `python -m src.data_collection.fetch_clinvar` first."
        )

    df = pd.read_csv(raw_path)
    cleaned = clean(df, load_reference_sequences())

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
