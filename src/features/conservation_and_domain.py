"""
Step 3 (part B): Conservation score and domain-membership features.

Conservation scores are precomputed by build_conservation_scores.py and read
from data/raw/conservation_scores.json. Domain membership is taken from the
UniProt positional features saved by fetch_uniprot.py.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import math


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSERVATION_FILE = PROJECT_ROOT / "data" / "raw" / "conservation_scores.json"


@dataclass
class ConservationFeatures:
    conservation_score: float | None
    in_annotated_domain: bool
    domain_type: str | None


# UniProt feature types that mark a structured or functional unit. "Region"
# is deliberately excluded: in these entries it is mostly used for
# "Disordered" segments, so counting it made disordered residues look like
# domain residues. "Topological domain" (e.g. "Cytoplasmic") is excluded for
# the same reason: it covers whole sides of a membrane, not a domain.
DOMAIN_FEATURE_TYPES = {
    "Domain",
    "Transmembrane",
    "Active site",
    "Binding site",
    "Disulfide bond",
}


def compute_domain_membership(
    position: int,
    uniprot_features: list[dict]
) -> tuple[bool, str | None]:

    for feat in uniprot_features:
        if feat.get("type") not in DOMAIN_FEATURE_TYPES:
            continue
        start, end = feat.get("start"), feat.get("end")

        if start is not None and end is not None:
            if start <= position <= end:
                return True, feat.get("type")

    return False, None


def load_conservation_scores() -> dict:
    """
    Load cached conservation scores.

    Expected format:

    {
        "MYH7": {
            "1": 0.95,
            "2": 1.0,
            ...
        },
        ...
    }
    """

    if not CONSERVATION_FILE.exists():
        return {}

    try:
        with open(CONSERVATION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_conservation_score(
    gene_symbol: str,
    position: int,
    conservation_scores: dict
) -> float | None:

    gene_scores = conservation_scores.get(gene_symbol, {})

    score = gene_scores.get(str(position))

    if score is None:
        return None

    try:
        score = float(score)
    except (TypeError, ValueError):
        return None

    if math.isnan(score):
        return None

    return score


def compute_conservation_and_domain_features(
    gene_symbol: str,
    position: int,
    uniprot_features: list[dict],
    conservation_scores: dict | None = None,
) -> ConservationFeatures:

    in_domain, domain_type = compute_domain_membership(
        position,
        uniprot_features
    )

    if conservation_scores is None:
        conservation_scores = load_conservation_scores()

    score = get_conservation_score(
        gene_symbol,
        position,
        conservation_scores
    )

    return ConservationFeatures(
        conservation_score=score,
        in_annotated_domain=in_domain,
        domain_type=domain_type,
    )