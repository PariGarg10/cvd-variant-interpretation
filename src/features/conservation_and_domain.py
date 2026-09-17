# """
# Step 3 (part B): Conservation score and domain-membership features.

# Conservation scoring requires either:
#   (a) a multiple sequence alignment (MSA) of orthologs, scored with
#       phyloP/phastCons-style methods, or
#   (b) a precomputed per-residue score from a service like ConSurf-DB
#       (https://consurfdb.tau.ac.il).

# This module defines the feature interface and a placeholder implementation
# so the rest of the pipeline can be developed and tested before the
# conservation-scoring piece is wired up. Two real implementation paths are
# sketched below — pick whichever fits your timeline (see README "Known gaps
# to fill in").
# """
# from dataclasses import dataclass


# @dataclass
# class ConservationFeatures:
#     conservation_score: float  # higher = more conserved; None if unavailable
#     in_annotated_domain: bool
#     domain_type: str | None


# def compute_domain_membership(position: int, uniprot_features: list[dict]) -> tuple[bool, str | None]:
#     """
#     Determine whether a residue position falls inside an annotated UniProt
#     Domain/Region/Active site/Binding site (from fetch_uniprot.py output).
#     """
#     for feat in uniprot_features:
#         start, end = feat.get("start"), feat.get("end")
#         if start is not None and end is not None and start <= position <= end:
#             return True, feat.get("type")
#     return False, None


# def compute_conservation_score_placeholder(position: int) -> float | None:
#     """
#     Placeholder — returns None until a real conservation source is wired in.

#     Implementation options (see README):
#       1. ConSurf-DB: download the precomputed grades file for the UniProt
#          accession from https://consurfdb.tau.ac.il and index by position.
#       2. Compute locally: pull ortholog sequences (e.g. via UniProt's
#          "Similar proteins" / NCBI HomoloGene), align with MAFFT/Clustal
#          Omega (Biopython wraps both), then compute a Shannon-entropy or
#          phyloP-style conservation score per alignment column, mapped back
#          to the reference sequence's residue numbering.
#     """
#     return None


# def compute_conservation_and_domain_features(
#     position: int, uniprot_features: list[dict]
# ) -> ConservationFeatures:
#     in_domain, domain_type = compute_domain_membership(position, uniprot_features)
#     score = compute_conservation_score_placeholder(position)
#     return ConservationFeatures(
#         conservation_score=score,
#         in_annotated_domain=in_domain,
#         domain_type=domain_type,
#     )
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


def compute_domain_membership(
    position: int,
    uniprot_features: list[dict]
) -> tuple[bool, str | None]:

    for feat in uniprot_features:
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