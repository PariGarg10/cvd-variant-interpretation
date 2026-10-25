"""
Build sequence-based conservation scores for the CVD gene panel.

For each human UniProt reference protein:
    1. Search UniProtKB for vertebrate orthologs carrying the same gene name
       (reviewed and unreviewed, one sequence per species, length-filtered).
    2. Align each ortholog to the human reference (global, BLOSUM62).
    3. Score each human residue as the fraction of aligned orthologs that
       keep the same amino acid.

Output:
    data/raw/conservation_scores.json
"""

import json
import time
from pathlib import Path

import requests
from Bio.Align import PairwiseAligner, substitution_matrices

from src.config import RAW_DIR, ensure_dirs


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

CONSERVATION_OUTPUT = RAW_DIR / "conservation_scores.json"

REQUEST_TIMEOUT = 30

MIN_HOMOLOGS = 3
MAX_HOMOLOGS = 150


def load_uniprot_entries():
    path = RAW_DIR / "uniprot_entries.json"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. "
            "Run `python -m src.data_collection.fetch_uniprot` first."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def search_homologs(gene_symbol, human_uniprot_id, human_length):
    """
    Retrieve vertebrate orthologs of a gene from UniProtKB.

    Reviewed (Swiss-Prot) entries alone give only a handful of sequences per
    gene, mostly from closely related mammals, which makes nearly every
    position look conserved. We therefore query reviewed and unreviewed
    entries with the same gene name across Vertebrata (taxon 7742), keep one
    sequence per organism (preferring reviewed entries and lengths closest to
    the human protein), and drop fragments or fusions whose length differs
    from the human sequence by more than 30%.
    """
    params = {
        "query": f"gene_exact:{gene_symbol} AND taxonomy_id:7742",
        "format": "json",
        "size": 500,
        "fields": "accession,reviewed,organism_name,length,sequence",
    }
    try:
        response = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        results = response.json().get("results", [])
    except requests.RequestException as exc:
        print(f"[warn] UniProt request failed for {gene_symbol}: {exc}")
        return []

    best_per_organism = {}
    for result in results:
        accession = result.get("primaryAccession")
        organism = (result.get("organism") or {}).get("scientificName")
        sequence = (result.get("sequence") or {}).get("value")
        if not accession or not sequence or accession == human_uniprot_id:
            continue
        if organism == "Homo sapiens":
            continue
        if not (0.7 * human_length <= len(sequence) <= 1.3 * human_length):
            continue
        reviewed = "Swiss-Prot" in str(result.get("entryType", ""))
        rank = (not reviewed, abs(len(sequence) - human_length))
        current = best_per_organism.get(organism)
        if current is None or rank < current["rank"]:
            best_per_organism[organism] = {
                "accession": accession,
                "sequence": sequence,
                "organism": organism,
                "rank": rank,
            }

    homologs = sorted(best_per_organism.values(), key=lambda h: h["rank"])
    return homologs[:MAX_HOMOLOGS]


def create_aligner():
    """
    Global pairwise aligner with BLOSUM62 scores and affine gap penalties.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    return aligner


def align_and_score(reference_sequence, homologs):
    """
    Calculate residue-level identity conservation.

    For each position in the human reference:

        conservation =
            identical aligned residues /
            homolog residues observed at that position

    Positions with fewer than MIN_HOMOLOGS observations are omitted.
    """

    aligner = create_aligner()

    counts = {
        position: {
            "same": 0,
            "observed": 0,
        }
        for position in range(1, len(reference_sequence) + 1)
    }

    for homolog in homologs:

        homolog_sequence = homolog["sequence"]

        try:
            alignment = aligner.align(
                reference_sequence,
                homolog_sequence,
            )[0]

        except Exception as exc:
            print(
                f"[warn] Alignment failed for "
                f"{homolog['accession']}: {exc}"
            )
            continue

        # Convert the alignment into aligned strings.
        aligned_reference, aligned_homolog = alignment_to_strings(
            alignment,
            reference_sequence,
            homolog_sequence,
        )

        reference_position = 0

        for ref_aa, homolog_aa in zip(
            aligned_reference,
            aligned_homolog,
        ):

            if ref_aa != "-":
                reference_position += 1

            if ref_aa == "-":
                continue

            if homolog_aa == "-":
                continue

            counts[reference_position]["observed"] += 1

            if ref_aa == homolog_aa:
                counts[reference_position]["same"] += 1

    scores = {}

    for position, values in counts.items():

        if values["observed"] < MIN_HOMOLOGS:
            continue

        score = (
            values["same"] /
            values["observed"]
        )

        scores[str(position)] = round(score, 4)

    return scores


def alignment_to_strings(
    alignment,
    reference_sequence,
    homolog_sequence,
):
    """
    Convert a Biopython PairwiseAligner alignment into
    two gapped strings.

    The coordinates describe blocks in the original sequences.
    """

    aligned_reference = []
    aligned_homolog = []

    ref_blocks = alignment.aligned[0]
    hom_blocks = alignment.aligned[1]

    ref_cursor = 0
    hom_cursor = 0

    for (ref_start, ref_end), (hom_start, hom_end) in zip(
        ref_blocks,
        hom_blocks,
    ):

        # Gap in homolog.
        if ref_start > ref_cursor:
            aligned_reference.append(
                reference_sequence[ref_cursor:ref_start]
            )

            aligned_homolog.append(
                "-" * (ref_start - ref_cursor)
            )

        # Gap in reference.
        if hom_start > hom_cursor:
            aligned_reference.append(
                "-" * (hom_start - hom_cursor)
            )

            aligned_homolog.append(
                homolog_sequence[hom_cursor:hom_start]
            )

        # Aligned region.
        aligned_reference.append(
            reference_sequence[ref_start:ref_end]
        )

        aligned_homolog.append(
            homolog_sequence[hom_start:hom_end]
        )

        ref_cursor = ref_end
        hom_cursor = hom_end

    # Remaining sequence.
    if ref_cursor < len(reference_sequence):
        aligned_reference.append(
            reference_sequence[ref_cursor:]
        )

        aligned_homolog.append(
            "-" * (len(reference_sequence) - ref_cursor)
        )

    if hom_cursor < len(homolog_sequence):
        aligned_reference.append(
            "-" * (len(homolog_sequence) - hom_cursor)
        )

        aligned_homolog.append(
            homolog_sequence[hom_cursor:]
        )

    return (
        "".join(aligned_reference),
        "".join(aligned_homolog),
    )


def build_conservation_for_gene(entry):

    gene = entry["gene_symbol"]
    uniprot_id = entry["uniprot_id"]
    reference_sequence = entry["sequence"]

    print(f"\n[{gene}]")
    print(f"Reference: {uniprot_id}")
    print(f"Sequence length: {len(reference_sequence)}")

    homologs = search_homologs(
        gene,
        uniprot_id,
        len(reference_sequence),
    )

    print(
        f"Retrieved vertebrate orthologs (one per species): "
        f"{len(homologs)}"
    )

    if len(homologs) < MIN_HOMOLOGS:

        print(
            f"[warn] Not enough homologs for {gene}; "
            f"need at least {MIN_HOMOLOGS}."
        )

        return {}

    scores = align_and_score(
        reference_sequence,
        homologs,
    )

    print(
        f"Calculated conservation scores: "
        f"{len(scores)}"
    )

    return scores


def main():

    ensure_dirs()

    entries = load_uniprot_entries()

    print(f"Found {len(entries)} genes.")

    conservation_scores = {}

    for entry in entries:

        gene = entry["gene_symbol"]

        try:

            scores = build_conservation_for_gene(
                entry
            )

            conservation_scores[gene] = scores

        except Exception as exc:

            print(
                f"[error] Failed to process "
                f"{gene}: {exc}"
            )

            conservation_scores[gene] = {}

        # Avoid sending requests too quickly.
        time.sleep(1)

    with open(
        CONSERVATION_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            conservation_scores,
            f,
            indent=2,
        )

    print(
        f"\nSaved conservation scores to:"
        f"\n{CONSERVATION_OUTPUT}"
    )

    print("\nSummary:")

    for gene, scores in conservation_scores.items():

        print(
            f"  {gene}: "
            f"{len(scores)} residue scores"
        )


if __name__ == "__main__":
    main()