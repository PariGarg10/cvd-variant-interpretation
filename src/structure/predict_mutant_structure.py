"""
Step 6: Mutant protein structure prediction.

ESMFold is used as the primary tool (per methodology Section 5.6) because
it predicts structure from a single sequence with no MSA step, making it
far more practical than AlphaFold2 for on-the-fly mutant structures. This
uses the public ESM Metagenomic Atlas API.

For a handful of high-priority variants where higher accuracy matters, you
can swap in a local AlphaFold2/ColabFold run instead (not scripted here —
that needs GPU + a separate environment, e.g. Google Colab's ColabFold
notebook).

NOTE: this is scoped as PROOF-OF-CONCEPT tooling (methodology Section 8 —
Steps 6-8 are demonstrated on a curated subset, not the full dataset),
since structure prediction is comparatively expensive to run at scale.

Usage:
    python -m src.structure.predict_mutant_structure --sequence <FASTA seq> --out mutant.pdb
"""
import argparse
import requests

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"


def apply_substitution(sequence: str, wt_aa: str, position: int, mut_aa: str) -> str:
    """
    Apply a single amino-acid substitution to a protein sequence.
    `position` is 1-indexed, matching UniProt/ClinVar numbering.
    """
    idx = position - 1
    if sequence[idx] != wt_aa:
        raise ValueError(
            f"Sequence mismatch at position {position}: expected {wt_aa}, found {sequence[idx]}. "
            f"Check that you're using the same reference sequence ClinVar's numbering is based on."
        )
    return sequence[:idx] + mut_aa + sequence[idx + 1 :]


def predict_structure_esmfold(
    sequence: str,
    max_retries: int = 3,
    retry_delay: int = 10,
) -> str:
    """
    Submit a sequence to the ESMFold API and return PDB-format text.

    Retries temporary gateway/server failures because the public
    endpoint can occasionally return 5xx responses or time out.
    """

    last_error = None

    for attempt in range(1, max_retries + 1):

        try:
            print(
                f"Submitting sequence to ESMFold "
                f"(attempt {attempt}/{max_retries})..."
            )

            resp = requests.post(
                ESMFOLD_API_URL,
                data=sequence,
                timeout=300,
            )

            if resp.status_code >= 500:
                print(
                    f"[warn] ESMFold server returned "
                    f"HTTP {resp.status_code}."
                )

                last_error = requests.HTTPError(
                    f"HTTP {resp.status_code}: {resp.text[:300]}"
                )

                if attempt < max_retries:
                    import time
                    print(
                        f"Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)

                continue

            resp.raise_for_status()

            if not resp.text.strip():
                raise RuntimeError(
                    "ESMFold returned an empty response."
                )

            return resp.text

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as exc:

            last_error = exc

            if attempt < max_retries:
                import time
                print(
                    f"[warn] Request failed: {exc}"
                )
                print(
                    f"Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)

    raise RuntimeError(
        f"ESMFold prediction failed after "
        f"{max_retries} attempts."
    ) from last_error


def predict_mutant_structure(wt_sequence: str, wt_aa: str, position: int, mut_aa: str) -> str:
    mutant_sequence = apply_substitution(wt_sequence, wt_aa, position, mut_aa)
    return predict_structure_esmfold(mutant_sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, help="Full wild-type protein sequence")
    parser.add_argument("--wt_aa", required=True, help="Wild-type residue, e.g. R")
    parser.add_argument("--position", type=int, required=True, help="1-indexed residue position")
    parser.add_argument("--mut_aa", required=True, help="Mutant residue, e.g. Q")
    parser.add_argument("--out", required=True, help="Output PDB file path")
    args = parser.parse_args()

    pdb_text = predict_mutant_structure(args.sequence, args.wt_aa, args.position, args.mut_aa)
    with open(args.out, "w") as f:
        f.write(pdb_text)
    print(f"Mutant structure written to {args.out}")
