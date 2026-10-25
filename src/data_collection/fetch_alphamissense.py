"""
Step 1 (part F): AlphaMissense scores for the gene panel.

AlphaMissense (Cheng et al., Science 2023) publishes a pathogenicity score
for every possible single amino-acid substitution in the human proteome.
The full table (~1.2 GB compressed) is streamed from Google Cloud Storage
and only rows for the panel's UniProt accessions are kept.

AlphaMissense is used in this project as an external comparator, not as a
model input: it was partly calibrated on ClinVar labels, so using it as a
feature would mix an outside predictor into the interpretable model.

Data licence: CC BY-NC-SA 4.0 (non-commercial use).

Usage:
    python -m src.data_collection.fetch_alphamissense
"""
import gzip

import requests

from src.config import RAW_DIR, load_gene_panel, ensure_dirs

AM_URL = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz"
OUTPUT = RAW_DIR / "alphamissense_panel.tsv"


def main():
    ensure_dirs()
    wanted = {g["uniprot_id"] for g in load_gene_panel("both")}
    kept = 0
    with requests.get(AM_URL, stream=True, timeout=60) as resp, open(OUTPUT, "w") as out:
        resp.raise_for_status()
        resp.raw.decode_content = False
        with gzip.GzipFile(fileobj=resp.raw) as gz:
            for raw_line in gz:
                line = raw_line.decode()
                if line.startswith("#"):
                    continue
                if line.startswith("uniprot_id") or line.split("\t", 1)[0] in wanted:
                    out.write(line)
                    kept += 1
    print(f"Saved {kept - 1} AlphaMissense scores to {OUTPUT}")


if __name__ == "__main__":
    main()
