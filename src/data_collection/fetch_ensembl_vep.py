"""
Step 1 (part D): Use Ensembl VEP (Variant Effect Predictor) to map genomic
variant calls (e.g. from dbSNP rsIDs found in ClinVar records) to
amino-acid-level substitutions and additional annotation.

Uses the Ensembl REST API (https://rest.ensembl.org) — free, no auth.

Usage:
    python -m src.data_collection.fetch_ensembl_vep --rsids rs121913343 rs397515967
"""
import argparse
import time
import requests

VEP_URL = "https://rest.ensembl.org/vep/human/id/{rsid}?content-type=application/json"
REQUEST_DELAY = 0.1  # Ensembl allows ~15 req/sec per IP


def annotate_rsid(rsid: str) -> list[dict]:
    resp = requests.get(VEP_URL.format(rsid=rsid), timeout=30)
    resp.raise_for_status()
    return resp.json()


def annotate_rsids(rsids: list[str]) -> list[dict]:
    results = []
    for rsid in rsids:
        try:
            results.extend(annotate_rsid(rsid))
        except requests.RequestException as e:
            print(f"[warn] VEP lookup failed for {rsid}: {e}")
        time.sleep(REQUEST_DELAY)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsids", nargs="+", required=True, help="dbSNP rsIDs to annotate")
    args = parser.parse_args()

    out = annotate_rsids(args.rsids)
    import json

    print(json.dumps(out, indent=2))
