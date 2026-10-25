"""
Step 1 (part E): Download AlphaFold DB models for the gene panel.

Uses the AlphaFold DB prediction API (https://alphafold.ebi.ac.uk) to find
the current model for each UniProt accession and saves the PDB file plus a
manifest. Proteins longer than AlphaFold DB's single-model limit (here,
RYR2 at 4,967 residues) have no model and are skipped.

Usage:
    python -m src.data_collection.fetch_alphafold
"""
import json
import time

import requests
from tqdm import tqdm

from src.config import RAW_DIR, load_gene_panel, ensure_dirs

API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
OUT_DIR = RAW_DIR / "alphafold"


def main():
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for gene in tqdm(load_gene_panel("both"), desc="Fetching AlphaFold models"):
        uid = gene["uniprot_id"]
        resp = requests.get(API_URL.format(uniprot_id=uid), timeout=60)
        if resp.status_code == 404:
            print(f"[warn] no AlphaFold model for {gene['symbol']} ({uid})")
            continue
        resp.raise_for_status()
        for pred in resp.json():
            if pred.get("uniprotAccession") != uid:
                continue
            pdb_name = pred["pdbUrl"].split("/")[-1]
            pdb_path = OUT_DIR / pdb_name
            if not pdb_path.exists():
                model = requests.get(pred["pdbUrl"], timeout=300)
                model.raise_for_status()
                pdb_path.write_bytes(model.content)
            manifest.append({
                "gene_symbol": gene["symbol"],
                "uniprot_id": uid,
                "entry": pred["entryId"],
                "start": pred.get("uniprotStart"),
                "end": pred.get("uniprotEnd"),
                "pdb": pdb_name,
                "global_plddt": pred.get("globalMetricValue"),
            })
        time.sleep(0.2)

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"Saved {len(manifest)} AlphaFold models to {OUT_DIR}")


if __name__ == "__main__":
    main()
