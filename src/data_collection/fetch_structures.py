"""
Step 1 (part C): Experimental protein structures for the gene panel.

UniProt lists every PDB entry mapped to an accession, with its method,
resolution and the UniProt residue range each chain covers. For each gene
we record how many PDB entries exist and download the one covering the
most residues (ties broken by resolution) as mmCIF.

Predicted full-length models come from AlphaFold DB (fetch_alphafold.py);
those are what the structural features and mutant modelling use, because
they cover the whole protein in UniProt numbering. Experimental entries are
kept for reference and for checking residue numbering (for example, PDB
9B3R numbers cardiac actin from the mature N-terminus, two residues behind
UniProt).

Usage:
    python -m src.data_collection.fetch_structures
"""
import re
import time

import pandas as pd
import requests
from tqdm import tqdm

from src.config import RAW_DIR, load_gene_panel, ensure_dirs

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{uid}.json"
PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
STRUCTURES_DIR = RAW_DIR / "structures"


def pdb_entries_for_uniprot(uniprot_id: str) -> list[dict]:
    resp = requests.get(UNIPROT_URL.format(uid=uniprot_id), timeout=60)
    resp.raise_for_status()
    entries = []
    for xref in resp.json().get("uniProtKBCrossReferences", []):
        if xref.get("database") != "PDB":
            continue
        props = {p["key"]: p["value"] for p in xref.get("properties", [])}
        covered = 0
        for _, start, end in re.findall(r"([A-Za-z0-9/]+)=(\d+)-(\d+)", props.get("Chains", "")):
            covered = max(covered, int(end) - int(start) + 1)
        resolution = re.match(r"([\d.]+)", props.get("Resolution", ""))
        entries.append({
            "pdb_id": xref["id"],
            "method": props.get("Method"),
            "resolution": float(resolution.group(1)) if resolution else None,
            "chains": props.get("Chains"),
            "residues_covered": covered,
        })
    return entries


def main():
    ensure_dirs()
    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for gene in tqdm(load_gene_panel("both"), desc="Fetching structures"):
        symbol, uid = gene["symbol"], gene["uniprot_id"]
        try:
            entries = pdb_entries_for_uniprot(uid)
        except requests.RequestException as e:
            print(f"[warn] UniProt lookup failed for {symbol}: {e}")
            entries = []
        row = {"gene_symbol": symbol, "uniprot_id": uid, "n_pdb_entries": len(entries),
               "best_pdb": None, "method": None, "resolution": None, "residues_covered": 0}
        if entries:
            best = max(entries, key=lambda e: (e["residues_covered"], -(e["resolution"] or 99)))
            try:
                resp = requests.get(PDB_DOWNLOAD_URL.format(pdb_id=best["pdb_id"]), timeout=120)
                resp.raise_for_status()
                (STRUCTURES_DIR / f"{symbol}_{best['pdb_id']}.cif").write_bytes(resp.content)
                row.update(best_pdb=best["pdb_id"], method=best["method"], resolution=best["resolution"],
                           residues_covered=best["residues_covered"])
            except requests.RequestException as e:
                print(f"[warn] PDB download failed for {symbol}/{best['pdb_id']}: {e}")
        manifest.append(row)
        time.sleep(0.2)

    pd.DataFrame(manifest).to_csv(RAW_DIR / "structure_manifest.csv", index=False)
    print(f"Structure manifest saved to {RAW_DIR / 'structure_manifest.csv'}")


if __name__ == "__main__":
    main()
