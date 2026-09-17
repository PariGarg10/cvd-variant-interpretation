"""
Step 1 (part C): Retrieve wild-type protein structures.

Strategy (per methodology Section 5.1): prefer an experimentally resolved
PDB structure; fall back to the AlphaFold DB predicted model if no PDB
entry exists for the gene's UniProt accession.

APIs used:
  - RCSB PDB Search API + Data API (https://www.rcsb.org)
  - AlphaFold DB (https://alphafold.ebi.ac.uk)

Usage:
    python -m src.data_collection.fetch_structures
"""
import time
import requests
from tqdm import tqdm

from src.config import RAW_DIR, load_gene_panel, ensure_dirs

PDB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
ALPHAFOLD_MODEL_URL_TMPL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
STRUCTURES_DIR = RAW_DIR / "structures"


def find_pdb_ids_for_uniprot(uniprot_id: str) -> list[str]:
    """Query RCSB search API for PDB entries mapped to a UniProt accession."""
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id,
            },
        },
        "return_type": "entry",
    }
    resp = requests.post(PDB_SEARCH_URL, json=query, timeout=30)
    if resp.status_code == 204:
        return []  # no hits
    resp.raise_for_status()
    hits = resp.json().get("result_set", [])
    return [h["identifier"] for h in hits]


def download_pdb_structure(pdb_id: str, dest_path):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest_path.write_bytes(resp.content)


def download_alphafold_structure(uniprot_id: str, dest_path) -> bool:
    url = ALPHAFOLD_MODEL_URL_TMPL.format(uniprot_id=uniprot_id)
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    dest_path.write_bytes(resp.content)
    return True


def main():
    ensure_dirs()
    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    genes = load_gene_panel("both")

    manifest = []
    for gene in tqdm(genes, desc="Fetching structures"):
        symbol, uid = gene["symbol"], gene["uniprot_id"]
        source, structure_id = None, None
        try:
            pdb_ids = find_pdb_ids_for_uniprot(uid)
        except requests.RequestException as e:
            print(f"[warn] PDB search failed for {symbol}: {e}")
            pdb_ids = []

        if pdb_ids:
            # Prefer the first hit; for a real run you'd rank by resolution /
            # coverage of the region containing your variants of interest.
            pdb_id = pdb_ids[0]
            dest = STRUCTURES_DIR / f"{symbol}_{pdb_id}.pdb"
            try:
                download_pdb_structure(pdb_id, dest)
                source, structure_id = "PDB", pdb_id
            except requests.RequestException as e:
                print(f"[warn] PDB download failed for {symbol}/{pdb_id}: {e}")

        if source is None:
            # Fallback to AlphaFold DB
            dest = STRUCTURES_DIR / f"{symbol}_AF_{uid}.pdb"
            try:
                ok = download_alphafold_structure(uid, dest)
                if ok:
                    source, structure_id = "AlphaFoldDB", uid
            except requests.RequestException as e:
                print(f"[warn] AlphaFold download failed for {symbol}: {e}")

        manifest.append(
            {"gene_symbol": symbol, "uniprot_id": uid, "source": source, "structure_id": structure_id}
        )
        time.sleep(0.2)

    import pandas as pd

    pd.DataFrame(manifest).to_csv(RAW_DIR / "structure_manifest.csv", index=False)
    print(f"Structure manifest saved to {RAW_DIR / 'structure_manifest.csv'}")


if __name__ == "__main__":
    main()
