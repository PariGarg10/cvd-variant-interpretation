"""
Step 3 (part D): Residue-level structural features from AlphaFold models.

For every residue of each panel protein that has an AlphaFold DB model:

  - plddt            AlphaFold per-residue confidence (0-100). Low values
                     usually mark intrinsically disordered segments.
  - rsa              Relative solvent accessibility: side-chain + backbone
                     SASA (Shrake-Rupley) divided by the residue's maximum
                     theoretical SASA (Tien et al. 2013). ~0 = buried.
  - contact_count    Number of other C-alpha atoms within 10 A, a simple
                     measure of how tightly packed the neighbourhood is.

AlphaFold models use UniProt canonical numbering, so positions map directly
onto the (UniProt-verified) variant positions from preprocessing.

Models are downloaded by:
    python -m src.data_collection.fetch_alphafold

Output:
    data/raw/structure_features.json  {gene: {position: {plddt, rsa, contact_count}}}

Usage:
    python -m src.features.structural_features
"""
import json

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

from src.config import RAW_DIR, load_gene_panel, ensure_dirs

ALPHAFOLD_DIR = RAW_DIR / "alphafold"
OUTPUT = RAW_DIR / "structure_features.json"

# Theoretical maximum SASA (A^2), Tien et al. (2013) PLoS ONE 8:e80635
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}
CONTACT_RADIUS = 10.0


def residue_features(pdb_path) -> dict:
    structure = PDBParser(QUIET=True).get_structure("af", pdb_path)
    model = next(structure.get_models())
    ShrakeRupley().compute(model, level="R")

    residues = [r for r in model.get_residues() if "CA" in r and r.get_resname() in MAX_ASA]
    ca = np.array([r["CA"].coord for r in residues])
    dist = np.sqrt(((ca[:, None, :] - ca[None, :, :]) ** 2).sum(-1))
    contacts = (dist < CONTACT_RADIUS).sum(1) - 1

    out = {}
    for res, n_contacts in zip(residues, contacts):
        out[str(res.id[1])] = {
            "plddt": round(float(res["CA"].get_bfactor()), 2),
            "rsa": round(min(res.sasa / MAX_ASA[res.get_resname()], 1.0), 4),
            "contact_count": int(n_contacts),
        }
    return out


def main():
    ensure_dirs()
    manifest_path = ALPHAFOLD_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run `python -m src.data_collection.fetch_alphafold` first."
        )
    manifest = {m["uniprot_id"]: m for m in json.load(open(manifest_path))}

    features = {}
    for gene in load_gene_panel("both"):
        entry = manifest.get(gene["uniprot_id"])
        if entry is None:
            print(f"[warn] no AlphaFold model for {gene['symbol']}; structural features left empty")
            features[gene["symbol"]] = {}
            continue
        features[gene["symbol"]] = residue_features(ALPHAFOLD_DIR / entry["pdb"])
        print(f"{gene['symbol']}: {len(features[gene['symbol']])} residues")

    with open(OUTPUT, "w") as f:
        json.dump(features, f)
    print(f"Saved structural features to {OUTPUT}")


if __name__ == "__main__":
    main()
