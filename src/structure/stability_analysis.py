"""
Step 7: Structural Stability Assessment.

Two parts:

1. RMSD (fully implemented here) — structural deviation between wild-type
   and mutant structures, computed with Biopython's Superimposer over
   aligned CA atoms.

2. Delta-delta-G (ddG) stability prediction — per methodology Section 8
   (Limitations), this is NOT fully automatable from this codebase:
     - FoldX requires an academic license/registration and a local binary
       (https://foldxsuite.crg.eu).
     - DynaMut2 (https://biosig.lab.uq.edu.au/dynamut2) is a web server
       with no public bulk API.
   This module therefore provides (a) a working RMSD calculator, and
   (b) a thin wrapper you fill in once you have FoldX installed locally,
   plus instructions for the manual DynaMut2 fallback.

Usage:
    python -m src.structure.stability_analysis --wt wt.pdb --mut mutant.pdb
"""
import argparse
import subprocess
from pathlib import Path

from Bio.PDB import PDBParser, Superimposer


def get_ca_atoms(structure, chain_id="A"):
    """
    Get C-alpha atoms from one specific chain.

    Using a single corresponding chain is important when a PDB
    contains multiple copies of the same protein.
    """
    atoms = []

    model = next(structure.get_models())

    if chain_id not in model:
        raise ValueError(
            f"Chain {chain_id} not found in structure."
        )

    chain = model[chain_id]

    for residue in chain:
        if "CA" in residue:
            atoms.append(residue["CA"])

    return atoms


def compute_rmsd(wt_pdb_path: str, mut_pdb_path: str) -> float:
    """
    Compute CA-atom RMSD between wild-type and mutant structures after
    superimposition. Assumes both structures have the same residue
    numbering/length (true for a single point mutation predicted with the
    same tool, e.g. ESMFold on sequences of equal length).
    """
    parser = PDBParser(QUIET=True)
    wt_structure = parser.get_structure("wt", wt_pdb_path)
    mut_structure = parser.get_structure("mut", mut_pdb_path)

    wt_atoms = get_ca_atoms(wt_structure, chain_id="A")
    mut_atoms = get_ca_atoms(mut_structure, chain_id="A")

    n = min(len(wt_atoms), len(mut_atoms))
    if len(wt_atoms) != len(mut_atoms):
        print(f"[warn] CA atom count mismatch ({len(wt_atoms)} vs {len(mut_atoms)}); "
              f"comparing first {n} common positions only.")

    sup = Superimposer()
    sup.set_atoms(wt_atoms[:n], mut_atoms[:n])
    return sup.rms


def run_foldx_ddg(pdb_path: str, mutation_spec: str, foldx_binary: str = "foldx") -> float | None:
    """
    Wrapper for a LOCAL FoldX installation's BuildModel command, which
    estimates ddG for a point mutation. Requires:
      1. A FoldX license + binary (free for academic use) placed on PATH
         or pointed to via `foldx_binary`.
      2. A repaired PDB file (`foldx --command=RepairPDB`) as input.
      3. A mutant_file.txt in FoldX's individual-list format, e.g. "RA403Q;"

    This function is left as a subprocess wrapper you complete once FoldX
    is installed in your own environment — the sandbox this project was
    scaffolded in does not have FoldX available. Returns None if FoldX is
    not found, so the rest of the pipeline can still run without it.
    """
    try:
        subprocess.run([foldx_binary, "--help"], capture_output=True, check=False)
    except FileNotFoundError:
        print("[info] FoldX binary not found — skipping ddG calculation. "
              "Install FoldX (https://foldxsuite.crg.eu) or use the DynaMut2 "
              "web server manually as a fallback (methodology Section 8).")
        return None

    # TODO once FoldX is installed: build the mutant_file.txt, invoke
    # `foldx --command=BuildModel --pdb=<repaired.pdb> --mutant-file=mutant_file.txt`,
    # and parse the resulting Average_*.fxout file for the ddG value.
    raise NotImplementedError("Fill in the FoldX BuildModel invocation once the binary is available.")


def interpret_stability(rmsd: float, ddg: float | None) -> str:
    """Simple rule-of-thumb interpretation combining RMSD and ddG, if available."""
    if ddg is not None:
        if ddg > 1.0:
            stability = "likely destabilizing"
        elif ddg < -1.0:
            stability = "likely stabilizing"
        else:
            stability = "near-neutral effect on stability"
        return f"{stability} (ddG={ddg:.2f} kcal/mol, RMSD={rmsd:.2f} \u00c5)"
    else:
        # Fall back to RMSD-only heuristic; thresholds are illustrative —
        # calibrate against your own dataset once you have labelled examples.
        if rmsd > 2.0:
            return f"substantial structural deviation (RMSD={rmsd:.2f} \u00c5); ddG not available"
        return f"modest structural deviation (RMSD={rmsd:.2f} \u00c5); ddG not available"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt", required=True, help="Wild-type PDB file path")
    parser.add_argument("--mut", required=True, help="Mutant PDB file path")
    args = parser.parse_args()

    rmsd = compute_rmsd(args.wt, args.mut)
    ddg = run_foldx_ddg(args.wt, mutation_spec="") if Path(args.wt).exists() else None
    print(interpret_stability(rmsd, None))  # ddg left as None until FoldX is wired in
