"""
Step 7: Structural and stability assessment of a mutant model.

Compares the relaxed wild-type and mutant models written by mutate_pdb.py:

  - Global C-alpha RMSD and local heavy-atom RMSD (residues within 8 A of the
    site, site excluded) after superposition on the C-alpha atoms.
  - Relative solvent accessibility (RSA) of the site residue in each model.
  - Side-chain interactions made by the site residue: hydrogen bonds
    (N/O pairs <= 3.5 A), salt bridges (oppositely charged groups <= 4.0 A)
    and non-polar contacts (C-C <= 4.0 A), and which of them are lost or
    gained in the mutant.
  - Charged groups within 8 A of the site, to show the local electrostatic
    environment a charge change lands in.

The minimised energies from OpenMM are NOT a ddG: the two models contain
different atoms, so their absolute energies cannot be subtracted. A
quantitative ddG needs FoldX (licensed) or a dedicated predictor;
run_foldx_ddg() calls FoldX if a binary is available and otherwise returns
None, and the result is reported as unavailable rather than estimated.

Usage:
    python -m src.structure.stability_analysis                       # case studies
    python -m src.structure.stability_analysis --gene ACTC1 --variant E101K
"""
import argparse
import json
import re
import shutil
import subprocess

import numpy as np
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.SASA import ShrakeRupley

from src.config import OUTPUTS_DIR
from src.features.structural_features import MAX_ASA

STRUCT_OUT = OUTPUTS_DIR / "structures"
LOCAL_RADIUS = 8.0
POSITIVE = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("HIS", "ND1"), ("HIS", "NE2")}
NEGATIVE = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
BACKBONE = {"N", "CA", "C", "O", "OXT"}

# Variants examined in the report: two ACTC1 variants with opposite ClinVar
# classifications, and two classic pathogenic variants from the training set.
CASE_STUDIES = [("ACTC1", "E101K"), ("ACTC1", "R39H"), ("MYH7", "R403Q"), ("KCNQ1", "R243H")]


def load_heavy(path):
    structure = PDBParser(QUIET=True).get_structure(path.stem, path)
    for residue in structure.get_residues():
        for atom in [a for a in residue if a.element == "H"]:
            residue.detach_child(atom.get_id())
    return structure


def ca_map(structure) -> dict:
    return {(r.get_parent().id, r.id[1]): r["CA"] for r in structure.get_residues() if "CA" in r}


def compute_rmsd(wt, mut, site: int) -> dict:
    wt_ca, mut_ca = ca_map(wt), ca_map(mut)
    keys = sorted(set(wt_ca) & set(mut_ca))
    sup = Superimposer()
    sup.set_atoms([wt_ca[k] for k in keys], [mut_ca[k] for k in keys])
    rot, tran = sup.rotran

    site_res = wt[0]["A"][site]
    site_xyz = np.array([a.coord for a in site_res])
    local_pairs = []
    for res in wt[0].get_residues():
        if res.id[1] == site and res.get_parent().id == "A":
            continue
        if np.min(np.linalg.norm(np.array([a.coord for a in res])[:, None] - site_xyz[None], axis=-1)) > LOCAL_RADIUS:
            continue
        other = mut[0][res.get_parent().id][res.id[1]]
        for atom in res:
            if atom.get_id() in other:
                local_pairs.append((atom.coord, np.dot(other[atom.get_id()].coord, rot) + tran))
    local = np.sqrt(np.mean([np.sum((a - b) ** 2) for a, b in local_pairs]))
    return {"global_ca_rmsd": round(float(sup.rms), 3), "local_heavy_rmsd": round(float(local), 3),
            "n_local_atoms": len(local_pairs)}


def site_rsa(structure, site: int) -> float:
    ShrakeRupley().compute(structure[0], level="R")
    res = structure[0]["A"][site]
    return round(min(res.sasa / MAX_ASA[res.get_resname()], 1.0), 3)


def interactions(structure, site: int) -> dict:
    res = structure[0]["A"][site]
    side = [a for a in res if a.get_id() not in BACKBONE]
    others = [a for a in structure[0].get_atoms() if a.get_parent() is not res]
    hbonds, salt, nonpolar = set(), set(), set()
    for a in side:
        for b in others:
            d = a - b
            if d > 4.0:
                continue
            partner_res = b.get_parent()
            # Chain ids other than "A" are only crop segments, so the
            # residue number alone identifies the partner.
            partner = f"{partner_res.get_resname().title()}{partner_res.id[1]}"
            ka, kb = (res.get_resname(), a.get_id()), (partner_res.get_resname(), b.get_id())
            if a.element in "NO" and b.element in "NO" and d <= 3.5:
                hbonds.add(f"{a.get_id()}–{partner}:{b.get_id()}")
            if (ka in POSITIVE and kb in NEGATIVE) or (ka in NEGATIVE and kb in POSITIVE):
                salt.add(partner)
            if a.element == "C" and b.element == "C":
                nonpolar.add(partner)
    return {"hbonds": sorted(hbonds), "salt_bridges": sorted(salt), "nonpolar_contacts": sorted(nonpolar)}


def charged_neighbours(structure, site: int) -> dict:
    res = structure[0]["A"][site]
    ca = res["CA"].coord
    pos, neg = set(), set()
    for atom in structure[0].get_atoms():
        r = atom.get_parent()
        if r is res or np.linalg.norm(atom.coord - ca) > LOCAL_RADIUS:
            continue
        key = (r.get_resname(), atom.get_id())
        label = f"{r.get_resname().title()}{r.id[1]}"
        if key in POSITIVE and r.get_resname() != "HIS":
            pos.add(label)
        elif key in NEGATIVE:
            neg.add(label)
    return {"positive": sorted(pos), "negative": sorted(neg)}


def run_foldx_ddg(pdb_path: str, mutation_spec: str, foldx_binary: str = "foldx") -> float | None:
    """ddG with a local FoldX BuildModel run, or None if FoldX is unavailable.

    FoldX needs an academic licence and a local binary
    (https://foldxsuite.crg.eu); it is not redistributable, so the project
    cannot ship it. `mutation_spec` uses FoldX individual-list format, e.g.
    "EA101K;".
    """
    if shutil.which(foldx_binary) is None:
        return None
    workdir = STRUCT_OUT / "foldx"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "individual_list.txt").write_text(mutation_spec + "\n")
    subprocess.run([foldx_binary, "--command=BuildModel", f"--pdb={pdb_path}",
                    "--mutant-file=individual_list.txt", "--output-dir=."], cwd=workdir, check=True)
    fxout = next(workdir.glob("Average_*.fxout"))
    last = fxout.read_text().strip().splitlines()[-1].split("\t")
    return float(last[2])


def interpret(result: dict) -> str:
    lost = set(result["wt"]["salt_bridges"]) - set(result["mutant"]["salt_bridges"])
    lost_hb = len(result["wt"]["hbonds"]) - len(result["mutant"]["hbonds"])
    parts = []
    if result["wt"]["rsa"] < 0.25:
        parts.append(f"the site is largely buried (RSA {result['wt']['rsa']:.2f})")
    else:
        parts.append(f"the site is solvent exposed (RSA {result['wt']['rsa']:.2f})")
    if lost:
        parts.append("salt bridge(s) lost with " + ", ".join(sorted(lost)))
    if lost_hb > 0:
        parts.append(f"{lost_hb} side-chain hydrogen bond(s) fewer in the mutant")
    parts.append(f"local backbone/side-chain shift after relaxation {result['rmsd']['local_heavy_rmsd']:.2f} Å")
    parts.append("ΔΔG not available (FoldX not installed)" if result["ddg"] is None
                 else f"FoldX ΔΔG {result['ddg']:.2f} kcal/mol")
    return "; ".join(parts) + "."


def analyse(gene: str, variant: str) -> dict:
    tag = f"{gene}_{variant}"
    site = int(re.search(r"\d+", variant).group())
    wt = load_heavy(STRUCT_OUT / f"{tag}_wt.pdb")
    mut = load_heavy(STRUCT_OUT / f"{tag}_mutant.pdb")
    result = {"gene": gene, "variant": variant, "rmsd": compute_rmsd(wt, mut, site)}
    for name, s in (("wt", wt), ("mutant", mut)):
        result[name] = {"rsa": site_rsa(s, site), **interactions(s, site)}
    result["charged_neighbours_8A"] = charged_neighbours(wt, site)
    result["ddg"] = run_foldx_ddg(str(STRUCT_OUT / f"{tag}_wt.pdb"), f"{variant[0]}A{variant[1:]};")
    result["interpretation"] = interpret(result)
    with open(STRUCT_OUT / f"{tag}_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene")
    parser.add_argument("--variant")
    parser.add_argument("--skip-build", action="store_true", help="reuse existing relaxed models")
    args = parser.parse_args()
    cases = [(args.gene, args.variant)] if args.gene else CASE_STUDIES
    for gene, variant in cases:
        if not args.skip_build or not (STRUCT_OUT / f"{gene}_{variant}_mutant.pdb").exists():
            from src.structure.mutate_pdb import run
            run(gene, variant)
        result = analyse(gene, variant)
        print(f"\n{gene} {variant}: {result['interpretation']}")
        print(json.dumps({k: result[k] for k in ("rmsd", "wt", "mutant", "charged_neighbours_8A")}, indent=1))


if __name__ == "__main__":
    main()
