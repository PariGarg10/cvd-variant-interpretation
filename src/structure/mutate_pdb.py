"""
Step 6: Build a mutant structure model for a single missense variant.

Earlier versions of this script only renamed the residue (e.g. ARG -> HIS)
and left every atom where it was, so the "mutant" was geometrically
identical to the wild type. This version builds a real mutant side chain:

  1. Load the AlphaFold model of the protein. AlphaFold DB models use UniProt
     canonical numbering, which is also the numbering of the (verified)
     variant positions, so no residue-number offset can creep in.
  2. Build the mutant residue from ideal internal geometry (PeptideBuilder),
     superimpose its N/CA/C atoms onto the native backbone and keep the new
     side chain.
  3. Scan side-chain rotamers (chi angles at -60, 60 and 180 degrees) and keep
     the one with the least steric overlap with the surrounding atoms.
  4. Crop both models to residues within CROP_RADIUS of the site and relax
     them identically with OpenMM
     (Amber14 + OBC2 implicit solvent): hydrogens are added, atoms more than
     RELAX_RADIUS from the site are held by harmonic restraints, and the
     energy is minimised.

The relaxed WT/mutant pair is what stability_analysis.py compares.

Usage:
    python -m src.structure.mutate_pdb --gene ACTC1 --variant E101K
"""
import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser, Select, Superimposer
from Bio.PDB.Atom import Atom
from Bio.PDB.Polypeptide import index_to_one, three_to_index
from PeptideBuilder import Geometry, initialize_res

from src.config import RAW_DIR, OUTPUTS_DIR, load_gene_panel

ALPHAFOLD_DIR = RAW_DIR / "alphafold"
STRUCT_OUT = OUTPUTS_DIR / "structures"
RELAX_RADIUS = 10.0  # A; atoms beyond this are restrained during minimisation
CHI_VALUES = (-60.0, 60.0, 180.0)
N_CHI = {"R": 4, "K": 4, "Q": 3, "E": 3, "M": 3, "H": 2, "F": 2, "Y": 2, "W": 2, "D": 2,
         "N": 2, "L": 2, "I": 2, "C": 1, "S": 1, "T": 1, "V": 1, "P": 0, "A": 0, "G": 0}
BACKBONE = {"N", "CA", "C", "O", "OXT"}
ONE_TO_THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
                "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
                "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
CROP_RADIUS = 20.0  # A; residues with any atom within this distance are kept for relaxation


def one_letter(resname: str) -> str:
    return index_to_one(three_to_index(resname))


def alphafold_model_path(gene: str) -> Path:
    uid = {g["symbol"]: g["uniprot_id"] for g in load_gene_panel("both")}[gene]
    manifest = json.load(open(ALPHAFOLD_DIR / "manifest.json"))
    entry = next((m for m in manifest if m["uniprot_id"] == uid), None)
    if entry is None:
        raise FileNotFoundError(f"No AlphaFold model for {gene}; run fetch_alphafold first.")
    return ALPHAFOLD_DIR / entry["pdb"]


def chi_groups(geo) -> list[list[str]]:
    """Group the geometry's side-chain dihedral attributes by rotation axis
    (their first three atoms), in order: chi1, chi2, ..."""
    groups: dict[str, list[str]] = {}
    for name in vars(geo):
        if not name.endswith("_diangle") or name in ("N_CA_C_O_diangle", "N_C_CA_CB_diangle"):
            continue
        axis = "_".join(name.split("_")[:3])
        groups.setdefault(axis, []).append(name)
    return list(groups.values())


def build_side_chain(mut_aa: str, chis: tuple) -> dict:
    """Ideal-geometry residue of type mut_aa with the requested chi angles."""
    geo = Geometry.geometry(mut_aa)
    for group, value in zip(chi_groups(geo), chis):
        delta = value - getattr(geo, group[0])
        for name in group:  # dependent atoms (e.g. ring partners) move together
            setattr(geo, name, getattr(geo, name) + delta)
    structure = initialize_res(geo)
    return {atom.get_id(): atom for atom in structure.get_atoms()}


def place_side_chain(target_res, built: dict) -> list[Atom]:
    sup = Superimposer()
    sup.set_atoms([target_res[a] for a in ("N", "CA", "C")], [built[a] for a in ("N", "CA", "C")])
    rot, tran = sup.rotran
    atoms = []
    for name, atom in built.items():
        if name in BACKBONE:
            continue
        new = atom.copy()
        new.transform(rot, tran)
        new.set_bfactor(0.0)
        atoms.append(new)
    return atoms


def clash_score(side_chain: list[Atom], env_coords: np.ndarray) -> float:
    if len(side_chain) == 0 or len(env_coords) == 0:
        return 0.0
    sc = np.array([a.coord for a in side_chain])
    d = np.sqrt(((sc[:, None, :] - env_coords[None, :, :]) ** 2).sum(-1))
    overlap = np.clip(3.3 - d, 0, None)
    return float((overlap ** 2).sum())


def build_mutant(structure, chain_id: str, position: int, wt_aa: str, mut_aa: str) -> dict:
    chain = structure[0][chain_id]
    residue = chain[position]
    found = one_letter(residue.get_resname())
    if found != wt_aa:
        raise ValueError(f"Residue {position} is {found}, expected {wt_aa}; check numbering.")

    env = [a for a in structure.get_atoms()
           if a.get_parent() is not residue and a.element != "H"
           and np.linalg.norm(a.coord - residue["CA"].coord) < 16.0]
    # Exclude the immediate neighbours' backbone atoms bonded to this residue.
    env = [a for a in env if not (abs(a.get_parent().id[1] - position) == 1 and a.get_id() in ("C", "N"))]
    env_coords = np.array([a.coord for a in env])

    best = None
    for chis in itertools.product(CHI_VALUES, repeat=N_CHI.get(mut_aa, 0)):
        side_chain = place_side_chain(residue, build_side_chain(mut_aa, chis))
        score = clash_score(side_chain, env_coords)
        if best is None or score < best[0]:
            best = (score, chis, side_chain)

    for atom in [a for a in residue if a.get_id() not in BACKBONE]:
        residue.detach_child(atom.get_id())
    for atom in best[2]:
        residue.add(atom)
    residue.resname = ONE_TO_THREE[mut_aa]
    return {"rotamer_chis": list(best[1]), "clash_score": round(best[0], 3)}


def add_missing_oxt(structure):
    """AlphaFold PDB files can lack the terminal OXT atom, which the Amber
    C-terminal templates require; place it opposite O around the C-CA axis."""
    for chain in structure[0]:
        last = [r for r in chain if r.id[0] == " "][-1]
        if "OXT" in last:
            continue
        c, ca, o = last["C"].coord, last["CA"].coord, last["O"].coord
        axis = (c - ca) / np.linalg.norm(c - ca)
        v = o - c
        reflected = 2 * np.dot(v, axis) * axis - v
        oxt = c + reflected / np.linalg.norm(reflected) * 1.25
        last.add(Atom("OXT", oxt, 0.0, 1.0, " ", " OXT", None, "O"))


def crop_around_site(structure, position: int, chain_id: str = "A"):
    """Keep residues within CROP_RADIUS of the site, as contiguous segments.

    Relaxing a whole 2,000-residue protein on a CPU takes far too long, and
    atoms beyond RELAX_RADIUS are restrained anyway. Each contiguous run of
    kept residues becomes its own chain (so the force field sees proper
    termini); runs shorter than three residues are dropped. The site keeps
    its chain id "A" and residue number. Applied identically to WT and
    mutant so the two models contain the same atoms apart from the site.
    """
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure

    chain = structure[0][chain_id]
    site = chain[position]
    site_coords = np.array([a.coord for a in site])
    keep = []
    for res in chain:
        coords = np.array([a.coord for a in res])
        if np.min(np.linalg.norm(coords[:, None, :] - site_coords[None, :, :], axis=-1)) < CROP_RADIUS:
            keep.append(res)
    segments, current = [], []
    for res in keep:
        if current and res.id[1] != current[-1].id[1] + 1:
            segments.append(current)
            current = []
        current.append(res)
    segments.append(current)

    new = Structure("crop")
    model = Model(0)
    new.add(model)
    other_ids = iter("BCDEFGHIJKLMNOPQRSTUVWXYZ")
    for seg in segments:
        has_site = any(r.id[1] == position for r in seg)
        if len(seg) < 3 and not has_site:
            continue
        ch = Chain("A" if has_site else next(other_ids))
        for res in seg:
            ch.add(res.copy())
        model.add(ch)
    add_missing_oxt(new)
    return new


class HeavyAtoms(Select):
    def accept_atom(self, atom):
        return atom.element != "H"


def relax(in_pdb: Path, out_pdb: Path, site_position: int, chain_id: str = "A") -> dict:
    """Restrained energy minimisation around the site with OpenMM."""
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    pdb = app.PDBFile(str(in_pdb))
    forcefield = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=7.0)
    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1.2 * unit.nanometer, constraints=app.HBonds,
    )
    positions = modeller.positions
    site_atoms = [a.index for a in modeller.topology.atoms()
                  if a.residue.chain.id == chain_id and int(a.residue.id) == site_position]
    site_center = np.mean([positions[i].value_in_unit(unit.angstrom) for i in site_atoms], axis=0)

    restraint = openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", 1000.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    n_free = 0
    for atom in modeller.topology.atoms():
        xyz = positions[atom.index].value_in_unit(unit.angstrom)
        if atom.element.symbol != "H" and np.linalg.norm(np.array(xyz) - site_center) > RELAX_RADIUS:
            restraint.addParticle(atom.index, positions[atom.index].value_in_unit(unit.nanometer))
        else:
            n_free += 1
    system.addForce(restraint)

    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    sim = app.Simulation(modeller.topology, system, integrator, openmm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(positions)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    sim.minimizeEnergy(maxIterations=1000)
    state = sim.context.getState(getEnergy=True, getPositions=True)
    e1 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    with open(out_pdb, "w") as f:
        app.PDBFile.writeFile(sim.topology, state.getPositions(), f, keepIds=True)
    return {"energy_before_kj": round(e0, 1), "energy_after_kj": round(e1, 1), "free_atoms": n_free}


def run(gene: str, variant: str) -> dict:
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", variant)
    if not m:
        raise ValueError("variant must look like E101K")
    wt_aa, position, mut_aa = m.group(1), int(m.group(2)), m.group(3)
    STRUCT_OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{gene}_{variant}"

    parser = PDBParser(QUIET=True)
    model_path = alphafold_model_path(gene)
    wt = parser.get_structure("wt", model_path)
    mut = parser.get_structure("mut", model_path)
    info = build_mutant(mut, "A", position, wt_aa, mut_aa)
    for name, s in (("wt", wt), ("mutant", mut)):
        io = PDBIO()
        io.set_structure(s)
        io.save(str(STRUCT_OUT / f"{tag}_{name}_full.pdb"), HeavyAtoms())
    wt, mut = crop_around_site(wt, position), crop_around_site(mut, position)

    io = PDBIO()
    raw = {}
    for name, s in (("wt", wt), ("mutant", mut)):
        raw[name] = STRUCT_OUT / f"{tag}_{name}_unrelaxed.pdb"
        io.set_structure(s)
        io.save(str(raw[name]), HeavyAtoms())

    result = {"gene": gene, "variant": variant, "model": model_path.name, **info}
    for name in ("wt", "mutant"):
        out = STRUCT_OUT / f"{tag}_{name}.pdb"
        result[f"relax_{name}"] = relax(raw[name], out, position)
        print(f"{tag} {name}: {result[f'relax_{name}']}")
    with open(STRUCT_OUT / f"{tag}_build.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Build and relax a mutant model")
    parser.add_argument("--gene", required=True)
    parser.add_argument("--variant", required=True, help="e.g. E101K (UniProt numbering)")
    args = parser.parse_args()
    print(json.dumps(run(args.gene, args.variant), indent=2))


if __name__ == "__main__":
    main()
