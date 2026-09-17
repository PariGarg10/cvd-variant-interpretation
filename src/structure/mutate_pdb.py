from pathlib import Path

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Polypeptide import protein_letters_3to1


INPUT_PDB = Path("outputs/ACTC1_R39H_wt.pdb")
OUTPUT_PDB = Path("outputs/ACTC1_R39H_mutant.pdb")

TARGET_POSITION = 39
WT_AA = "R"
MUT_AA = "H"


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


parser = PDBParser(QUIET=True)
structure = parser.get_structure("ACTC1", INPUT_PDB)

found = False

for model in structure:

    for chain in model:

        for residue in chain:

            if residue.id[1] != TARGET_POSITION:
                continue

            if residue.resname not in THREE_TO_ONE:
                continue

            current_aa = THREE_TO_ONE[residue.resname]

            print(
                f"Found chain {chain.id}, "
                f"position {TARGET_POSITION}: "
                f"{residue.resname} ({current_aa})"
            )

            if current_aa != WT_AA:
                continue

            # Change residue identity.
            residue.resname = "HIS"

            found = True

            print(
                f"Applied {WT_AA}{TARGET_POSITION}{MUT_AA} "
                f"on chain {chain.id}"
            )

            break

        if found:
            break

    if found:
        break


if not found:
    raise ValueError(
        f"Could not find {WT_AA}{TARGET_POSITION} "
        "in the structure."
    )


io = PDBIO()
io.set_structure(structure)
io.save(str(OUTPUT_PDB))

print(f"Saved mutant structure to {OUTPUT_PDB}")