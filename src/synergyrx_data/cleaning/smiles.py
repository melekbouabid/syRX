"""SMILES standardisation with RDKit.

Pipeline: parse -> clean up -> keep the parent fragment (drop salts and solvents) ->
neutralise -> canonical SMILES and InChIKey. A SMILES that cannot be processed is returned
with `valid=False` and a reason; it is never dropped here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")  # RDKit prints every parse failure; we record them instead

# Metal-containing drugs (cisplatin, carboplatin, lithium salts, ...) are often written as
# disconnected fragments. Keeping only the largest fragment would throw the metal away.
_METALS = frozenset(
    {
        "Li", "Al", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
        "Sr", "Y", "Zr", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Ba",
        "La", "Gd", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    }
)  # fmt: skip


@dataclass(frozen=True)
class StandardizedSmiles:
    input: str | None
    smiles: str | None
    inchikey: str | None
    valid: bool
    issue: str | None = None
    note: str | None = None


def _invalid(smiles: str | None, issue: str) -> StandardizedSmiles:
    return StandardizedSmiles(smiles, None, None, False, issue)


@lru_cache(maxsize=200_000)
def standardize_smiles(smiles: str | None) -> StandardizedSmiles:
    if smiles is None or not str(smiles).strip():
        return _invalid(smiles, "missing")
    text = str(smiles).strip()
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return _invalid(text, "unparseable SMILES")
    if mol.GetNumAtoms() == 0:
        return _invalid(text, "empty molecule")
    notes: list[str] = []
    try:
        mol = rdMolStandardize.Cleanup(mol)
        fragments = Chem.GetMolFrags(mol)
        has_metal = any(atom.GetSymbol() in _METALS for atom in mol.GetAtoms())
        if len(fragments) > 1:
            if has_metal:
                notes.append("metal complex: all fragments kept")
            else:
                mol = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
                notes.append("salt or solvent removed")
        if not has_metal:
            before = Chem.GetFormalCharge(mol)
            mol = rdMolStandardize.Uncharger().uncharge(mol)
            if Chem.GetFormalCharge(mol) != before:
                notes.append("neutralised")
        Chem.SanitizeMol(mol)
        canonical = Chem.MolToSmiles(mol)
        inchikey = Chem.MolToInchiKey(mol) or None
    except Exception as exc:  # RDKit raises several unrelated exception types
        return _invalid(text, f"standardisation failed: {type(exc).__name__}")
    if not canonical or inchikey is None:
        return _invalid(text, "no canonical form or InChIKey")
    return StandardizedSmiles(text, canonical, inchikey, True, None, "; ".join(notes) or None)


def connectivity_block(inchikey: str | None) -> str | None:
    """First 14 characters of an InChIKey: same skeleton regardless of stereo or charge."""
    return inchikey[:14] if inchikey else None
