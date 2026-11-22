"""Chemistry and hidden structure-activity model for the scaffold-hop task.

Compounds are two-substituent derivatives of fused heteroaromatic cores,
a common shape for kinase-hinge-binder series. A core has two attachment
points, labelled 1 and 2; an R1 and an R2 fragment are joined with RDKit's
`molzip`. The hidden activity of a compound is

    pIC50 = 6 + offset(core)                         series potency shift
              + g1(R1) + g2(R2)                      transferable effects, from
                                                     fragment properties
              + e(core, R1) + e(core, R2)            series-specific effects
                                                     (do not transfer to a new core)

The transferable part is a smooth function of fragment lipophilicity, size
and hydrogen-bond donors, drawn per seed, so a model that sees substituents
as chemistry (not as bit patterns tied to one core) can carry it to a new
core. The series-specific part rewards models that memorise a core, and is
what makes a random split overstate how good a model is on a new core.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

# Cores with attachment points [*:1] and [*:2].
CORES: dict[str, str] = {
    "quinazoline": "[*:1]c1nc([*:2])c2ccccc2n1",
    "pyrazolopyrimidine": "[*:1]c1nc([*:2])c2cn[nH]c2n1",
    "thienopyrimidine": "[*:1]c1nc([*:2])c2ccsc2n1",
    "purine": "[*:1]c1nc([*:2])c2nc[nH]c2n1",
    "pyrrolopyrimidine": "[*:1]c1nc([*:2])c2cc[nH]c2n1",
    "furopyrimidine": "[*:1]c1nc([*:2])c2ccoc2n1",
}

R1_GROUPS: tuple[str, ...] = (
    "[*:1]N1CCOCC1",
    "[*:1]N1CCNCC1",
    "[*:1]NC",
    "[*:1]NC1CC1",
    "[*:1]N1CCCC1",
    "[*:1]Nc1ccccc1",
    "[*:1]Nc1ccc(F)cc1",
    "[*:1]Nc1ccc(Cl)cc1",
    "[*:1]Nc1ccc(OC)cc1",
    "[*:1]NCc1ccccc1",
    "[*:1]N1CCC(O)CC1",
    "[*:1]N(C)C",
    "[*:1]NCCO",
    "[*:1]NC1CCCCC1",
)
R2_GROUPS: tuple[str, ...] = (
    "[*:2]c1ccccc1",
    "[*:2]c1ccc(F)cc1",
    "[*:2]c1ccc(C)cc1",
    "[*:2]c1ccncc1",
    "[*:2]c1ccc(OC)cc1",
    "[*:2]c1ccc(C(F)(F)F)cc1",
    "[*:2]C",
    "[*:2]CC",
    "[*:2]C1CC1",
    "[*:2]c1cccs1",
    "[*:2]c1ccc(N)cc1",
    "[*:2]c1ccc(C#N)cc1",
    "[*:2]c1ccc(O)cc1",
    "[*:2]c1ccc(Cl)cc1",
)

SALTS = (".Cl", ".Cl.O", ".[Na+].[Cl-]", ".OC(=O)C(F)(F)F")


@dataclass(frozen=True)
class FragmentProps:
    logp: float
    mr: float
    hbd: float


@cache
def fragment_props(smiles_with_dummy: str) -> FragmentProps:
    """Properties of a fragment capped with hydrogen."""
    capped = Chem.MolFromSmiles(smiles_with_dummy.replace("[*:1]", "[H]").replace("[*:2]", "[H]"))
    assert capped is not None, smiles_with_dummy
    capped = Chem.AddHs(capped)
    return FragmentProps(
        Crippen.MolLogP(capped),  # type: ignore[attr-defined]
        Crippen.MolMR(capped),  # type: ignore[attr-defined]
        float(rdMolDescriptors.CalcNumHBD(capped)),
    )


@cache
def _core_mol(core_smiles: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(core_smiles)
    assert m is not None, core_smiles
    return m


def assemble(core: str, r1: str, r2: str) -> Chem.Mol:
    """Join a core with its two substituents; returns a sanitised molecule."""
    combo = Chem.CombineMols(
        Chem.CombineMols(_core_mol(CORES[core]), Chem.MolFromSmiles(r1)),
        Chem.MolFromSmiles(r2),
    )
    mol = Chem.molzip(combo)
    Chem.SanitizeMol(mol)
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # round trip: ring info, canonical form


def canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol)


def reference_cores() -> dict[str, int]:
    """Sanity: every core parses and has exactly two attachment points."""
    out = {}
    for name, smi in CORES.items():
        m = _core_mol(smi)
        out[name] = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 0)
    return out


@dataclass(frozen=True)
class Sar:
    """Hidden structure-activity function of one seed."""

    w_logp1: float
    w_hbd1: float
    w_size1: float
    opt_mr1: float
    w_logp2: float
    w_hbd2: float
    w_size2: float
    opt_mr2: float
    offset: dict[str, float]
    series_r1: dict[tuple[str, str], float]
    series_r2: dict[tuple[str, str], float]

    def g(self, which: int, frag: str) -> float:
        p = fragment_props(frag)
        if which == 1:
            return (
                self.w_logp1 * p.logp
                + self.w_hbd1 * p.hbd
                - self.w_size1 * ((p.mr - self.opt_mr1) / 10.0) ** 2
            )
        return (
            self.w_logp2 * p.logp
            + self.w_hbd2 * p.hbd
            - self.w_size2 * ((p.mr - self.opt_mr2) / 10.0) ** 2
        )

    def pic50(self, core: str, r1: str, r2: str) -> float:
        return (
            6.0
            + self.offset[core]
            + self.g(1, r1)
            + self.g(2, r2)
            + self.series_r1[(core, r1)]
            + self.series_r2[(core, r2)]
        )


def draw_sar(rng: np.random.Generator, series_sd: float, offset_sd: float) -> Sar:
    def shifted(
        names: tuple[str, ...], groups: tuple[str, ...], sd: float
    ) -> dict[tuple[str, str], float]:
        return {(c, g): float(rng.normal(0.0, sd)) for c in names for g in groups}

    names = tuple(CORES)
    return Sar(
        w_logp1=float(rng.uniform(0.15, 0.45)),
        w_hbd1=float(rng.uniform(-0.4, 0.4)),
        w_size1=float(rng.uniform(0.05, 0.25)),
        opt_mr1=float(rng.uniform(25.0, 40.0)),
        w_logp2=float(rng.uniform(0.2, 0.55)),
        w_hbd2=float(rng.uniform(-0.4, 0.4)),
        w_size2=float(rng.uniform(0.05, 0.25)),
        opt_mr2=float(rng.uniform(28.0, 42.0)),
        offset={c: float(rng.normal(0.0, offset_sd)) for c in names},
        series_r1=shifted(names, R1_GROUPS, series_sd),
        series_r2=shifted(names, R2_GROUPS, series_sd),
    )


def molecular_weight(mol: Chem.Mol) -> float:
    return float(Descriptors.MolWt(mol))  # type: ignore[attr-defined]
