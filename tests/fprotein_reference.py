"""Shared fixture adapter; deliberately contains no structure-factor implementation."""

from pathlib import Path

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch
from SFC_Torch.Fmodel import SFcalculator
from SFC_Torch.io import PDBParser

REFERENCE_DIR = Path(__file__).parent / "data" / "fprotein_reference"
PARAMETERS = ("positions", "b_iso", "aniso_u", "occupancy")
CASES = (
    "cryo_1lj5",
    "cryo_p1_iso",
    "cryo_p21_mixed",
    "xray_p212121_mixed",
    "xray_p1_anomalous",
)


def load_case(name):
    with np.load(REFERENCE_DIR / f"{name}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def make_calculator(data, device="cpu"):
    """Exercise the public constructor, including scattering and ASU/HKL mapping."""
    structure = gemmi.Structure()
    structure.cell = gemmi.UnitCell(*data["cell"])
    structure.spacegroup_hm = str(data["spacegroup"])
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "UNK"
    residue.seqid = gemmi.SeqId(1, " ")
    for i, element in enumerate(data["elements"]):
        atom = gemmi.Atom()
        atom.name = f"{element}{i}"
        atom.element = gemmi.Element(str(element))
        atom.pos = gemmi.Position(*data["positions"][i])
        atom.b_iso = float(data["b_iso"][i])
        atom.occ = float(data["occupancy"][i])
        u = data["aniso_u"][i]
        atom.aniso = gemmi.SMat33f(u[0, 0], u[1, 1], u[2, 2], u[0, 1], u[0, 2], u[1, 2])
        residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    reflections = rs.DataSet(
        {axis: rs.DataSeries(data["hkl"][:, i], dtype="H") for i, axis in enumerate("HKL")},
        cell=structure.cell,
        spacegroup=structure.spacegroup_hm,
    ).set_index(["H", "K", "L"])
    return SFcalculator(
        PDBParser(structure),
        mtzdata=reflections,
        n_bins=4,
        set_experiment=False,
        expcolumns=[],
        mode=str(data["mode"]),
        anomalous=bool(data["anomalous"]),
        wavelength=float(data["wavelength"]),
        device=device,
    )


def parameter_tensors(data, device="cpu", requires_grad=True):
    return {
        name: torch.tensor(
            data[name], device=device, dtype=torch.float32, requires_grad=requires_grad
        )
        for name in PARAMETERS
    }


def calculate(calculator, parameters, return_value=True):
    result = calculator.calc_fprotein(
        atoms_position_tensor=parameters["positions"],
        atoms_biso_tensor=parameters["b_iso"],
        atoms_aniso_uw_tensor=parameters["aniso_u"],
        atoms_occ_tensor=parameters["occupancy"],
        Return=return_value,
    )
    return result


def projected_loss(fprotein, probe):
    """Real scalar with independent real/imaginary weights, not just amplitudes."""
    return (fprotein.conj() * probe).real.sum() / fprotein.numel() ** 0.5
