"""Explicit, source-pinned generation of the pre-optimization numerical baseline.

Run from the repository root with ``python -m tests.generate_fprotein_reference``.
This is not invoked by pytest. See data/fprotein_reference/README.md.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch
from SFC_Torch.io import PDBParser

from tests.fprotein_reference import (
    PARAMETERS,
    REFERENCE_DIR,
    calculate,
    make_calculator,
    parameter_tensors,
    projected_loss,
)

BASELINE_COMMIT = "0b583b837c5ed94994e239669260c3d8c61fa1b7"
SOURCE_HASHES = {
    "Fmodel.py": "eaabefed8161420e1e05b25b5bf53db4fdf835f46e92a6d069ba763296f60a1a",
    "utils.py": "3524017de20cce10b8e336294f8f2324145e35faafc5da15d855b06a58f78b08",
    "symmetry.py": "f83690b8260be22589df4cd63c4440a8cbfdc1cb0b773eb2eae8c2feee5ec5bb",
    "io.py": "cda2783eee3fbd9bd8a9d95ca472d794a5dfadf405d96c16db4be1370e974a07",
}
SEED = 20261109


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_hkl(cell, spacegroup, rng, anomalous=False):
    candidates = gemmi.make_miller_array(cell, gemmi.SpaceGroup(spacegroup), 3.0, 0.0, unique=True)
    # Span the resolution range, including the endpoints, with deliberately
    # unsorted input and a non-power-of-two length to expose ordering/tail bugs.
    order = np.argsort(cell.calculate_1_d2_array(candidates), kind="stable")
    selected = candidates[order[np.linspace(0, len(order) - 1, 257, dtype=int)]]
    if anomalous:
        selected[1::2] *= -1
    return selected[rng.permutation(len(selected))].copy()


def synthetic_case(cell, spacegroup, mode, rng, mixed=False, anomalous=False):
    cell = gemmi.UnitCell(*cell)
    elements = np.array(["C", "N", "O", "S", "C", "H", "P", "Se", "Fe", "Zn", "O"])
    n_atoms = len(elements)
    fractional = rng.uniform(0.07, 0.93, (n_atoms, 3))
    positions = fractional @ np.array(cell.orthogonalization_matrix.tolist()).T
    aniso = np.zeros((n_atoms, 3, 3), dtype=np.float32)
    if mixed:
        for i in range(1, n_atoms, 2):
            matrix = rng.normal(0.0, 0.12, (3, 3))
            aniso[i] = matrix @ matrix.T + np.eye(3) * 0.02
    occupancy = np.linspace(0.0, 1.0, n_atoms, dtype=np.float32)
    return {
        "cell": np.array(cell.parameters),
        "spacegroup": np.array(spacegroup),
        "mode": np.array(mode),
        "anomalous": np.array(anomalous),
        "wavelength": np.array(1.0),
        "elements": elements,
        "positions": positions.astype(np.float32),
        "b_iso": np.linspace(0.0, 80.0, n_atoms, dtype=np.float32),
        "aniso_u": aniso,
        "occupancy": occupancy,
        "hkl": sample_hkl(cell, spacegroup, rng, anomalous),
    }


def tutorial_case(rng):
    model = PDBParser(str(REFERENCE_DIR / "1lj5.pdb"))
    cell = model.cell
    # Exercise the electron-scattering path with public tutorial coordinates.
    # P1 is an intentional test setup, not the experimental symmetry of 1LJ5.
    return {
        "cell": np.array(cell.parameters),
        "spacegroup": np.array("P 1"),
        "mode": np.array("cryoem"),
        "anomalous": np.array(False),
        "wavelength": np.array(1.0),
        "elements": np.array(model.atom_name),
        "positions": model.atom_pos.astype(np.float32),
        "b_iso": model.atom_b_iso.astype(np.float32),
        "aniso_u": model.atom_b_aniso.astype(np.float32),
        "occupancy": model.atom_occ.astype(np.float32),
        "hkl": sample_hkl(cell, "P 1", rng),
    }


def capture(data, rng):
    calculator = make_calculator(data)
    parameters = parameter_tensors(data)
    output = calculate(calculator, parameters)
    probes = (
        rng.normal(size=(2, output.numel())) + 1j * rng.normal(size=(2, output.numel()))
    ).astype(np.complex64)
    result = dict(data)
    result.update(
        expected_hasu=calculator.Hasu_array,
        expected_hkl=calculator.HKL_array,
        expected_f_asu=calculator.Fprotein_asu.detach().numpy(),
        expected_f_hkl=output.detach().numpy(),
        probes=probes,
    )
    losses = []
    gradients = {name: [] for name in PARAMETERS}
    for i, probe in enumerate(probes):
        loss = projected_loss(output, torch.from_numpy(probe))
        derivatives = torch.autograd.grad(loss, tuple(parameters.values()), retain_graph=i == 0)
        losses.append(loss.item())
        for index, name in enumerate(PARAMETERS):
            gradients[name].append(derivatives[index].numpy())
    result["expected_losses"] = np.array(losses, dtype=np.float32)
    for name in PARAMETERS:
        result[f"expected_grad_{name}"] = np.stack(gradients[name])
    for key, value in result.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Non-finite reference array: {key}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--write-reference", action="store_true", required=True)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for name, expected in SOURCE_HASHES.items():
        if sha256(root / "SFC_Torch" / name) != expected:
            parser.error(
                f"{name} differs from the pinned legacy source; refusing to bless new results"
            )
    if (REFERENCE_DIR / "manifest.json").exists():
        parser.error(
            "Reference already exists; remove it explicitly only after "
            "reviewing the baseline change"
        )
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    cases = {
        "cryo_1lj5": tutorial_case(rng),
        "cryo_p1_iso": synthetic_case((31, 37, 43, 90, 90, 90), "P 1", "cryoem", rng),
        "cryo_p21_mixed": synthetic_case(
            (31, 37, 43, 90, 103, 90), "P 1 21 1", "cryoem", rng, mixed=True
        ),
        "xray_p212121_mixed": synthetic_case(
            (31, 37, 43, 90, 90, 90), "P 21 21 21", "xray", rng, mixed=True
        ),
        "xray_p1_anomalous": synthetic_case(
            (31, 37, 43, 90, 90, 90), "P 1", "xray", rng, mixed=True, anomalous=True
        ),
    }
    manifest = {
        "schema_version": 1,
        "baseline_commit": BASELINE_COMMIT,
        "source_sha256": SOURCE_HASHES,
        "seed": SEED,
        "device": "cpu",
        "threads": 1,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "gemmi": gemmi.__version__,
            "reciprocalspaceship": rs.__version__,
        },
        "model_source": {
            "file": "1lj5.pdb",
            "sha256": sha256(REFERENCE_DIR / "1lj5.pdb"),
            "public_target": "1LJ5",
            "target_url": "https://www.rcsb.org/structure/1LJ5",
            "description": "Aligned predicted model from the public 1LJ5 tutorial target; "
            "only coordinate, displacement, cell, and terminator records retained. "
            "Electron scattering in P1 is a test setup, not experimental data.",
        },
        "cases": {},
    }
    for name, data in cases.items():
        result = capture(data, rng)
        path = REFERENCE_DIR / f"{name}.npz"
        np.savez_compressed(path, **result)
        manifest["cases"][name] = {
            "sha256": sha256(path),
            "atoms": len(data["positions"]),
            "reflections": len(result["expected_f_hkl"]),
            "asu_reflections": len(result["expected_f_asu"]),
            "symmetry_operations": len(gemmi.SpaceGroup(str(data["spacegroup"])).operations()),
        }
        print(f"Captured {name}: {manifest['cases'][name]}", flush=True)
    (REFERENCE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
