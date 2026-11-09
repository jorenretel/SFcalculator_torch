"""Frozen integration contract for replacing the dense structure-factor kernel."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import SFC_Torch
import torch

from tests.fprotein_reference import (
    CASES,
    PARAMETERS,
    REFERENCE_DIR,
    calculate,
    load_case,
    make_calculator,
    parameter_tensors,
    projected_loss,
)

# Absolute tolerances matter for cancellation/near-zero values. These are
# elementwise comparisons, not amplitude correlation or a single aggregate loss.
OUTPUT_RTOL = 3e-4
OUTPUT_ATOL = 3e-3
GRAD_RTOL = 5e-4
GRAD_ATOL = {"positions": 3e-4, "b_iso": 3e-6, "aniso_u": 3e-4, "occupancy": 3e-4}
DEVICES = [
    "cpu",
    pytest.param(
        "cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


@pytest.fixture(scope="module", autouse=True)
def reference_contract():
    # Running against an installed wheel would falsely pass while the edited
    # source is never exercised. The source checkout must be on PYTHONPATH.
    root = Path(__file__).resolve().parents[1]
    assert Path(SFC_Torch.__file__).resolve().parent == root / "SFC_Torch"
    manifest = json.loads((REFERENCE_DIR / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert set(manifest["cases"]) == set(CASES)
    model_source = manifest["model_source"]
    assert model_source["public_target"] == "1LJ5"
    model_bytes = (REFERENCE_DIR / model_source["file"]).read_bytes()
    assert hashlib.sha256(model_bytes).hexdigest() == model_source["sha256"]
    assert {line[:6].strip() for line in model_bytes.decode().splitlines()} <= {
        "CRYST1",
        "ATOM",
        "HETATM",
        "ANISOU",
        "TER",
        "END",
    }
    for name in CASES:
        content = (REFERENCE_DIR / f"{name}.npz").read_bytes()
        assert hashlib.sha256(content).hexdigest() == manifest["cases"][name]["sha256"]
    # Avoid oversubscribing small matrix operations on CI/macOS.
    previous_threads = torch.get_num_threads()
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.set_num_threads(previous_threads)


def assert_close(actual, expected, *, parameter=None):
    actual = actual.detach().cpu().numpy()
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=GRAD_RTOL if parameter else OUTPUT_RTOL,
        atol=GRAD_ATOL[parameter] if parameter else OUTPUT_ATOL,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("return_value", [True, False])
def test_fprotein_forward_reference(case, device, return_value):
    """The detached RSCC path and both public return modes must remain valid."""
    data = load_case(case)
    calculator = make_calculator(data, device)
    parameters = parameter_tensors(data, device, requires_grad=False)
    with torch.no_grad():
        output = calculate(calculator, parameters, return_value=return_value)
    if return_value:
        assert_close(output, data["expected_f_hkl"])
    else:
        assert output is None
    assert_close(calculator.Fprotein_asu, data["expected_f_asu"])
    assert_close(calculator.Fprotein_HKL, data["expected_f_hkl"])
    np.testing.assert_array_equal(calculator.Hasu_array, data["expected_hasu"])
    np.testing.assert_array_equal(calculator.HKL_array, data["expected_hkl"])
    assert torch.count_nonzero(calculator.Fmask_HKL).item() == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("probe_index", [0, 1])
def test_fprotein_backward_reference(case, device, probe_index):
    """Compare VJPs for coordinates, Biso, all U entries, and occupancies."""
    data = load_case(case)
    calculator = make_calculator(data, device)
    parameters = parameter_tensors(data, device)
    output = calculate(calculator, parameters)
    assert_close(output, data["expected_f_hkl"])
    probe = torch.tensor(data["probes"][probe_index], device=device)
    loss = projected_loss(output, probe)
    assert_close(loss, data["expected_losses"][probe_index])
    derivatives = torch.autograd.grad(loss, tuple(parameters.values()))
    for index, name in enumerate(PARAMETERS):
        assert_close(derivatives[index], data[f"expected_grad_{name}"][probe_index], parameter=name)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", CASES)
def test_default_and_partial_overrides(case, device):
    """Stored constructor values and setter state must agree with the baseline."""
    data = load_case(case)
    calculator = make_calculator(data, device)
    with torch.no_grad():
        assert_close(calculator.calc_fprotein(Return=True), data["expected_f_hkl"])
        # Structure factors are linear in occupancies. A partial override must
        # use the existing coordinates/B factors, not stale or default values.
        occupancy = torch.tensor(data["occupancy"] * 0.5, device=device)
        output = calculator.calc_fprotein(atoms_occ_tensor=occupancy, Return=True)
        assert_close(output, data["expected_f_hkl"] * 0.5)
        assert_close(calculator.calc_fprotein(Return=True), data["expected_f_hkl"] * 0.5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("coordinates_only", [True, False])
def test_repeated_forward_backward(case, device, coordinates_only):
    """A reused calculator must not cache stale inputs or an old autograd graph."""
    data = load_case(case)
    calculator = make_calculator(data, device)
    probe = torch.tensor(data["probes"][0], device=device)
    for scale in (1.0, 0.5):
        parameters = parameter_tensors(data, device, requires_grad=not coordinates_only)
        parameters["positions"].requires_grad_(True)
        # New leaf tensors each iteration, as in refinement. Scaling occupancy
        # gives an independent exact prediction without another implementation.
        parameters["occupancy"] = torch.tensor(
            data["occupancy"] * scale, device=device, requires_grad=not coordinates_only
        )
        output = calculate(calculator, parameters)
        assert_close(output, data["expected_f_hkl"] * scale)
        loss = projected_loss(output, probe)
        loss.backward()
        for name, parameter in parameters.items():
            if coordinates_only and name != "positions":
                assert parameter.grad is None
                continue
            expected_scale = 1.0 if name == "occupancy" else scale
            assert parameter.grad is not None
            assert_close(
                parameter.grad,
                data[f"expected_grad_{name}"][0] * expected_scale,
                parameter=name,
            )


@pytest.mark.parametrize("case", CASES)
def test_reference_fixture_is_nontrivial(case):
    """Guard against accidentally replacing the baseline with degenerate data."""
    data = load_case(case)
    assert len(data["hkl"]) == 257
    assert len(np.unique(data["hkl"], axis=0)) == 257
    assert np.ptp(data["expected_f_hkl"].real) > 1
    assert np.ptp(data["expected_f_hkl"].imag) > 1
    for name in ("positions", "b_iso", "occupancy"):
        assert np.any(np.abs(data[f"expected_grad_{name}"]) > 1e-4)
    if case.endswith("mixed") or case.endswith("anomalous"):
        assert np.any(data["aniso_u"] != 0)
        assert np.any(np.all(data["aniso_u"] == 0, axis=(1, 2)))
        assert np.any(np.abs(data["expected_grad_aniso_u"]) > 1e-4)
    if case == "cryo_1lj5":
        assert len(data["positions"]) == 3012
        np.testing.assert_allclose(data["cell"], [156.57, 46.88, 63.05, 90, 106.54, 90])


@pytest.mark.parametrize("case", CASES)
def test_coordinate_gradient_finite_difference(case):
    """Independent numerical check of one nonzero coordinate derivative."""
    data = load_case(case)
    expected = data["expected_grad_positions"][0]
    # The largest component avoids an ill-conditioned relative comparison.
    index = np.unravel_index(np.abs(expected).argmax(), expected.shape)
    losses = []
    calculator = make_calculator(data)
    probe = torch.from_numpy(data["probes"][0])
    step = 0.01  # Angstrom; large enough for float32 phase roundoff.
    for delta in (-step, step):
        parameters = parameter_tensors(data, requires_grad=False)
        parameters["positions"][index] += delta
        with torch.no_grad():
            losses.append(projected_loss(calculate(calculator, parameters), probe).item())
    finite_difference = (losses[1] - losses[0]) / (2 * step)
    np.testing.assert_allclose(finite_difference, expected[index], rtol=1e-2, atol=1e-2)
