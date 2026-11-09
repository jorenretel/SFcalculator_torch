"""Focused streaming contracts supplementing, not replacing, the frozen suite."""

import inspect
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
import torch
from SFC_Torch.Fmodel import SFcalculator
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from tests.fprotein_reference import (
    PARAMETERS,
    calculate,
    load_case,
    make_calculator,
    parameter_tensors,
    projected_loss,
)
from tests.test_fprotein_reference import DEVICES, assert_close


@pytest.fixture(scope="module", autouse=True)
def small_tensor_execution():
    previous_threads = torch.get_num_threads()
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.set_num_threads(previous_threads)


def configured_calculator(data, device="cpu", **configuration):
    # Reuse the frozen fixture's structure/MTZ adapter without changing it. The
    # partial still invokes the real public constructor with the new options.
    with patch(
        "tests.fprotein_reference.SFcalculator",
        partial(SFcalculator, **configuration),
    ):
        return make_calculator(data, device)


class RejectDenseAtomReflection(TorchDispatchMode):
    """Catch transient dense allocations as well as persistent calculator state."""

    def __init__(self, n_atoms, n_reflections):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_reflections = n_reflections

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        tensors, _ = tree_flatten(result)
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                assert not (self.n_atoms in tensor.shape and self.n_reflections in tensor.shape), (
                    f"{func} allocated atom-by-reflection tensor {tuple(tensor.shape)}"
                )
        return result


def assert_compact(calculator):
    n_atoms = calculator.n_atoms
    n_reflections = len(calculator.Hasu_array)
    scattering = calculator._scattering_by_type
    indices = calculator._atom_type_indices
    assert calculator._fullsf_tensor is None
    assert scattering.shape == (len(set(calculator.atom_name)), n_reflections)
    assert scattering.dtype == (torch.complex64 if calculator.anomalous else torch.float32)
    assert indices.shape == (n_atoms,)
    assert indices.dtype == torch.long
    assert scattering.device == calculator.atom_pos_orth.device == indices.device
    assert indices.min().item() >= 0
    assert indices.max().item() < len(scattering)
    assert scattering.numel() + indices.numel() < n_atoms * n_reflections
    assert not scattering.requires_grad
    assert not indices.requires_grad
    for name, value in vars(calculator).items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            assert value.shape not in ((n_atoms, n_reflections), (n_reflections, n_atoms)), name


def test_public_configuration_and_unchanged_calc_signature():
    constructor = inspect.signature(SFcalculator.__init__).parameters
    assert list(constructor)[-2:] == ["fprotein_backend", "reflection_chunk_size"]
    assert constructor["fprotein_backend"].default == "streamed"
    assert constructor["reflection_chunk_size"].default == 2048
    assert list(inspect.signature(SFcalculator.calc_fprotein).parameters) == [
        "self",
        "atoms_position_tensor",
        "atoms_biso_tensor",
        "atoms_aniso_uw_tensor",
        "atoms_occ_tensor",
        "Return",
    ]
    calculator = make_calculator(load_case("cryo_p1_iso"))
    assert calculator.fprotein_backend == "streamed"
    assert calculator.reflection_chunk_size == 2048
    assert_compact(calculator)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["cryo_1lj5", "xray_p212121_mixed", "xray_p1_anomalous"])
def test_compact_initialization_and_forward_do_not_allocate_dense(case, device):
    data = load_case(case)
    with RejectDenseAtomReflection(len(data["positions"]), len(data["expected_hasu"])):
        calculator = configured_calculator(data, device, reflection_chunk_size=31)
        assert_compact(calculator)
        parameters = parameter_tensors(data, device, requires_grad=False)
        with torch.no_grad():
            output = calculate(calculator, parameters)
        assert_close(output, data["expected_f_hkl"])
        assert_close(calculator.Fprotein_asu, data["expected_f_asu"])
        assert_compact(calculator)


# ASU sizes differ from the 257 requested HKLs. The cryo cases exercise exact
# division, one-reflection chunks, and 9/1-reflection tails; X-ray adds symmetry
# expansion and complex anomalous scattering without thousands of size-1 graphs.
CHUNK_CASES = [
    (case, chunk) for case in ("cryo_p1_iso", "cryo_p21_mixed") for chunk in (1, 31, 128, 257, 2048)
] + [(case, chunk) for case in ("xray_p212121_mixed", "xray_p1_anomalous") for chunk in (128, 257)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case,chunk_size", CHUNK_CASES)
def test_chunked_outputs_and_all_four_gradients_match_fixture(case, chunk_size, device):
    data = load_case(case)
    calculator = make_calculator(data, device)
    # Public attributes, not extra calc_fprotein arguments, control dispatch.
    calculator.fprotein_backend = "streamed"
    calculator.reflection_chunk_size = chunk_size
    parameters = parameter_tensors(data, device)
    output = calculate(calculator, parameters)
    assert_close(output, data["expected_f_hkl"])
    assert_close(calculator.Fprotein_asu, data["expected_f_asu"])
    probe_index = 1
    probe = torch.tensor(data["probes"][probe_index], device=device)
    loss = projected_loss(output, probe)
    gradients = torch.autograd.grad(loss, tuple(parameters.values()))
    for index, name in enumerate(PARAMETERS):
        assert_close(gradients[index], data[f"expected_grad_{name}"][probe_index], parameter=name)
    assert calculator._fullsf_tensor is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["dense", "streamed"])
@pytest.mark.parametrize("mutation", ["getter_inplace", "setter"])
@pytest.mark.parametrize("case", ["cryo_p21_mixed", "xray_p1_anomalous"])
def test_public_dense_cache_mutations_override_compact_scattering(case, mutation, backend, device):
    data = load_case(case)
    calculator = configured_calculator(
        data, device, fprotein_backend=backend, reflection_chunk_size=31
    )
    assert_compact(calculator)
    compact_before = calculator._scattering_by_type.clone()
    expected_dense = torch.tensor(
        np.asarray([calculator.full_atomic_sf_asu[atom] for atom in calculator.atom_name]),
        device=device,
        dtype=calculator._scattering_by_type.dtype,
    )
    if mutation == "getter_inplace":
        dense = calculator.fullsf_tensor
        assert calculator._fullsf_tensor is dense
        assert calculator.fullsf_tensor is dense
        torch.testing.assert_close(dense, expected_dense)
        dense.mul_(0.5)
    else:
        # Assignment must work even before the lazy getter has been invoked.
        dense = expected_dense * 0.5
        calculator.fullsf_tensor = dense
        assert calculator._fullsf_tensor is dense
        assert calculator.fullsf_tensor is dense
    torch.testing.assert_close(calculator._scattering_by_type, compact_before)
    parameters = parameter_tensors(data, device)
    output = calculate(calculator, parameters)
    assert_close(output, data["expected_f_hkl"] * 0.5)
    assert_close(calculator.Fprotein_asu, data["expected_f_asu"] * 0.5)
    loss = projected_loss(output, torch.tensor(data["probes"][0], device=device))
    gradients = torch.autograd.grad(loss, tuple(parameters.values()))
    for index, name in enumerate(PARAMETERS):
        assert_close(gradients[index], data[f"expected_grad_{name}"][0] * 0.5, parameter=name)
    assert calculator.fullsf_tensor is dense
    # In-place edits after a completed graph must not be hidden by stale caches.
    dense.zero_()
    with torch.no_grad():
        output = calculate(calculator, parameter_tensors(data, device, requires_grad=False))
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["dense", "streamed"])
@pytest.mark.parametrize("occupancy_only", [True, False])
def test_lazy_cache_first_access_in_inference_mode_allows_edits_and_gradients(
    device, backend, occupancy_only
):
    data = load_case("cryo_p21_mixed")
    calculator = configured_calculator(
        data, device, fprotein_backend=backend, reflection_chunk_size=31
    )
    assert calculator._fullsf_tensor is None
    with torch.inference_mode():
        dense = calculator.fullsf_tensor
        assert calculator._fullsf_tensor is dense
    # The cache outlives inference: public edits and later refinement must work
    # without cloning/replacing it or reinitializing scattering factors.
    dense.mul_(0.5)
    assert calculator.fullsf_tensor is dense
    parameters = parameter_tensors(data, device, requires_grad=not occupancy_only)
    parameters["occupancy"].requires_grad_(True)
    output = calculate(calculator, parameters)
    assert_close(output, data["expected_f_hkl"] * 0.5)
    assert_close(calculator.Fprotein_asu, data["expected_f_asu"] * 0.5)
    loss = projected_loss(output, torch.tensor(data["probes"][0], device=device))
    names = ("occupancy",) if occupancy_only else PARAMETERS
    gradients = torch.autograd.grad(loss, tuple(parameters[name] for name in names))
    for index, name in enumerate(names):
        assert_close(gradients[index], data[f"expected_grad_{name}"][0] * 0.5, parameter=name)
    assert calculator.fullsf_tensor is dense


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("occupancy_only", [True, False])
def test_implicit_dense_inference_cache_allows_later_gradients(device, occupancy_only):
    data = load_case("cryo_p21_mixed")
    calculator = configured_calculator(data, device, fprotein_backend="dense")
    assert calculator._fullsf_tensor is None
    with torch.inference_mode():
        output = calculator.calc_fprotein(Return=True)
        assert_close(output, data["expected_f_hkl"])
    dense = calculator._fullsf_tensor
    assert dense is not None
    parameters = parameter_tensors(data, device, requires_grad=not occupancy_only)
    parameters["occupancy"].requires_grad_(True)
    output = calculate(calculator, parameters)
    assert_close(output, data["expected_f_hkl"])
    assert_close(calculator.Fprotein_asu, data["expected_f_asu"])
    loss = projected_loss(output, torch.tensor(data["probes"][0], device=device))
    names = ("occupancy",) if occupancy_only else PARAMETERS
    gradients = torch.autograd.grad(loss, tuple(parameters[name] for name in names))
    for index, name in enumerate(names):
        assert_close(gradients[index], data[f"expected_grad_{name}"][0], parameter=name)
    assert calculator.fullsf_tensor is dense


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("coordinates_only", [True, False])
def test_streamed_coordinates_only_and_frozen_parameters(device, coordinates_only):
    data = load_case("cryo_p21_mixed")
    calculator = configured_calculator(data, device, reflection_chunk_size=31)
    parameters = parameter_tensors(data, device, requires_grad=False)
    parameters["positions"].requires_grad_(coordinates_only)
    output = calculate(calculator, parameters, return_value=False)
    assert output is None
    assert_close(calculator.Fprotein_HKL, data["expected_f_hkl"])
    assert_close(calculator.Fprotein_asu, data["expected_f_asu"])
    if coordinates_only:
        projected_loss(
            calculator.Fprotein_HKL, torch.tensor(data["probes"][0], device=device)
        ).backward()
        assert_close(
            parameters["positions"].grad, data["expected_grad_positions"][0], parameter="positions"
        )
    else:
        assert not calculator.Fprotein_HKL.requires_grad
        assert not calculator.Fprotein_asu.requires_grad
    for name in PARAMETERS[1:]:
        assert parameters[name].grad is None
    assert calculator._fullsf_tensor is None


@pytest.mark.parametrize("device", DEVICES)
def test_no_mtz_streamed_matches_dense(device):
    data = load_case("cryo_p21_mixed")
    template = make_calculator(data, device)
    outputs = {}
    gradients = {}
    for backend in ("dense", "streamed"):
        calculator = SFcalculator(
            template._pdb,
            dmin=5.0,
            n_bins=4,
            set_experiment=False,
            mode=str(data["mode"]),
            device=device,
            fprotein_backend=backend,
            reflection_chunk_size=31,
        )
        assert calculator.HKL_array is None
        assert_compact(calculator)
        parameters = parameter_tensors(data, device)
        output = calculate(calculator, parameters)
        assert output is calculator.Fprotein_asu
        assert len(output) == len(calculator.Hasu_array) > 31
        assert torch.count_nonzero(calculator.Fmask_asu).item() == 0
        probe = torch.complex(torch.ones_like(output.real), torch.full_like(output.real, 0.25))
        loss = projected_loss(output, probe)
        outputs[backend] = output.detach().cpu().numpy()
        gradients[backend] = torch.autograd.grad(loss, tuple(parameters.values()))
        if backend == "streamed":
            assert calculator._fullsf_tensor is None
        with torch.no_grad():
            assert calculator.calc_fprotein(Return=False) is None
    assert_close(torch.from_numpy(outputs["streamed"]), outputs["dense"])
    for index, name in enumerate(PARAMETERS):
        assert_close(
            gradients["streamed"][index],
            gradients["dense"][index].detach().cpu().numpy(),
            parameter=name,
        )


@pytest.mark.parametrize(
    "configuration",
    [
        {"fprotein_backend": "unknown"},
        {"fprotein_backend": ""},
        {"reflection_chunk_size": 0},
        {"reflection_chunk_size": -1},
        {"reflection_chunk_size": 1.5},
        {"reflection_chunk_size": "31"},
        {"reflection_chunk_size": True},
        {"reflection_chunk_size": False},
        {"reflection_chunk_size": np.bool_(True)},
        {"reflection_chunk_size": np.bool_(False)},
    ],
)
def test_invalid_constructor_configuration(configuration):
    option = next(iter(configuration))
    assert option in inspect.signature(SFcalculator.__init__).parameters
    with pytest.raises((ValueError, TypeError), match=option):
        configured_calculator(load_case("cryo_p1_iso"), **configuration)


@pytest.mark.parametrize("device", DEVICES)
def test_small_coordinate_hessian_vector_product(device):
    data = load_case("cryo_p1_iso")
    results = {}
    for backend in ("dense", "streamed"):
        calculator = configured_calculator(
            data, device, fprotein_backend=backend, reflection_chunk_size=31
        )
        parameters = parameter_tensors(data, device, requires_grad=False)
        positions = parameters["positions"].requires_grad_(True)
        output = calculate(calculator, parameters)
        loss = projected_loss(output, torch.tensor(data["probes"][0], device=device))
        (gradient,) = torch.autograd.grad(loss, positions, create_graph=True)
        direction = torch.linspace(-0.5, 0.5, positions.numel(), device=device).reshape_as(
            positions
        )
        (hessian_vector,) = torch.autograd.grad((gradient * direction).sum(), positions)
        assert torch.isfinite(hessian_vector).all()
        assert hessian_vector.abs().max().item() > 1e-5
        results[backend] = hessian_vector.detach().cpu().numpy()
    assert_close(torch.from_numpy(results["streamed"]), results["dense"], parameter="positions")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk_size", [31, 128])
def test_checkpoint_saved_storage_is_compact_and_deduplicated(device, chunk_size):
    data = load_case("cryo_1lj5")
    calculator = configured_calculator(data, device, reflection_chunk_size=chunk_size)
    parameters = parameter_tensors(data, device)
    n_atoms = calculator.n_atoms
    n_reflections = len(calculator.Hasu_array)
    storages = {}
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        storage = tensor.untyped_storage()
        # Checkpoints can save the same input for every chunk. Count underlying
        # allocations, not tensor views or repeated hook invocations.
        key = (tensor.device, storage.data_ptr())
        storages[key] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = calculate(calculator, parameters)
        loss = projected_loss(output, torch.tensor(data["probes"][0], device=device))
    assert saved_shapes, "The differentiable forward must save checkpoint inputs"
    assert not any(n_atoms in shape and n_reflections in shape for shape in saved_shapes), (
        saved_shapes
    )
    # A generous O(atoms + types*reflections + reflections + symmetry) budget,
    # independent of chunk count; retaining each chunk's intermediates exceeds it.
    budget = 16 * (
        13 * n_atoms
        + calculator._scattering_by_type.numel()
        + 8 * n_reflections
        + 12 * len(calculator.R_G_tensor_stack)
    )
    saved_bytes = sum(storages.values())
    assert saved_bytes <= budget, f"saved {saved_bytes:,} unique-storage bytes; budget {budget:,}"
    assert calculator._fullsf_tensor is None
    gradients = torch.autograd.grad(loss, tuple(parameters.values()))
    for index, name in enumerate(PARAMETERS):
        assert_close(gradients[index], data[f"expected_grad_{name}"][0], parameter=name)
