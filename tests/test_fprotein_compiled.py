"""Optional compiled backend contracts; CUDA/Inductor runs are explicitly opt-in.

Run the expensive first-order CUDA checks with SFC_TEST_COMPILED_CUDA=1.
CPU checks use the same tensor tile with compile_kernel=False, not Inductor.
No test relies on the representation of the reciprocal-input or kernel cache.
"""

import inspect
import os
from functools import partial
from unittest.mock import patch

import gemmi
import numpy as np
import pytest
import torch
from SFC_Torch.Fmodel import SFcalculator
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from tests.fprotein_reference import (
    CASES,
    PARAMETERS,
    calculate,
    load_case,
    make_calculator,
    parameter_tensors,
    projected_loss,
)
from tests.test_fprotein_reference import assert_close

compiled_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("SFC_TEST_COMPILED_CUDA") != "1",
    reason="requires CUDA and opt-in SFC_TEST_COMPILED_CUDA=1 (expensive Inductor JIT)",
)


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


@pytest.fixture(scope="module")
def compiled_core():
    module = pytest.importorskip(
        "SFC_Torch.compiled", reason="optional compiled core is not implemented"
    )
    return module.f_protein_compiled


def configured_calculator(data, device="cpu", **configuration):
    with patch(
        "tests.fprotein_reference.SFcalculator",
        partial(SFcalculator, **configuration),
    ):
        return make_calculator(data, device)


def core_callable(core, calculator, *, chunk_size=128, compile_kernel=False):
    """Use exactly the 257 requested HKLs, including both X-ray chunk tails."""
    index = torch.as_tensor(calculator.asu2HKL_index, device=calculator.device, dtype=torch.long)
    np.testing.assert_array_equal(
        calculator.Hasu_array[calculator.asu2HKL_index], calculator.HKL_array
    )
    return partial(
        core,
        torch.tensor(calculator.HKL_array, device=calculator.device, dtype=torch.float32),
        torch.tensor(calculator.dr2HKL_array, device=calculator.device, dtype=torch.float32),
        calculator._scattering_by_type.index_select(1, index),
        calculator._atom_type_indices,
        calculator.R_G_tensor_stack,
        calculator.T_G_tensor_stack,
        calculator.orth2frac_tensor,
        chunk_size=chunk_size,
        compile_kernel=compile_kernel,
    )


def evaluate_core(evaluate, calculator, parameters):
    return evaluate(
        calculator.orth2frac(parameters["positions"]),
        parameters["b_iso"],
        parameters["aniso_u"],
        parameters["occupancy"],
    )


def check_frozen_output_and_vjps(output, parameters, data):
    assert output.is_complex()
    assert_close(output, data["expected_f_hkl"])
    for probe_index in (0, 1):
        probe = torch.tensor(data["probes"][probe_index], device=output.device)
        loss = projected_loss(output, probe)
        assert_close(loss, data["expected_losses"][probe_index])
        gradients = torch.autograd.grad(
            loss, tuple(parameters.values()), retain_graph=probe_index == 0
        )
        for index, name in enumerate(PARAMETERS):
            assert_close(
                gradients[index], data[f"expected_grad_{name}"][probe_index], parameter=name
            )


def check_against_streamed(output, parameters, calculator, data, names=PARAMETERS):
    """Fresh eager graphs are an oracle for inputs not in the frozen archive."""
    reference_parameters = {
        name: value.detach().clone().requires_grad_(value.requires_grad)
        for name, value in parameters.items()
    }
    reference = calculate(calculator, reference_parameters)
    assert_close(output, reference.detach().cpu().numpy())
    for probe_index in (0, 1):
        probe = torch.tensor(data["probes"][probe_index], device=output.device)
        actual_gradients = torch.autograd.grad(
            projected_loss(output, probe),
            tuple(parameters[name] for name in names),
            retain_graph=probe_index == 0,
        )
        expected_gradients = torch.autograd.grad(
            projected_loss(reference, probe),
            tuple(reference_parameters[name] for name in names),
            retain_graph=probe_index == 0,
        )
        for index, name in enumerate(names):
            assert_close(
                actual_gradients[index],
                expected_gradients[index].detach().cpu().numpy(),
                parameter=name,
            )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("chunk_size", (31, 128, 257))
def test_eager_tensor_core_frozen_outputs_and_both_vjps(compiled_core, case, chunk_size):
    data = load_case(case)
    calculator = make_calculator(data)
    parameters = parameter_tensors(data)
    evaluate = core_callable(compiled_core, calculator, chunk_size=chunk_size)
    output = evaluate_core(evaluate, calculator, parameters)
    check_frozen_output_and_vjps(output, parameters, data)


@compiled_cuda
@pytest.mark.parametrize("case", CASES)
def test_actual_inductor_frozen_outputs_and_both_vjps(compiled_core, case):
    # One tile size per case limits JIT specializations. No higher-order AD is
    # required: AOTAutograd/Inductor only promises first-order derivatives here.
    data = load_case(case)
    calculator = make_calculator(data, "cuda:0")
    evaluate = core_callable(compiled_core, calculator, compile_kernel=True)
    parameters = parameter_tensors(data, "cuda:0")
    check_frozen_output_and_vjps(evaluate_core(evaluate, calculator, parameters), parameters, data)


@pytest.mark.parametrize(
    "device,compile_kernel",
    [("cpu", False), pytest.param("cuda:0", True, marks=compiled_cuda)],
)
def test_reused_callable_observes_new_coordinate_b_and_u_inputs(
    compiled_core, device, compile_kernel
):
    data = load_case("cryo_p21_mixed")
    calculator = make_calculator(data, device)
    evaluate = core_callable(compiled_core, calculator, compile_kernel=compile_kernel)
    # Consume each graph completely before calling the identical callable again.
    for changed in (False, False, True):
        parameters = parameter_tensors(data, device)
        if changed:
            with torch.no_grad():
                parameters["positions"][:, 0].add_(0.17)
                parameters["b_iso"].mul_(1.15)
                # Activate U on formerly isotropic atoms as well as changing U
                # on anisotropic atoms: a cached mask must not survive the call.
                parameters["aniso_u"][:, 0, 0].add_(0.008)
        output = evaluate_core(evaluate, calculator, parameters)
        if changed:
            assert not torch.allclose(
                output.detach().cpu(), torch.from_numpy(data["expected_f_hkl"])
            )
        check_against_streamed(output, parameters, calculator, data)


@pytest.mark.parametrize("u_requires_grad", (False, True))
def test_isotropic_specialization_preserves_u_autograd_connection(compiled_core, u_requires_grad):
    data = load_case("cryo_p1_iso")
    calculator = make_calculator(data)
    evaluate = core_callable(compiled_core, calculator)
    parameters = parameter_tensors(data)
    parameters["aniso_u"].requires_grad_(u_requires_grad)
    assert torch.count_nonzero(parameters["aniso_u"]).item() == 0
    names = (
        PARAMETERS if u_requires_grad else tuple(name for name in PARAMETERS if name != "aniso_u")
    )
    # With trainable zero U the legacy mask gives zero U gradients, not an
    # unused input. autograd.grad deliberately does not use allow_unused=True.
    output = evaluate_core(evaluate, calculator, parameters)
    check_against_streamed(output, parameters, calculator, data, names)
    if u_requires_grad:
        parameters = parameter_tensors(data)
        with torch.no_grad():
            parameters["aniso_u"][:, 1, 1].fill_(0.012)
        check_against_streamed(
            evaluate_core(evaluate, calculator, parameters), parameters, calculator, data
        )


class RejectDenseAtomReflection(TorchDispatchMode):
    """Reject transient full-size tensors in forward and backward, not tiles."""

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
                    f"{func} allocated full atom-by-reflection tensor {tuple(tensor.shape)}"
                )
        return result


@pytest.mark.parametrize("case", ("cryo_1lj5", "xray_p212121_mixed", "xray_p1_anomalous"))
def test_eager_core_never_allocates_dense_all_reflections(compiled_core, case):
    data = load_case(case)
    calculator = make_calculator(data)
    parameters = parameter_tensors(data)
    evaluate = core_callable(compiled_core, calculator, chunk_size=31)
    with RejectDenseAtomReflection(len(data["positions"]), 257):
        output = evaluate_core(evaluate, calculator, parameters)
        check_frozen_output_and_vjps(output, parameters, data)
    assert calculator._fullsf_tensor is None


def test_default_backend_remains_streamed():
    assert inspect.signature(SFcalculator).parameters["fprotein_backend"].default == "streamed"
    calculator = make_calculator(load_case("cryo_p1_iso"))
    assert calculator.fprotein_backend == "streamed"


@pytest.mark.parametrize("at_constructor", (False, True))
def test_explicit_compiled_cpu_is_rejected_not_silently_streamed(at_constructor):
    data = load_case("cryo_p1_iso")
    # Rejection may happen at construction or evaluation, but must explain the
    # CUDA requirement rather than treating 'compiled' as an unknown backend.
    with pytest.raises((ValueError, RuntimeError, NotImplementedError), match="(?i)cuda"):
        if at_constructor:
            calculator = configured_calculator(data, fprotein_backend="compiled")
        else:
            calculator = make_calculator(data)
            calculator.fprotein_backend = "compiled"
        calculate(calculator, parameter_tensors(data))


def test_compiling_core_on_cpu_requires_explicit_eager_opt_out(compiled_core):
    data = load_case("cryo_p1_iso")
    calculator = make_calculator(data)
    parameters = parameter_tensors(data)
    evaluate = core_callable(compiled_core, calculator, compile_kernel=True)
    with pytest.raises((ValueError, RuntimeError, NotImplementedError), match="(?i)cuda"):
        evaluate_core(evaluate, calculator, parameters)


@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda:0", marks=compiled_cuda)))
def test_previously_read_dense_cache_remains_authoritative_for_compiled(device):
    data = load_case("cryo_p21_mixed")
    calculator = make_calculator(data, device)
    dense = calculator.fullsf_tensor
    dense.mul_(0.5)
    # A materialized public dense cache retains the legacy compatibility path,
    # including on CPU; only the true compact compiled path requires CUDA.
    calculator.fprotein_backend = "compiled"
    calculator.reflection_chunk_size = 128
    parameters = parameter_tensors(data, device)
    output = calculate(calculator, parameters)
    scaled = dict(data)
    for key in (
        "expected_f_hkl",
        "expected_losses",
        *(f"expected_grad_{name}" for name in PARAMETERS),
    ):
        scaled[key] = data[key] * 0.5
    check_frozen_output_and_vjps(output, parameters, scaled)
    assert_close(calculator.Fprotein_asu, data["expected_f_asu"] * 0.5)
    assert calculator.fullsf_tensor is dense
    dense.zero_()
    with torch.no_grad():
        output = calculate(calculator, parameter_tensors(data, device, requires_grad=False))
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


@compiled_cuda
def test_public_compiled_integration_outputs_gradients_and_repeated_calls():
    data = load_case("cryo_p21_mixed")
    calculator = configured_calculator(
        data, "cuda:0", fprotein_backend="compiled", reflection_chunk_size=128
    )
    for _ in range(2):
        parameters = parameter_tensors(data, "cuda:0")
        output = calculate(calculator, parameters)
        check_frozen_output_and_vjps(output, parameters, data)
        assert_close(calculator.Fprotein_asu, data["expected_f_asu"])
        assert calculator._fullsf_tensor is None
    with torch.no_grad():
        assert (
            calculate(calculator, parameter_tensors(data, "cuda:0", requires_grad=False), False)
            is None
        )
    assert_close(calculator.Fprotein_HKL, data["expected_f_hkl"])


@compiled_cuda
def test_reciprocal_numpy_edits_invalidate_cache_via_public_outputs():
    data = load_case("cryo_p21_mixed")
    calculator = configured_calculator(
        data, "cuda:0", fprotein_backend="compiled", reflection_chunk_size=128
    )
    oracle = configured_calculator(data, "cuda:0", fprotein_backend="dense")
    parameters = parameter_tensors(data, "cuda:0", requires_grad=False)
    with torch.no_grad():
        before = calculate(calculator, parameters).clone()
        assert_close(before, data["expected_f_hkl"])
        # Mutate in place (same ndarray identity, shape and dtype). Updating
        # both calculators identically isolates stale tensor conversion caches
        # without requiring physically regenerated scattering tables or bins.
        for attribute in ("dr2asu_array", "Hasu_array"):
            for target in (calculator, oracle):
                array = getattr(target, attribute)
                if attribute == "dr2asu_array":
                    array *= 1.4
                else:
                    array[:, 0] += 1
            actual = calculate(calculator, parameters)
            expected = calculate(oracle, parameters)
            assert_close(actual, expected.detach().cpu().numpy())
            assert not torch.allclose(actual, before)
            before = actual.clone()
            assert calculator._fullsf_tensor is None


@pytest.mark.parametrize(
    "device,compile_kernel",
    [
        ("cpu", False),
        pytest.param("cuda:0", True, marks=compiled_cuda),
    ],
)
def test_two_outstanding_graphs_use_current_inputs(compiled_core, device, compile_kernel):
    data = load_case("cryo_p21_mixed")
    calculator = make_calculator(data, device)
    evaluate = core_callable(compiled_core, calculator, compile_kernel=compile_kernel)
    first = parameter_tensors(data, device)
    second = parameter_tensors(data, device)
    with torch.no_grad():
        second["positions"].add_(0.07)
        second["aniso_u"][:, 0, 0].add_(0.01)
    first_output = evaluate_core(evaluate, calculator, first)
    second_output = evaluate_core(evaluate, calculator, second)
    check_against_streamed(second_output, second, calculator, data)
    check_frozen_output_and_vjps(first_output, first, data)


@pytest.mark.parametrize("chunk_size", (31, 128))
def test_eager_core_checkpoint_saved_storage_is_compact(compiled_core, chunk_size):
    from benchmarks.fprotein import SavedTensors

    data = load_case("cryo_1lj5")
    calculator = make_calculator(data)
    parameters = parameter_tensors(data)
    evaluate = core_callable(compiled_core, calculator, chunk_size=chunk_size)
    counter = SavedTensors()
    with torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack):
        output = evaluate_core(evaluate, calculator, parameters)
    budget = 16 * (
        13 * len(data["positions"]) + 8 * len(data["hkl"]) + calculator._scattering_by_type.numel()
    )
    assert counter.report()["unique_storage_bytes"] <= budget
    check_frozen_output_and_vjps(output, parameters, data)


@compiled_cuda
@pytest.mark.parametrize("trainable", ("positions", "occupancy", "all"))
def test_first_inference_then_training_on_same_calculator(trainable):
    data = load_case("cryo_p1_iso")
    calculator = configured_calculator(
        data, "cuda:0", fprotein_backend="compiled", reflection_chunk_size=128
    )
    with torch.inference_mode():
        output = calculate(calculator, parameter_tensors(data, "cuda:0", requires_grad=False))
        assert_close(output, data["expected_f_hkl"])
    parameters = parameter_tensors(data, "cuda:0", requires_grad=trainable == "all")
    names = PARAMETERS if trainable == "all" else (trainable,)
    if trainable != "all":
        parameters[trainable].requires_grad_(True)
    output = calculate(calculator, parameters)
    oracle = make_calculator(data, "cuda:0")
    check_against_streamed(output, parameters, oracle, data, names)


@compiled_cuda
def test_more_than_eight_static_signatures_without_global_reset(compiled_core):
    data = load_case("cryo_p1_iso")
    calculator = make_calculator(data, "cuda:0")
    for chunk_size in range(120, 129):
        parameters = parameter_tensors(data, "cuda:0")
        evaluate = core_callable(
            compiled_core, calculator, chunk_size=chunk_size, compile_kernel=True
        )
        check_frozen_output_and_vjps(
            evaluate_core(evaluate, calculator, parameters), parameters, data
        )
    parameters = parameter_tensors(data, "cuda:0")
    evaluate = core_callable(compiled_core, calculator, chunk_size=120, compile_kernel=True)
    check_frozen_output_and_vjps(evaluate_core(evaluate, calculator, parameters), parameters, data)


@compiled_cuda
@pytest.mark.parametrize("mode", ("coordinates", "all"))
@pytest.mark.parametrize("chunk_size", (2048, 8192))
def test_full_reflection_public_model_matches_streamed(mode, chunk_size):
    data = load_case("cryo_1lj5")
    data["hkl"] = gemmi.make_miller_array(
        gemmi.UnitCell(*data["cell"]),
        gemmi.SpaceGroup(str(data["spacegroup"])),
        3.0,
        0.0,
        unique=True,
    )
    calculator = make_calculator(data, "cuda:0")
    calculator.reflection_chunk_size = chunk_size
    probe = torch.randn(
        len(data["hkl"]),
        dtype=torch.complex64,
        device="cuda:0",
        generator=torch.Generator(device="cuda:0").manual_seed(7),
    )
    results = {}
    for backend in ("streamed", "compiled"):
        parameters = parameter_tensors(data, "cuda:0", requires_grad=mode == "all")
        parameters["positions"].requires_grad_(True)
        calculator.fprotein_backend = backend
        output = calculate(calculator, parameters)
        names = PARAMETERS if mode == "all" else ("positions",)
        gradients = torch.autograd.grad(
            projected_loss(output, probe), tuple(parameters[name] for name in names)
        )
        results[backend] = (
            output.detach().cpu().numpy(),
            [gradient.detach().cpu().numpy() for gradient in gradients],
        )
    assert_close(torch.from_numpy(results["compiled"][0]), results["streamed"][0])
    for index, name in enumerate(names):
        assert_close(
            torch.from_numpy(results["compiled"][1][index]),
            results["streamed"][1][index],
            parameter=name,
        )
    assert calculator._fullsf_tensor is None


def test_frozen_zero_u_retains_trainable_orth2frac_connection(compiled_core):
    data = load_case("cryo_p1_iso")
    calculator = make_calculator(data)
    parameters = parameter_tensors(data, requires_grad=False)
    evaluate = core_callable(compiled_core, calculator)
    args = list(evaluate.args)
    matrix = args[6].detach().clone().requires_grad_(True)
    args[6] = matrix
    output = compiled_core(
        *args,
        calculator.orth2frac(parameters["positions"]),
        parameters["b_iso"],
        parameters["aniso_u"],
        parameters["occupancy"],
        **evaluate.keywords,
    )
    (gradient,) = torch.autograd.grad(output.real.sum(), matrix)
    torch.testing.assert_close(gradient, torch.zeros_like(matrix), rtol=0, atol=0)


@pytest.mark.parametrize("name", PARAMETERS)
def test_compiled_core_rejects_non_float32_parameters(compiled_core, name):
    data = load_case("cryo_p1_iso")
    calculator = make_calculator(data)
    parameters = parameter_tensors(data)
    parameters[name] = parameters[name].detach().double().requires_grad_(True)
    evaluate = core_callable(compiled_core, calculator)
    # Pass fractional positions directly so the core, not the public coordinate
    # conversion's matrix multiply, is responsible for dtype validation.
    positions = calculator.orth2frac(parameters["positions"].float()).to(
        parameters["positions"].dtype
    )
    with pytest.raises(ValueError, match="float32"):
        evaluate(positions, parameters["b_iso"], parameters["aniso_u"], parameters["occupancy"])
