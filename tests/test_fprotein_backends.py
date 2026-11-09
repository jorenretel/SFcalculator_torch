"""Small runtime-oracle checks; CUDA compilation is explicitly opt-in."""

import os

import numpy as np
import pytest
import torch
from SFC_Torch.compiled import f_protein_compiled
from SFC_Torch.Fmodel import F_protein, SFcalculator
from SFC_Torch.streaming import f_protein_streamed

CUDA = torch.cuda.is_available()
COMPILED_CUDA = CUDA and os.environ.get("SFC_TEST_COMPILED_CUDA") == "1"
COMPILED = pytest.param(
    "compiled",
    "cuda",
    marks=pytest.mark.skipif(
        not COMPILED_CUDA, reason="requires CUDA and SFC_TEST_COMPILED_CUDA=1"
    ),
    id="compiled-cuda",
)
BACKENDS = [
    pytest.param("streamed", "cpu", id="streamed-cpu"),
    pytest.param("compiled-eager", "cpu", id="compiled-eager-cpu"),
    pytest.param(
        "streamed",
        "cuda",
        marks=pytest.mark.skipif(not CUDA, reason="requires CUDA"),
        id="streamed-cuda",
    ),
    COMPILED,
]
CALCULATORS = [pytest.param("streamed", "cpu", id="streamed-cpu"), COMPILED]


def _case(n_atoms, n_hkl, anomalous, device):
    rng = np.random.default_rng(2026 + n_atoms)

    def tensor(value):
        return torch.tensor(np.asarray(value), dtype=torch.float32, device=device)

    hkl = rng.integers(-4, 5, size=(n_hkl, 3), dtype=np.int32)
    hkl[np.all(hkl == 0, axis=1), 0] = 1
    orth2frac = tensor([[0.08, -0.015, 0.007], [0, 0.07, -0.011], [0, 0, 0.06]])
    reciprocal = hkl.astype(np.float32) @ orth2frac.cpu().numpy()
    dr2 = np.sum(reciprocal**2, axis=1)
    scattering = tensor(rng.uniform(1, 8, size=(3, n_hkl)))
    if anomalous:
        scattering = torch.complex(scattering, tensor(rng.uniform(-0.4, 0.4, (3, n_hkl))))
    indices = torch.arange(n_atoms, device=device) % 3
    rotations = tensor([np.eye(3), np.diag([-1, 1, -1])])
    translations = tensor([[0, 0, 0], [0, 0.5, 0]])
    u = rng.normal(0, 0.025, size=(n_atoms, 3, 3)).astype(np.float32)
    u = u @ u.transpose(0, 2, 1) + np.eye(3, dtype=np.float32)[None] * 0.02
    u[::2] = 0
    parameters = (
        tensor(rng.uniform(0.1, 0.9, (n_atoms, 3))),
        tensor(rng.uniform(3, 20, n_atoms)),
        tensor(u),
        tensor(rng.uniform(0.3, 1, n_atoms)),
    )
    return (hkl, dr2, scattering, indices, rotations, translations, orth2frac), parameters


def _leaves(parameters):
    # U stays a full nine-entry leaf, rather than a six-entry symmetric encoding.
    return tuple(p.detach().clone().requires_grad_() for p in parameters)


def _evaluate(backend, case, parameters, chunk_size):
    hkl, dr2, scattering, indices, *geometry = case
    if backend == "dense":
        return F_protein(hkl, dr2, scattering[indices], *geometry, *parameters)
    if backend == "streamed":
        return f_protein_streamed(*case, *parameters, chunk_size=chunk_size)
    return f_protein_compiled(
        torch.tensor(hkl, dtype=torch.float32, device=scattering.device),
        torch.tensor(dr2, dtype=torch.float32, device=scattering.device),
        scattering,
        indices,
        *geometry,
        *parameters,
        chunk_size=chunk_size,
        compile_kernel=backend == "compiled",
    )


def _assert_output_and_vjps(actual, expected, parameters, reference_parameters):
    assert actual.dtype == expected.dtype == torch.complex64
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-3)
    for seed in (19, 83):
        generator = torch.Generator().manual_seed(seed)
        weights = torch.randn(expected.shape + (2,), generator=generator)
        # Keep projection scale independent of the number of reflections.
        cotangent = torch.view_as_complex(weights).to(expected.device) / expected.numel() ** 0.5
        actual_grads = torch.autograd.grad(actual, parameters, cotangent, retain_graph=True)
        expected_grads = torch.autograd.grad(
            expected, reference_parameters, cotangent, retain_graph=True
        )
        for i, grad in enumerate(actual_grads):
            reference = expected_grads[i]
            assert grad.shape == parameters[i].shape
            assert torch.isfinite(grad).all()
            torch.testing.assert_close(grad, reference, rtol=5e-4, atol=3e-6 if i == 1 else 3e-4)


@pytest.mark.parametrize("backend,device", BACKENDS)
@pytest.mark.parametrize("n_atoms,n_hkl", [(7, 17), (9, 31)])
@pytest.mark.parametrize("anomalous", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("chunk_size", [6, 64], ids=["irregular-tail", "larger-than-R"])
def test_output_and_two_vjps(backend, device, n_atoms, n_hkl, anomalous, chunk_size):
    case, initial = _case(n_atoms, n_hkl, anomalous, device)
    parameters, reference = _leaves(initial), _leaves(initial)
    actual = _evaluate(backend, case, parameters, chunk_size)
    expected = _evaluate("dense", case, reference, chunk_size)
    _assert_output_and_vjps(actual, expected, parameters, reference)


def _calculator(data_pdb, device, **kwargs):
    return SFcalculator(
        data_pdb,
        mtzdata=kwargs.pop("mtzdata", None),
        dmin=2.5,
        n_bins=4,
        set_experiment=False,
        device=torch.device(device),
        reflection_chunk_size=257,
        **kwargs,
    )


def test_calculator_default_is_eager_dense(data_pdb):
    calculator = _calculator(data_pdb, "cpu")
    assert calculator.fprotein_backend == "dense"
    assert calculator._fullsf_tensor is not None  # Before any calculation/access.
    assert calculator._fullsf_tensor.shape == (calculator.n_atoms, len(calculator.Hasu_array))


@pytest.mark.parametrize("backend,device", CALCULATORS)
@pytest.mark.parametrize("cache_action", ["access", "assignment"])
def test_calculator_compact_and_public_dense_cache(
    data_pdb, data_mtz_exp, backend, device, cache_action
):
    dense = _calculator(data_pdb, device, fprotein_backend="dense", mtzdata=data_mtz_exp)
    compact = _calculator(data_pdb, device, fprotein_backend=backend, mtzdata=data_mtz_exp)
    assert compact._fullsf_tensor is None
    assert compact._scattering_by_type.shape[0] < compact.n_atoms
    with torch.no_grad():
        baseline = dense.calc_fprotein(Return=True)
        assert len(baseline) == len(compact.HKL_array)
        torch.testing.assert_close(
            compact.calc_fprotein(Return=True), baseline, rtol=3e-4, atol=3e-3
        )
        assert compact._fullsf_tensor is None
        if cache_action == "access":
            table = compact.fullsf_tensor
        else:
            table = dense.fullsf_tensor.clone()
            compact.fullsf_tensor = table
        assert compact._fullsf_tensor is table
        torch.testing.assert_close(
            compact.calc_fprotein(Return=True), baseline, rtol=3e-4, atol=3e-3
        )
        # Public dense edits must take precedence over the compact table thereafter.
        table.mul_(0.75)
        dense.fullsf_tensor.mul_(0.75)
        mutated = compact.calc_fprotein(Return=True)
        torch.testing.assert_close(mutated, dense.calc_fprotein(Return=True), rtol=3e-4, atol=3e-3)
        torch.testing.assert_close(mutated, baseline * 0.75, rtol=3e-4, atol=3e-3)
        assert compact.fullsf_tensor is table


@pytest.mark.parametrize("backend,device", CALCULATORS)
def test_calculator_inference_then_training(data_pdb, backend, device):
    calculator = _calculator(data_pdb, device, fprotein_backend=backend)
    dense = _calculator(data_pdb, device, fprotein_backend="dense")
    with torch.inference_mode():
        inference = calculator.calc_fprotein(Return=True)
        torch.testing.assert_close(
            inference, dense.calc_fprotein(Return=True), rtol=3e-4, atol=3e-3
        )
    assert calculator._fullsf_tensor is None
    names = ("atom_pos_orth", "atom_b_iso", "atom_aniso_uw", "atom_occ")
    initial = tuple(getattr(calculator, name) for name in names)
    parameters, reference = _leaves(initial), _leaves(initial)
    for i, name in enumerate(names):
        setattr(calculator, name, parameters[i])
        setattr(dense, name, reference[i])
    _assert_output_and_vjps(
        calculator.calc_fprotein(Return=True),
        dense.calc_fprotein(Return=True),
        parameters,
        reference,
    )
    assert calculator._fullsf_tensor is None


@pytest.mark.parametrize("backend", ["streamed", "compiled-eager"])
def test_checkpoint_saved_storage_budget(backend):
    case, initial = _case(9, 31, True, "cpu")
    parameters = _leaves(initial)

    def storage(tensor):
        allocation = tensor.untyped_storage()
        return (tensor.device, allocation.data_ptr()), allocation.nbytes()

    inputs = dict(storage(t) for t in (*case[2:], *parameters))
    # Allow device-local HKL/dr2 and reduced real/imag outputs, not dense tiles.
    budget = sum(inputs.values()) + len(case[0]) * (4 * 4 + 2 * 4)

    def saved_bytes(which):
        saved = {}

        def pack(tensor):
            key, size = storage(tensor)
            saved[key] = size
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            result = _evaluate(which, case, parameters, 6)
        gradients = torch.autograd.grad(result.real.sum(), parameters)
        assert all(torch.isfinite(grad).all() for grad in gradients)
        return sum(saved.values())

    assert saved_bytes(backend) <= budget
    assert saved_bytes("dense") > budget
