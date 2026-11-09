"""Opt-in, checkpointed reflection tiles with a CUDA Inductor arithmetic kernel.

No calculator integration or default backend selection is performed here. Compiled
execution supports ordinary first-order autograd; AOTAutograd double backward is
not qualified. Use the existing streamed backend for higher-order derivatives.
"""

from functools import lru_cache
from math import pi
from numbers import Integral
from types import FunctionType

import torch
from torch.utils.checkpoint import checkpoint

__all__ = ["f_protein_compiled", "tensor_tile", "get_compiled_tile"]


def tensor_tile(
    hkl,
    dr2,
    scattering_real,
    scattering_imag,
    atom_type_indices,
    rotations,
    translations,
    orth2frac,
    positions_frac,
    b_iso,
    aniso_u,
    occupancy,
    isotropic_only=False,
):
    """Pure real-tensor tile arithmetic, returning reduced ``(real, imag)``.

    ``hkl``/``dr2`` and compact scattering are already sliced to this tile.
    ``scattering_imag`` is None for non-anomalous scattering. Other shapes follow
    the legacy F_protein kernel. No complex operations or reflection-chunk loop
    enter the compiled graph; the small symmetry loop is statically unrolled.
    Set ``isotropic_only`` only for all-zero U with both U and orth2frac frozen;
    otherwise even all-zero U must take the general path for connected zero VJPs.
    """
    sf_real = scattering_real.index_select(0, atom_type_indices) * occupancy[:, None]
    sf_imag = None
    if scattering_imag is not None:
        sf_imag = scattering_imag.index_select(0, atom_type_indices) * occupancy[:, None]
    dwf_iso = torch.exp(-b_iso[:, None] * dr2 / 4.0)
    sym_positions = torch.einsum("oxy,ay->aox", rotations, positions_frac) + translations
    if not isotropic_only:
        mask = torch.all(torch.all(aniso_u == 0.0, dim=-1), dim=-1)
        # Preserve the legacy Ustar contraction and operand order.
        ustar = torch.einsum("xy,ayz,wz->axw", orth2frac, aniso_u, orth2frac)
        sym_hkl = torch.einsum("rx,oxy->roy", hkl, rotations)
    phase_real = torch.zeros_like(sf_real)
    phase_imag = torch.zeros_like(sf_real)
    for i in range(rotations.shape[0]):
        xyz = sym_positions[:, i, :]
        phase = (2 * pi) * (
            xyz[:, 0, None] * hkl[None, :, 0]
            + xyz[:, 1, None] * hkl[None, :, 1]
            + xyz[:, 2, None] * hkl[None, :, 2]
        )
        damping = dwf_iso
        if not isotropic_only:
            h, k, ell = (sym_hkl[None, :, i, j] for j in range(3))
            quadratic = (
                (h * ustar[:, 0, 0, None] + k * ustar[:, 1, 0, None] + ell * ustar[:, 2, 0, None])
                * h
                + (h * ustar[:, 0, 1, None] + k * ustar[:, 1, 1, None] + ell * ustar[:, 2, 1, None])
                * k
                + (h * ustar[:, 0, 2, None] + k * ustar[:, 1, 2, None] + ell * ustar[:, 2, 2, None])
                * ell
            )
            dwf_aniso = torch.exp((-2.0 * pi**2) * quadratic)
            damping = torch.where(mask[:, None], dwf_iso, dwf_aniso)
        phase_real = phase_real + damping * torch.cos(phase)
        phase_imag = phase_imag + damping * torch.sin(phase)
    if sf_imag is None:
        return (phase_real * sf_real).sum(dim=0), (phase_imag * sf_real).sum(dim=0)
    return (
        (phase_real * sf_real - phase_imag * sf_imag).sum(dim=0),
        (phase_imag * sf_real + phase_real * sf_imag).sum(dim=0),
    )


def _tile_signature(args):
    return (
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        tuple(
            (arg.shape, arg.stride(), arg.dtype, arg.device, arg.requires_grad)
            if isinstance(arg, torch.Tensor)
            else arg
            for arg in args
        ),
    )


@lru_cache(maxsize=32)
def get_compiled_tile(signature):
    """Cache up to 32 static tile variants without changing global compiler limits.

    Dynamo guards are keyed by Python code object. Separate code objects keep
    independent model shapes, short tails and grad modes from exhausting one
    function's recompile budget. Tensor values are never captured or cached.
    Cache counts describe wrapper reuse, not actual compilations; eviction does
    not clear PyTorch's on-disk compiler cache. Use TORCH_LOGS=recompiles to inspect
    unexpected specializations (e.g. autocast state).
    """
    tile = FunctionType(
        tensor_tile.__code__.replace(),
        tensor_tile.__globals__,
        tensor_tile.__name__,
        tensor_tile.__defaults__,
    )
    return torch.compile(tile, fullgraph=True, dynamic=False, backend="inductor")


def f_protein_compiled(
    hkl_tensor,
    dr2_tensor,
    scattering_by_type,
    atom_type_indices,
    rotations,
    translations,
    orth2frac,
    positions_frac,
    b_iso,
    aniso_u,
    occupancy,
    chunk_size=2048,
    *,
    compile_kernel=True,
):
    """Compute complex64 structure factors in bounded, checkpointed tiles.

    Inputs follow ``f_protein_streamed``, except ``hkl_tensor`` and ``dr2_tensor``
    are already device-local float32 tensors of shapes (R, 3) and (R,).
    All tensors must share a device; other floating inputs are float32 and the
    compact (types, R) scattering table is float32 or complex64. Type indices
    are int64. Inputs must not be mutated before backward completes.

    Compilation is opt-in CUDA-only. ``compile_kernel=False`` runs the same pure
    tensor arithmetic eagerly, including on CPU, for correctness testing. No
    silent eager fallback is used if compilation fails. Each tile's scattering
    gather and arithmetic are non-reentrantly checkpointed when gradients are
    needed, bounding dense workspace by atoms * chunk_size. Compact input storage,
    output storage and per-tile autograd bookkeeping still span all reflections.
    Complex assembly is outside the compiled callable. First-order gradients are
    supported; use the streamed backend for unqualified AOTAutograd double backward.

    All-zero U with frozen U and orth2frac skips anisotropic work (one host-
    synchronizing zero check per call on CUDA). Trainable zero U or orth2frac
    retains the general masked computation so its gradient is a connected zero
    tensor, not an unused input.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
        raise TypeError("chunk_size must be a positive, non-boolean integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunk_size = int(chunk_size)
    if not isinstance(compile_kernel, bool):
        raise TypeError("compile_kernel must be a bool")
    if compile_kernel and hkl_tensor.device.type != "cuda":
        raise ValueError("compile_kernel=True requires CUDA; use compile_kernel=False on CPU")
    if hkl_tensor.dtype != torch.float32 or dr2_tensor.dtype != torch.float32:
        raise ValueError("hkl_tensor and dr2_tensor must be float32")
    n_reflections = hkl_tensor.shape[0]
    if hkl_tensor.shape != (n_reflections, 3) or dr2_tensor.shape != (n_reflections,):
        raise ValueError("hkl_tensor and dr2_tensor must have shapes (R, 3) and (R,)")
    if scattering_by_type.ndim != 2 or scattering_by_type.shape[1] != n_reflections:
        raise ValueError("scattering_by_type must have shape (n_types, R)")
    inputs = (
        hkl_tensor,
        dr2_tensor,
        scattering_by_type,
        atom_type_indices,
        rotations,
        translations,
        orth2frac,
        positions_frac,
        b_iso,
        aniso_u,
        occupancy,
    )
    if any(t.device != hkl_tensor.device for t in inputs):
        raise ValueError("all inputs must be on the same device")
    if any(
        t.dtype != torch.float32
        for t in (
            rotations,
            translations,
            orth2frac,
            positions_frac,
            b_iso,
            aniso_u,
            occupancy,
        )
    ):
        raise ValueError("geometry and atomic parameter tensors must be float32")
    if scattering_by_type.dtype not in (torch.float32, torch.complex64):
        raise ValueError("scattering_by_type must be float32 or complex64")
    if atom_type_indices.dtype != torch.int64:
        raise ValueError("atom_type_indices must be int64")
    needs_checkpoint = torch.is_grad_enabled() and any(t.requires_grad for t in inputs)
    isotropic_only = (
        not aniso_u.requires_grad
        and not orth2frac.requires_grad
        and bool(torch.all(aniso_u == 0.0))
    )

    if scattering_by_type.is_complex():
        scattering_real, scattering_imag = scattering_by_type.real, scattering_by_type.imag
    else:
        scattering_real, scattering_imag = scattering_by_type, None
    tiles = []
    # An empty tile keeps dtype/device and the general path's gradient connections.
    for start in range(0, max(n_reflections, 1), chunk_size):
        stop = min(start + chunk_size, n_reflections)
        args = (
            hkl_tensor[start:stop],
            dr2_tensor[start:stop],
            scattering_real[:, start:stop],
            None if scattering_imag is None else scattering_imag[:, start:stop],
            atom_type_indices,
            rotations,
            translations,
            orth2frac,
            positions_frac,
            b_iso,
            aniso_u,
            occupancy,
            isotropic_only,
        )
        evaluate = get_compiled_tile(_tile_signature(args)) if compile_kernel else tensor_tile
        if needs_checkpoint:
            real, imag = checkpoint(evaluate, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            real, imag = evaluate(*args)
        tiles.append(torch.complex(real, imag))
    return torch.cat(tiles, dim=0)
