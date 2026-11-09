"""Reflection-tiled structure factors with recomputation-based autograd.

Only compact type-by-reflection scattering and the final reflection vector span
all reflections. Dense atom-by-reflection scattering, phases, and Debye-Waller
factors are constructed inside each tile. With gradients enabled, non-reentrant
checkpointing discards saved tile intermediates and recomputes them in backward.
This trades additional computation for memory without handwritten derivatives.
"""

from numbers import Integral

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def f_protein_streamed(
    hkl: np.ndarray,
    dr2: np.ndarray,
    scattering_by_type: torch.Tensor,
    atom_type_indices: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    orth2frac: torch.Tensor,
    positions_frac: torch.Tensor,
    b_iso: torch.Tensor,
    aniso_u: torch.Tensor,
    occupancy: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Evaluate protein structure factors in bounded reflection tiles.

    Parameters follow :func:`SFC_Torch.Fmodel.F_protein`, except that scattering
    is compact: ``scattering_by_type`` has shape ``(n_types, n_reflections)``
    and ``atom_type_indices`` is a device-local int64 tensor of shape
    ``(n_atoms,)`` mapping atoms to rows. ``hkl`` and ``dr2`` are NumPy arrays
    of shape ``(n_reflections, 3)`` and ``(n_reflections,)``, respectively.
    Other tensors use the legacy shapes, a common device, and float32; the
    scattering table may instead be complex64 for anomalous scattering.
    The result is a complex64 vector in the original reflection order.

    Each tile uses the legacy kernel unchanged, including its isotropic versus
    anisotropic selection, phase sign, and symmetry summation. Only the reduced
    reflection vectors are concatenated, never atom-by-reflection tensors.
    The dense workspace scales with ``n_atoms * chunk_size`` (and symmetry
    count), rather than ``n_atoms * n_reflections``. Input storage, the final
    output, and per-tile autograd bookkeeping are not eliminated.

    If autograd is enabled and any floating/complex input requires gradients,
    each tile is non-reentrantly checkpointed, including its scattering gather.
    Backward therefore recomputes the tile instead of retaining its dense saved
    intermediates. This supports ``torch.autograd.grad`` and higher derivatives,
    although higher-order graphs can themselves retain substantial memory.
    No-grad evaluation computes tiles directly. Inputs, including the NumPy
    arrays, must not be mutated before backward completes.

    ``chunk_size`` must be a positive, non-boolean integer. Empty reflection
    input returns an empty vector, retaining autograd connectivity when needed.
    """
    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, Integral):
        raise TypeError("chunk_size must be a positive, non-boolean integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunk_size = int(chunk_size)

    inputs = (
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
    needs_checkpoint = torch.is_grad_enabled() and any(t.requires_grad for t in inputs)
    n_reflections = len(hkl)
    tiles = []
    # One empty tile preserves the legacy dtype/device and autograd connections.
    for start in range(0, max(n_reflections, 1), chunk_size):
        stop = min(start + chunk_size, n_reflections)

        def evaluate_tile(
            scattering,
            type_indices,
            tile_rotations,
            tile_translations,
            tile_orth2frac,
            tile_positions,
            tile_b_iso,
            tile_aniso_u,
            tile_occupancy,
            *,
            hkl_tile=hkl[start:stop],
            dr2_tile=dr2[start:stop],
            tile_start=start,
            tile_stop=stop,
        ):
            # Local import permits Fmodel to import this module during integration.
            from .Fmodel import F_protein

            atomic_scattering = scattering[:, tile_start:tile_stop].index_select(0, type_indices)
            return F_protein(
                hkl_tile,
                dr2_tile,
                atomic_scattering,
                tile_rotations,
                tile_translations,
                tile_orth2frac,
                tile_positions,
                tile_b_iso,
                tile_aniso_u,
                tile_occupancy,
            )

        if needs_checkpoint:
            tile = checkpoint(
                evaluate_tile,
                *inputs,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            tile = evaluate_tile(*inputs)
        tiles.append(tile)

    return torch.cat(tiles, dim=0)
