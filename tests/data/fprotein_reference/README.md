# Frozen structure-factor integration baseline

This fixture records the **unmodified dense implementation** at commit
`0b583b837c5ed94994e239669260c3d8c61fa1b7` (SFcalculator-torch 0.3.3).
It is the numerical contract for future replacements of `F_protein` and
`SFcalculator.calc_fprotein`, including changes to scattering-factor storage.
No production calculation has been changed.

## What is captured

Each compressed NPZ contains self-contained input arrays (orthogonal coordinates,
element identities, Biso, anisotropic U matrices, occupancies, cell, space group,
mode, wavelength, and requested HKLs), plus:

- Complex64 structure factors in both ASU and requested-HKL order.
- The actual ASU and requested HKL arrays, to detect ordering/mapping changes.
- Two independent, seeded complex probes and their real scalar projections.
- Float32 gradients of each projection with respect to **all coordinates,
  isotropic B factors, all anisotropic U entries, and occupancies**.

The projection is `real(sum(conj(F) * probe)) / sqrt(number_of_reflections)`.
Unlike an amplitude correlation, this is sensitive to phase/sign/conjugation
errors. Two vector-Jacobian products are useful regression checks, not a proof of
identity of every Jacobian entry for all possible inputs.

| Case | Atoms | Requested reflections | ASU reflections | Purpose |
| --- | ---: | ---: | ---: | --- |
| `cryo_1lj5` | 3,012 | 257 | 257 | Public 1LJ5 tutorial-target model, electron scattering in P1 |
| `cryo_p1_iso` | 11 | 257 | 257 | All-isotropic electron scattering, zero/partial/full occupancy, Biso including zero |
| `cryo_p21_mixed` | 11 | 257 | 257 | Screw-axis translations, nonorthogonal cell, mixed isotropic/anisotropic atoms |
| `xray_p212121_mixed` | 11 | 257 | 1,128 | Four symmetry operations, mixed B factors, ASU-to-requested-subset mapping |
| `xray_p1_anomalous` | 11 | 257 | 7,660 | Complex atomic scattering, heavy elements, Friedel mates |

Reflections span the available resolution range down to 3 Å and are deliberately
shuffled; 257 is an irregular size for future chunk/tile tail handling.

The bundled `1lj5.pdb` is the aligned predicted model used for the public
[1LJ5](https://www.rcsb.org/structure/1LJ5) tutorial target, **not the original
experimental PDB deposition**. It retains all 3,012 atoms and the model cell
(156.57, 46.88, 63.05 Å; beta 106.54 degrees). Only `CRYST1`, `ATOM`, `HETATM`,
`ANISOU`, `TER`, and `END` records are allowed: headers, remarks, paths, and other
run metadata are not bundled. The manifest hashes this sanitized input, and the
tests enforce both its hash and the record allowlist.

The electron-scattering case intentionally uses **P1 and sampled HKLs**, rather
than the target's experimental C2 symmetry or measured amplitudes. This is a
numerical kernel fixture, not a reconstruction of an experimental cryo-EM map.

All remaining cases are synthetic. Generation and tests are self-contained: no
external model files, maps, pipeline outputs, GPU, network access, or tracking
services are required.

## Run

From the SFcalculator repository root, using an environment with its dependencies,
PyTorch, and pytest:

```sh
PYTHONPATH=. python -m pytest -o addopts= -q tests/test_fprotein_reference.py
```

`-o addopts=` disables the repository's default coverage plugin requirement; it
does not change test assertions. If an unrelated third-party pytest plugin prevents
startup, prefix the command with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

Tests exercise the **source checkout**, not the installed wheel (an assertion
catches accidental wheel imports). CPU cases always run; corresponding CUDA cases
run automatically on CUDA hosts and otherwise explicitly skip. They cover:

- Public constructor and scattering initialization, not just a standalone kernel.
- `Return=True`, `Return=False`, ASU/HKL outputs, and zero protein-only mask.
- Detached/no-grad RSCC-like calculations.
- Both full-parameter backward probes.
- Constructor defaults, partial overrides, and persistence of setter state.
- Successive forward/backward calls on one calculator with changed occupancy,
  using both all-parameter and coordinates-only gradients. Occupancy linearity
  supplies an exact expected change without regenerating the fixture.
- An independent central-finite-difference check of a nonzero coordinate gradient
  in every case (0.01 Å step, 1% relative/0.01 absolute numerical tolerance).
- Fixture checksums, finite values, shape/dtype, and nondegenerate signals.

Elementwise complex-output tolerance is `rtol=3e-4, atol=3e-3`. Gradient tolerance
is `rtol=5e-4`, with absolute budgets of `3e-4` for coordinates, anisotropic U, and
occupancy, and `3e-6` for the much smaller Biso gradients. Scalar projections use
the output tolerance. Absolute tolerances account for cancellation/near-zero
values; these are float32 numerical equivalence checks, not bitwise comparisons.
**CUDA behavior/tolerances still require validation on a CUDA host**; do not loosen
assertions merely to make a new implementation pass.

## Provenance and deliberate regeneration

`manifest.json` records the baseline commit, hashes of the legacy calculation,
parser and symmetry sources, the bundled public tutorial-model hash, RNG
seed, package versions, dimensions, and hashes of the generated NPZ files. The
frozen files are read with `allow_pickle=False`; tests never generate expectations
from the implementation under test.

The generator refuses altered legacy source files and refuses to overwrite a
completed baseline. **Do not regenerate these fixtures after optimization** to
make tests pass. Any baseline change requires explicit scientific review.

For initial capture from the pinned source only, from this repository root:

```sh
PYTHONPATH=. python -m tests.generate_fprotein_reference --write-reference
```

## Scope and next optimization gate

This establishes regression equivalence to the existing implementation; it is not
an independent certification of that implementation's scientific correctness.
It does not cover solvent masking, scale optimization, batched-model APIs,
higher-order derivatives, the no-MTZ constructor path, or a full OpenFold/LLG run.
It also does not establish memory or runtime improvement: the bounded reflection
sample intentionally avoids reproducing the huge OOM allocation.

Future implementations can replace the kernel or calculator internals and run
this suite unchanged. Each selectable backend/chunk-size configuration should be
run against the same frozen data. Before declaring an optimization complete,
benchmark forward **and backward** peak CUDA memory/runtime at production-sized
reflection counts on a CUDA host, and validate a full refinement separately.
