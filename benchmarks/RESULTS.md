# Reflection-streaming measurements

Measured on one allocated NVIDIA H100 80 GB GPU with PyTorch 2.7.1 (CUDA build
12.6), NumPy 1.26.4, Gemmi 0.6.7, float32/complex64, and float32 matrix precision
`highest`. No other application graph is included. All inputs derive from the
bundled public 1LJ5 tutorial-target model (3,012 atoms) and synthetic reflection
sets; no private scientific data is required.

## Coordinates-only gradients

Peak is total PyTorch allocated memory, including persistent storage, not merely
the incremental tile allocation. Times are medians, in milliseconds. Forward and
backward medians are separate measurements; summing them is an approximate
iteration cost, not a separately measured median.

| Reflections | Backend | Tile size | Peak GiB | Forward ms | Backward ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 34,405 | dense | — | 4.700 | 12.25 | 4.81 |
| 34,405 | streamed | 2,048 | 0.390 | 18.89 | 23.96 |
| 34,405 | streamed | 8,192 | 1.241 | 13.88 | 19.29 |
| 275,304 | dense | — | 37.149 | 100.47 | 31.02 |
| 275,304 | streamed | 2,048 | 0.397 | 143.10 | 172.98 |
| 275,304 | streamed | 8,192 | 1.248 | 104.54 | 130.13 |
| 929,040 | dense | — | OOM | — | — |
| 929,040 | streamed | 2,048 | 0.416 | 477.65 | 571.72 |
| 929,040 | streamed | 8,192 | 1.268 | 352.32 | 432.99 |

The 34,405-reflection values are a final 10-repeat confirmation. Larger cases
use five measured repeats; each process has one warmup. Box scales 1, 2, and 3
produce the three reflection sets at 3 Å, with fixed atomic coordinates. The
full sweep also tested forward-only and all-four-parameter gradients at each
size/backend/tile configuration: 27 configurations, of which 24 completed and
three dense configurations OOMed at 929,040 reflections. Streamed all-parameter
gradients at that size peaked at 0.483 GiB (2,048 tile) or 1.582 GiB (8,192 tile).

For the smallest coordinates-only case, saved-autograd unique storage fell from
1,659,284,412 bytes to 1,569,052 bytes (approximately 1.55 GiB to 1.50 MiB). This
is saved-storage accounting, not an additional allocator peak measurement.

## Interpretation and limitations

- Default tiles reduce peak by roughly 12x for the small case and 94x for the
  middle case. Workspace stays nearly fixed as reflections increase; compact
  per-element tables, outputs, and per-tile bookkeeping still grow.
- This is a **memory optimization, not a speedup**. Checkpointing recomputes tile
  intermediates during backward. Default-tile forward-plus-backward costs are
  about 2.4–2.5x the dense costs where dense fits. Larger tiles reduce overhead at
  the cost of more GPU memory.
- The dense comparison uses unchanged legacy arithmetic with the new compact/lazy
  constructor. It also retains a small compact scattering table alongside the
  expanded dense table. It is not a measurement of the historical constructor.
  The initial sweep materialized the dense table during warmup; the final small
  confirmation includes that preparation in constructor timing. Steady-state
  iteration peaks include the table in both cases.
- CUDA event timings and PyTorch allocation peaks do not include every CPU or
  driver/library cost. These results do not guarantee that a whole model-guided
  refinement fits, or that its overall runtime changes by the same factor.
- Reading/assigning the public `fullsf_tensor` intentionally restores dense
  execution for compatibility. Batched-model execution remains dense.

## Correctness validation

The final GPU run passed 218 tests, including the original scientific tests,
frozen complex outputs and both gradient probes, irregular chunk boundaries,
all four parameter gradients, lazy-cache mutation/inference transitions,
coordinates-only gradients, a Hessian-vector comparison, and compact allocation
and saved-storage guards. The network-dependent PDB-download test was explicitly
excluded. Frozen NPZ inputs and expected outputs were not regenerated.

See [README.md](README.md) for reproduction commands and profiling details. Raw
local reports and scheduler logs stay under the ignored `results/` directory;
this document contains only curated, public-input measurement summaries.
