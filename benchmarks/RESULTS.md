# Structure-factor backend measurements

## Compiled CUDA sweep

Measured on an NVIDIA H100 80 GB GPU with PyTorch 2.7.1 (CUDA build 12.6),
NumPy 1.26.4, Gemmi 0.6.7, float32/complex64, and float32 matrix precision
`highest`. Performance inputs are exclusively the bundled public 1LJ5 model
(3,012 atoms), P1 symmetry, and isotropic displacement parameters. Box scales
1, 2, and 3 generate 34,405, 275,304, and 929,040 reflections at 3 Å with fixed
atomic coordinates; this expands the test box, not the molecule. These are not
multi-symmetry performance measurements or full-refinement measurements.

The `compiled-sweep-v1` sweep contains **45 configurations: 42 completed and three
dense OOMs**, one in each mode at the largest size. Each configuration ran in a
fresh process with two warmups and five measured repeats; saved-tensor accounting
and profiling used additional untimed iterations. The sweep tables below use only
these five-repeat measurements, not the earlier `compiled-pilot-v1` timings. The
separate final-source 10-repeat confirmation is reported in its own table.

Times are separate median CUDA-event forward/backward intervals in milliseconds.
Their sum is an approximate iteration cost, not a separately measured median.
These intervals include host/JIT idle gaps and are not summed kernel execution
time. Peak GiB is the maximum total PyTorch allocated memory across measured
iterations, including persistent storage, not merely incremental tile workspace.
It excludes compiler/host and non-PyTorch allocations. Warmup/JIT peaks are not
these steady-state peaks.

### Coordinates-only gradients: all backends and both tile sizes

| Reflections | Backend | Tile size | Peak GiB | Forward ms | Backward ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 34,405 | dense | — | 4.700 | 12.23 | 4.77 |
| 34,405 | streamed | 2,048 | 0.390 | 18.81 | 24.26 |
| 34,405 | streamed | 8,192 | 1.241 | 13.83 | 19.85 |
| 34,405 | compiled | 2,048 | 0.089 | 10.58 | 10.90 |
| 34,405 | compiled | 8,192 | 0.158 | 2.57 | 2.67 |
| 275,304 | dense | — | 37.149 | 100.58 | 30.87 |
| 275,304 | streamed | 2,048 | 0.397 | 145.90 | 173.64 |
| 275,304 | streamed | 8,192 | 1.248 | 104.44 | 129.96 |
| 275,304 | compiled | 2,048 | 0.107 | 47.16 | 62.58 |
| 275,304 | compiled | 8,192 | 0.176 | 13.60 | 17.29 |
| 929,040 | dense | — | OOM | — | — |
| 929,040 | streamed | 2,048 | 0.416 | 484.18 | 574.60 |
| 929,040 | streamed | 8,192 | 1.268 | 352.59 | 433.93 |
| 929,040 | compiled | 2,048 | 0.194 | 152.55 | 198.45 |
| 929,040 | compiled | 8,192 | 0.225 | 40.54 | 51.46 |

### Final-source coordinates confirmation: compiled tile 8,192

The completed `compiled-confirm-v1` confirmation used three fresh processes on the
final source, one per reflection count, with two warmups and ten measured repeats
each. It does not replace or combine samples with the five-repeat sweep above.
Times use the same CUDA-event median convention.

| Reflections | Backend | Tile size | Peak bytes | Peak GiB | Forward ms | Backward ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 34,405 | compiled | 8,192 | 169,902,592 | 0.158 | 2.34 | 2.86 |
| 275,304 | compiled | 8,192 | 189,147,648 | 0.176 | 21.77 | 17.52 |
| 929,040 | compiled | 8,192 | 241,460,736 | 0.225 | 38.56 | 48.55 |

Timing variability matters: the medium-case forward confirmation median was
**21.77 ms versus 13.60 ms in the sweep**, with confirmation samples ranging from
**17.31–172.66 ms**. The cause is not established. Both runs are retained rather
than cherry-picking the faster median or attributing the difference to a known
cause. Allocated-memory peaks were stable: all three confirmation peaks exactly
match their corresponding sweep peaks.

For this P1/isotropic workload, start with **`--backend compiled --chunk-size 8192`**
when CUDA compilation and first-order autograd are appropriate. Both sweep and
confirmation timings support it over compiled tile 2,048, at the cost of increased
peak memory. Using confirmation timings rather than the faster medium sweep
median, forward-plus-backward is approximately 5.20, 39.30, and 87.12 ms. Compared
with the sweep dense baselines, that is approximately **3.3x faster at both small
and medium sizes**. At the largest size, it is approximately **9.0x faster than
streamed at tile 8,192** (786.52 / 87.12 ms); dense OOMed. Ratios use sums of
separate medians, not measured iteration medians, and compare separate processes.
This is not a universal speedup claim: the small compiled 2,048 case is slower
than dense in the sweep, and timings vary across runs.

### All-parameter gradients: tile 8,192 versus dense

This mode differentiates coordinates, Biso, anisotropic U, and occupancies.
Although the input U values are zero, trainable U requires the general masked
anisotropic path rather than the frozen-U isotropic specialization.

| Reflections | Backend | Tile size | Peak GiB | Forward ms | Backward ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 34,405 | dense | — | 6.244 | 12.33 | 16.34 |
| 34,405 | streamed | 8,192 | 1.536 | 13.94 | 31.14 |
| 34,405 | compiled | 8,192 | 0.250 | 2.68 | 4.17 |
| 275,304 | dense | — | 49.510 | 100.74 | 162.12 |
| 275,304 | streamed | 8,192 | 1.549 | 105.77 | 226.18 |
| 275,304 | compiled | 8,192 | 0.268 | 15.46 | 22.46 |
| 929,040 | dense | — | OOM | — | — |
| 929,040 | streamed | 8,192 | 1.582 | 357.08 | 760.39 |
| 929,040 | compiled | 8,192 | 0.317 | 56.33 | 80.20 |

### Forward-only and first-call costs

Forward-only medians at tile 8,192 were 1.52, 8.63, and 28.34 ms for compiled,
versus 13.36, 103.73, and 349.59 ms for streamed, in increasing reflection-count
order. Compiled peaks were 0.126, 0.137, and 0.166 GiB. Dense took 12.23 and
100.65 ms (3.897 and 30.940 GiB), then OOMed at the largest size.

Across all compiled sweep configurations, the first warmup's forward plus backward
wall time ranged from **3.5–15.8 seconds** (forward only in no-gradient mode).
`first_call` is the first entry in `warmup_samples`, not an extra iteration;
`warmup_samples`/`warmup_summary` are separate from steady-state `samples`/`summary`.
Compiler caches were not cleared: this is cache-dependent lazy setup/JIT cost,
**not a guaranteed cold compile** or a pure compiler-time measurement. Tail sizes,
shapes, and grad/inference modes can compile additional kernels later. The
signature-keyed LRU retains up to 32 static wrappers without changing global
Dynamo configuration; wrapper eviction does not clear on-disk compiler caches.

### What kernel inspection actually establishes

For the 34,405-reflection coordinates case, the untimed sweep profile recorded:

| Backend | Tile size | CUDA device-event calls | Summed device-event ms |
| --- | ---: | ---: | ---: |
| dense | — | 95 | 15.97 |
| streamed | 2,048 | 1,425 | 30.65 |
| compiled | 2,048 | 341 | 3.10 |

The streamed/compiled comparison uses the **same 2,048 tile size**. These profiles
include loss projection and validation, and CUDA device events can include copies
and memsets. They are separate from the timed benchmark iterations; fewer calls
and lower summed kernel time do not eliminate host launch/control overhead.

Generated kernel bodies were inspected, not just profiler names. For the
P1/frozen-U isotropic forward, the compiled path still materializes one float32
atom-by-tile dot-product buffer **before multiplication by 2π**: 3,012 × 2,048 ×
4 bytes = **23.53 MiB**. Scattering gather, B-factor exponential, sine/cosine,
occupancy multiplication, and partial reduction are fused; the reduction uses
small 24-partial buffers rather than full atom-by-tile intermediates for each
operation. The inspected coordinates backward fuses coordinate derivatives and
reductions without an additional atom-by-tile buffer, but checkpoint recomputation
still executes the forward and incurs its dot-product buffer.

The general-U forward also materializes an atom-by-tile damping buffer. Ustar and
symmetry contractions still use external batched matrix multiplications. This is
**not zero-intermediate or all-fused execution**, and the P1/isotropic speedup must
not be extrapolated to multiple symmetry operators or general anisotropic cases.

### Correctness and scope of the final source

The final GPU suite passed **282 tests, with one deselected** network-dependent
`test_fetchpdb`; all **52 compiled-backend tests passed**. Coverage includes frozen
complex outputs and both VJPs, the full approximately 34k-reflection reference in
coordinates/all-parameter modes at tiles 2,048 and 8,192, inference-to-training
transitions for positions/occupancy/all parameters, more than eight signatures
without a compiler reset, and multiple outstanding graphs. CPU checks also cover
dtype validation and connected zero cell-transform gradients. The final CPU suite
reported **192 passed, 90 skipped, one deselected**. CPU compiled-core arithmetic
checks use `compile_kernel=False`; actual CUDA/Inductor tests require
`SFC_TEST_COMPILED_CUDA=1`. Frozen fixtures and expected outputs were unchanged.

After the sweep, source changes only strengthened dtype validation, preserved the
connected zero VJP for trainable `orth2frac`, and adjusted formatting. The measured
float32 benchmark path is unchanged. The five-repeat sweep uses the sweep source;
the separate ten-repeat coordinates/tile-8,192 confirmation uses the final source.
Not every sweep configuration was rerun on the final source.

The default remains `fprotein_backend="streamed"`. Compiled is an explicit CUDA
opt-in with **first-order gradients only** and no silent eager fallback. Choose
streamed **before a fresh forward** for higher-order derivatives. Checkpointed
dense workspace is bounded by atoms times tile size; compact inputs, outputs, and
per-tile autograd bookkeeping still grow with reflections. Public dense-cache
compatibility is unchanged: reading/assigning `fullsf_tensor` intentionally
restores dense execution. Batched-model execution remains dense. These isolated
results do not establish full-pipeline memory use or refinement performance.

## Historical reflection-streaming measurements

The following earlier dense-versus-streamed results are preserved as historical
evidence; their repeat/warmup protocol differs from the compiled sweep above.
Their memory-versus-speed interpretation applies to **streamed**, not compiled.

Measured on one NVIDIA H100 80 GB GPU with PyTorch 2.7.1 (CUDA build
12.6), NumPy 1.26.4, Gemmi 0.6.7, float32/complex64, and float32 matrix precision
`highest`. No other application graph is included. All inputs derive from the
bundled public 1LJ5 tutorial-target model (3,012 atoms) and synthetic reflection
sets; no private scientific data is required.

### Historical coordinates-only gradients

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

### Historical interpretation and limitations (streamed)

- Default tiles reduce peak by roughly 12x for the small case and 94x for the
  middle case. Workspace stays nearly fixed as reflections increase; compact
  per-element tables, outputs, and per-tile bookkeeping still grow.
- Streamed is a **memory optimization, not a speedup** in these historical results.
  Checkpointing recomputes tile intermediates during backward. Default-tile
  forward-plus-backward costs are
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

### Historical correctness validation

The GPU run for this historical streaming measurement passed 218 tests, including
the original scientific tests, frozen complex outputs and both gradient probes,
irregular chunk boundaries,
all four parameter gradients, lazy-cache mutation/inference transitions,
coordinates-only gradients, a Hessian-vector comparison, and compact allocation
and saved-storage guards. The network-dependent PDB-download test was explicitly
excluded. Frozen NPZ inputs and expected outputs were not regenerated.

See [README.md](README.md) for reproduction commands and profiling details. Raw
local reports stay under the ignored `results/` directory;
this document contains only curated, public-input measurement summaries.
