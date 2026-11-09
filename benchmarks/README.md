# Structure-factor performance measurements

Frozen complex-output and vector-Jacobian product (VJP) tests check numerical
equivalence. This harness separately measures performance through the public
`SFcalculator.calc_fprotein()` API, including its constructor and default stored
parameters. `--backend streamed` uses the default memory-bounded path;
`--backend dense` uses the unchanged dense arithmetic kernel; `--backend compiled`
opts into CUDA `torch.compile`/Inductor pure real-tensor tile arithmetic. The
library default remains `fprotein_backend="streamed"`.

Use the source checkout and an environment with PyTorch and the project/test
dependencies. No extra benchmark package, model download, or external service is
needed. Inputs are the bundled public 1LJ5 tutorial-target model or synthetic cases.

## Pixi environment

The standalone `pixi.toml` installs this checkout in editable mode, its declared
scientific/test dependencies, and PyTorch using only public package channels.
The default environment is CPU-only on Linux or Apple Silicon; `gpu` is Linux-only
with CUDA 12.8 runtime libraries and requires a compatible NVIDIA driver. Dense
and streamed benchmarks do not need a system CUDA toolkit. The optional compiled
backend additionally requires CUDA development headers and the `ptxas` compiler;
runtime-only PyTorch installations may lack them.

For compiled runs, install a compatible [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
using the generic toolkit installation instructions. If header or compiler discovery
fails, configure `CPATH` to include the toolkit headers and `TRITON_PTXAS_PATH` to
point to its `ptxas` executable. Do not treat runtime CUDA availability alone as proof
that the development toolchain is installed. Compilation errors are surfaced;
there is no silent eager fallback.

From the repository root:

```sh
pixi run test-reference
pixi run -e gpu test-reference
```

Run the benchmark through Pixi rather than the shell's base `python`:

```sh
mkdir -p benchmarks/results
pixi run -e gpu benchmark --device cuda:0 --mode coordinates \
  --full-reflections --resolution 3 --saved-tensors \
  --output benchmarks/results/baseline-coordinates.json
```

Pixi creates the chosen environment on first use; no manual `pip install` is
required. For a CPU harness smoke test, use `pixi run benchmark --device cpu` with
an `--output` path. The equivalent direct-Python commands below assume an already
prepared environment.

## Run a baseline

From the repository root:

```sh
mkdir -p benchmarks/results
PYTHONPATH=. python -m benchmarks.fprotein --device cuda:0 \
  --mode coordinates --saved-tensors \
  --output benchmarks/results/baseline-small-coordinates.json
```

Run each configuration in a **fresh process**, not in a notebook or inside a
training process. Keep the GPU otherwise idle. Run the same command with different
output filenames on each implementation; outputs are never overwritten. Compare
these three modes:

- `--mode forward`: no gradients, representing the detached RSCC calculation.
- `--mode coordinates`: backpropagation to coordinates only, representing the main
  geometry-refinement path.
- `--mode all`: gradients to coordinates, Biso, anisotropic U, and occupancies.

The default is the 257-reflection public-model fixture, suitable for a smoke test.
For memory scaling use a full physical reciprocal ASU:

```sh
PYTHONPATH=. python -m benchmarks.fprotein --device cuda:0 \
  --mode coordinates --full-reflections --resolution 3 \
  --cell-scale 2 --repeats 5 \
  --output benchmarks/results/baseline-full-coordinates.json
```

`--cell-scale` expands the **test box**, keeping coordinates fixed; it does not
replicate atoms or claim a new experimental map. Reflection counts grow roughly
with box volume and inverse resolution cubed. Reports record the actual atom,
requested-reflection, and ASU-reflection counts. For X-ray cases, a requested subset
may still cause the constructor to prepare a larger ASU: compare actual dimensions,
not just the requested count. All generated HKLs are unique and deterministically
shuffled. Very large cases can legitimately OOM: the CLI writes a structured OOM
report and exits with status 2. Increase sizes deliberately on a CUDA host; do not
use a full-size stress test merely to validate the harness on a laptop.

For CPU protocol validation only, use `--device cpu`. CPU CUDA-memory fields are
`null`, not zero. CPU measurements do not establish CUDA performance or memory use.

## Dense, streamed, and optional compiled comparison

Use `--backend dense`, `--backend streamed`, or `--backend compiled`;
`--chunk-size` controls the streamed/compiled reflection tile size (default 2048).
A larger tile trades more workspace for fewer tile launches. Based on the measured
public 1LJ5 P1/isotropic cases, start compiled runs with `--chunk-size 8192` for
speed, or use 2048 when a smaller peak is more important. This is a workload-specific
recommendation, not a change to the default tile size. Using final-source ten-repeat
coordinates confirmation timings against the sweep baselines, combined
forward/backward is approximately 3.3x faster than dense at small/medium sizes and
9.0x faster than streamed 8192 at the largest size, where dense OOMs. These are
ratios of sums of separate medians, not measured iteration medians. See
[RESULTS.md](RESULTS.md) for the preserved five-repeat sweep, separate confirmation
table, timing variability, kernel inspection, and comparison limits. Ordinary
library calls remain streamed by default. Explicit compiled execution is CUDA-only.

The compiled backend compiles only pure real-tensor tile arithmetic; complex output
assembly and the reflection-tile loop stay outside the compiled graph.
Non-reentrant checkpointing bounds dense first-order workspace by atoms times tile
size, not total reflections. Compact inputs, outputs, and per-tile autograd
bookkeeping still grow with total reflections. Compiled autograd is **first-order
only**. For higher-order derivatives, choose `fprotein_backend="streamed"` before
a fresh forward; switching after forward cannot repair an already compiled graph.
Higher-order streamed graphs do not retain the first-order memory bound.

Static variants use a signature-keyed LRU cache of up to 32 wrappers without
changing global Dynamo configuration. Tail tiles, different shapes, and gradient
or inference modes can compile additional kernels. Wrapper reuse is not a count
of actual compilations, and eviction does not clear on-disk compiler caches.

For a single compiled run, with first-call/warmup accounting and an untimed CUDA
kernel summary:

```sh
pixi run -e gpu python -m benchmarks.fprotein --device cuda:0 \
  --backend compiled --mode coordinates --full-reflections --resolution 3 \
  --chunk-size 8192 --warmup=2 --profile \
  --output benchmarks/results/compiled-coordinates.json
```

For an automated sweep, explicitly include all three backends, with each
configuration in a fresh process (the sweep default remains dense and streamed):

```sh
pixi run -e gpu python -m benchmarks.sweep \
  --backends dense streamed compiled \
  --scales 1 2 3 --modes forward coordinates all --chunks 2048 8192 \
  --warmup=2 --profile --output-dir benchmarks/results/comparison
```

The output directory must be new. The sweep retains structured OOM reports and
continues to later configurations, checks source hashes have not changed during
the comparison, and writes `summary.json`. A process timeout is an error, not an
OOM measurement. Run this only on an otherwise idle GPU, with adequate CPU memory.
Dense runs once per shape/mode; streamed and compiled run at each requested chunk
size. Allow enough process timeout for JIT compilation when selecting `--timeout`.
The completed compiled sweep has 45 configurations: 42 completed and three dense
OOMs at the largest reflection count. [RESULTS.md](RESULTS.md) reports the five-repeat
measurements and the completed final-source coordinates/tile-8,192 confirmation
(three fresh processes, ten repeats and two warmups each). The medium forward
confirmation median was 21.77 ms versus 13.60 ms in the sweep, with a confirmation
range of 17.31–172.66 ms; the cause is not known. Both measurements are retained,
and allocated-memory peaks matched exactly at all three sizes.
Compiled first-call forward/backward wall time ranged from 3.5–15.8 seconds across
the sweep, including cache-dependent lazy setup/JIT, not a guaranteed cold compile.
Performance evidence is limited to the public 1LJ5 P1/isotropic workload, not
multi-symmetry performance or a complete refinement pipeline.

**Baseline qualification:** `dense` here means the legacy arithmetic kernel with
the new compact/lazy constructor, not the untouched historical constructor. The
dense backend retains both compact and expanded tables; old recorded reports
remain the historical baseline. Constructor measurements include explicit dense
table preparation when requested. Steady-state iteration peaks include all live
storage regardless of where it was allocated. Public `fullsf_tensor` cache
compatibility is unchanged: reading or assigning this lazy dense table makes it
authoritative and selects dense execution, including when `compiled` was selected.
Avoid accessing it when measuring memory-bounded execution. Neither the old nor
new benchmark includes a live OpenFold graph.

## Frozen outputs and VJPs

From the repository root, run the reference and optional-backend tests with:

```sh
PYTHONPATH=. python -m pytest tests/test_fprotein_reference.py tests/test_fprotein_compiled.py
```

CPU compiled-core checks explicitly use `compile_kernel=False`: they test the
same pure real-tensor arithmetic against frozen complex outputs and both VJPs,
not Inductor compilation or CUDA performance. This low-level testing switch is
not a public CPU fallback for `fprotein_backend="compiled"`. Actual CUDA/Inductor
checks are expensive and require explicit opt-in plus the development toolchain:

```sh
SFC_TEST_COMPILED_CUDA=1 PYTHONPATH=. python -m pytest tests/test_fprotein_compiled.py
```

The final H100/PyTorch 2.7.1 GPU suite reported **282 passed, one deselected**
(the network-dependent `test_fetchpdb`), including **all 52 compiled-backend tests**.
The final CPU suite reported **192 passed, 90 skipped, one deselected**. Frozen
fixtures and expected outputs were unchanged. Coverage includes the full
approximately 34k-reflection reference in coordinates/all-parameter modes at tiles
2,048 and 8,192, inference-to-training transitions for positions/occupancy/all
parameters, more than eight signatures without a compiler reset, multiple
outstanding graphs, and CPU dtype/connected-zero cell-transform VJP checks.
Public model inputs are limited to the bundled 1LJ5 fixtures; other cases are
synthetic. See [RESULTS.md](RESULTS.md) for measurement provenance and final-source
qualifications; test coverage is distinct from benchmark performance evidence.

## Measurements and interpretation

JSON includes:

- Revision, tracked-dirty flag, and hashes of scientific source, harness, fixture
  adapter, and fixture manifest. The hashes identify uncommitted code as well.
- PyTorch/CUDA runtime versions, GPU model/capacity/free memory, allocator backend,
  thread count, float32 matrix precision (`highest`), seed, and input dimensions.
- Constructor time, constructor peak allocated/reserved memory, and live memory.
  This is important when optimizing persistent scattering-factor storage.
- `first_call`: the first warmup forward/backward in this process, including lazy
  setup and any JIT work. It is also the first entry in `warmup_samples`, not an
  extra iteration. Compiler caches are not cleared, so this is cache-dependent,
  **not a guaranteed cold compile**; later calls may specialize/recompile too.
- `warmup_samples` and `warmup_summary`: warmup iterations reported separately.
  Use `--warmup=2` for the comparisons above (the CLI default is 1).
- `samples` and `summary`: measured iterations' synchronized wall time and
  CUDA-event time for forward and backward, plus median/min/max. First-call and
  all warmup samples are excluded from these steady-state statistics. CUDA-event
  intervals can include host/JIT idle gaps; they are not summed kernel time.
- Live memory before forward and after forward; forward/backward peak allocated and
  reserved memory; total operation allocated peak and incremental peak above the
  live iteration baseline. Loss-projection temporaries count toward backward peak;
  projection execution time is not included in forward/backward timings.
- Optional `--saved-tensors`: an additional **untimed** iteration counting autograd
  save calls, logical saved bytes, and distinct underlying storage bytes. Views
  sharing storage are deduplicated. This is diagnostic accounting, not an allocator
  peak measurement, and storage already present before forward can be included.
- OOM phase and operation, without exception text or machine paths. Partial samples
  are retained. The OS killing a process cannot be caught by Python.

No `empty_cache()` is called between iterations: normal steady-state allocator
behavior is intentional. Old public output references and leaf gradients are
cleared before each iteration to avoid accidentally retaining a previous graph.
Backward uses a deterministic complex projection that exercises both real and
imaginary output components. Output/gradient finiteness is checked outside timing;
**numerical equivalence is still enforced by the frozen integration tests**, not
by this benchmark's finiteness checks.

PyTorch allocated/reserved peaks exclude compiler/host memory and non-PyTorch
allocations, including some driver/library allocations; CPU memory is not measured.
This isolated benchmark does not include a live OpenFold graph. A full-pipeline
validation remains necessary even after an isolated operation fits in memory.

## Optional kernel summary and operator trace

`--profile` adds a separate untimed profiler iteration and a JSON CUDA kernel
summary (names, call counts, and total durations). It is available for individual
runs and sweeps. Inspect it for launches and kernel composition; kernel names or
counts alone do not prove fusion or a speedup. Profiling and saved-tensor inspection
are excluded from steady-state samples.

Use `--trace` for an optional Chrome trace from that untimed profiler iteration:

```sh
PYTHONPATH=. python -m benchmarks.fprotein --device cuda:0 \
  --trace benchmarks/results/operators.json \
  --output benchmarks/results/profiled-run.json
```

Profiling is a separate, untimed iteration so profiler overhead does not pollute
reported timings. Open the Chrome trace in a compatible trace viewer to inspect
operators, shapes, allocations, and kernel launches. Known identifying top-level
fields (`host_name`, `traceName`, `deviceProperties`, `distributedInfo`) are removed.
**Review traces before sharing:** profiler formats and CUDA metadata can change;
this is not a general-purpose anonymizer. Regular JSON intentionally excludes user,
host, output paths, environment dumps, device UUIDs, and Git identity metadata.

`benchmarks/results/` is ignored so local reports/traces do not accidentally enter
public commit history. Use generic report names and review any results deliberately
published elsewhere.

## Optimization acceptance gate

1. Pass frozen complex-output and gradient tests unchanged on CPU and CUDA.
2. Compare forward-only and forward/backward memory and speed on the same GPU,
   same shape, same numerical settings, and otherwise idle hardware.
3. Include constructor/persistent storage, not merely temporary kernel allocations.
4. Check several reflection counts and a case that the original implementation
   cannot fit. A slower but smaller method should be reported as a tradeoff, not
   unconditionally faster.
5. Validate a full scientific refinement independently.
