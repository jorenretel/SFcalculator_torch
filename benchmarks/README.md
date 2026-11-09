# Structure-factor performance measurements

The frozen tests establish numerical equivalence. This harness separately measures
performance through the public `SFcalculator.calc_fprotein()` API, including its
constructor and default stored parameters. `--backend streamed` uses the new
memory-bounded path; `--backend dense` uses the unchanged dense arithmetic kernel.

Use the source checkout and an environment with PyTorch and the project/test
dependencies. No extra benchmark package, model download, or external service is
needed. Inputs are the bundled public 1LJ5 tutorial-target model or synthetic cases.

## Pixi environment

The standalone `pixi.toml` installs this checkout in editable mode, its declared
scientific/test dependencies, and PyTorch using only public package channels.
The default environment is CPU-only on Linux or Apple Silicon; `gpu` is Linux-only
with CUDA 12.8 runtime libraries and requires a compatible NVIDIA driver. A system
CUDA toolkit is not needed for this benchmark.

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

## Dense-versus-streamed comparison

Use `--backend dense` or `--backend streamed`; `--chunk-size` controls the streamed
reflection tile size (default 2048). A larger tile often runs faster but uses more
memory. Ordinary library calls now default to the streamed path as well.

For an automated sweep, with each configuration in a fresh process:

```sh
pixi run -e gpu python -m benchmarks.sweep \
  --scales 1 2 3 --modes forward coordinates all --chunks 2048 8192 \
  --output-dir benchmarks/results/comparison
```

The output directory must be new. The sweep retains structured OOM reports and
continues to later configurations, checks source hashes have not changed during
the comparison, and writes `summary.json`. A process timeout is an error, not an
OOM measurement. Run this only on an allocated GPU, with adequate CPU memory.

**Baseline qualification:** `dense` here means the legacy arithmetic kernel with
the new compact/lazy constructor, not the untouched historical constructor. The
dense backend retains both compact and expanded tables; old recorded reports
remain the historical baseline. Constructor measurements include explicit dense
table preparation when requested. Steady-state iteration peaks include all live
storage regardless of where it was allocated. Neither the old nor new benchmark
includes a live OpenFold graph.

## Measurements and interpretation

JSON includes:

- Revision, tracked-dirty flag, and hashes of scientific source, harness, fixture
  adapter, and fixture manifest. The hashes identify uncommitted code as well.
- PyTorch/CUDA runtime versions, GPU model/capacity/free memory, allocator backend,
  thread count, float32 matrix precision (`highest`), seed, and input dimensions.
- Constructor time, constructor peak allocated/reserved memory, and live memory.
  This is important when optimizing persistent scattering-factor storage.
- Every measured iteration's synchronized wall time and CUDA-event time for forward
  and backward, plus median/min/max. Warmups are not included.
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

PyTorch allocated/reserved peaks do not include every driver/library allocation,
and this isolated benchmark does not include a live OpenFold graph. A full-pipeline
validation remains necessary even after an isolated operation fits in memory.

## Optional operator trace

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
