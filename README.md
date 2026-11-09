# SFcalculator_torch
![licence](https://img.shields.io/github/license/Hekstra-Lab/SFcalculator_torch)
![Build](https://github.com/Hekstra-Lab/SFcalculator_torch/workflows/Build/badge.svg)
[![codecov](https://codecov.io/github/Hekstra-Lab/SFcalculator_torch/branch/master/graph/badge.svg?token=02GUPGPUC1)](https://codecov.io/github/Hekstra-Lab/SFcalculator_torch)
[![PyPI](https://img.shields.io/pypi/v/SFcalculator-torch?color=blue)](https://pypi.org/project/SFcalculator-torch/)
[![Tutorial](https://img.shields.io/badge/Page-Tutorial-violet)](https://rs-station.github.io/SFcalculator_torch/)
[![BioRXiv](https://img.shields.io/badge/BioRXiv-10.1101%2F2025.01.12.632630v2-b31b1b?)](https://www.biorxiv.org/content/10.1101/2025.01.12.632630v2)


A pytorch implementation of `SFcalculator`. See the [main page](https://github.com/Hekstra-Lab/SFcalculator) for more information.

To install this package:

1. Create a python environment with package manager you like ([mambaforge](https://github.com/mamba-org/mamba) recommended).

2. Install [Pytorch](https://pytorch.org/get-started/locally/)

3. Install `SFcalculator-torch`
    ```bash
    pip install SFcalculator-torch
    ```
 
For a walkthrough tutorial, check this page: https://rs-station.github.io/SFcalculator_torch/

### Memory-bounded single-model structure factors

`SFcalculator.calc_fprotein()` now defaults to reflection-tiled evaluation with
compact, per-element scattering factors. Differentiable calls use non-reentrant
PyTorch checkpointing to recompute tile intermediates during backward instead of
saving full atom-by-reflection matrices. This reduces memory, at the cost of
additional computation; it does not lower resolution or subsample reflections.

```python
from SFC_Torch import SFcalculator

calculator = SFcalculator(
    "model.pdb", mtzdata="data.mtz", device="cuda:0",
    fprotein_backend="streamed", reflection_chunk_size=2048,
)
fprotein = calculator.calc_fprotein(Return=True)
```

Increase `reflection_chunk_size` to trade more memory for fewer tile launches.
Set `fprotein_backend="dense"` to use the unchanged legacy kernel for comparison.
Both settings are also available as calculator attributes for subsequent calls.
The default remains `fprotein_backend="streamed"`.

The legacy `fullsf_tensor` attribute remains available as a **lazy, cached dense
table**. Reading or assigning it intentionally restores dense execution, preserving
existing in-place edits. Avoid accessing it when memory-bounded execution is
needed; this public cache compatibility is unchanged for all backends, including
`compiled`. The batched-model API retains its legacy dense behavior. The streamed
backend supports higher-order reverse-mode gradients, but their graphs do not have
the same first-order memory bound; `torch.func`/forward-mode transforms are not
qualified. Do not mutate inputs before backward finishes. The streamed path requires
a PyTorch version supporting non-reentrant checkpointing (tested with PyTorch 2.7
and 2.14).

### Optional compiled CUDA tiles

Set `fprotein_backend="compiled"` explicitly to use a CUDA `torch.compile`/Inductor
backend instead of streamed eager tiles:

```python
calculator = SFcalculator(
    "model.pdb", mtzdata="data.mtz", device="cuda:0",
    fprotein_backend="compiled", reflection_chunk_size=8192,
)
fprotein = calculator.calc_fprotein(Return=True)
```

Only the pure real-tensor tile arithmetic is compiled; complex output assembly and
the reflection-tile loop remain outside the compiled graph. Non-reentrant
checkpointing recomputes tile intermediates for **first-order gradients only**,
bounding dense workspace by atoms times tile size. Compact scattering inputs,
outputs, and per-tile autograd bookkeeping still scale with total reflections.
For higher-order derivatives, set `calculator.fprotein_backend = "streamed"`
**before a fresh forward**; changing it after forward does not change that graph.

Compilation requires CUDA development headers and the `ptxas` compiler, which
runtime-only PyTorch installations may lack. Install a compatible
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads); configure `CPATH` for
its headers and `TRITON_PTXAS_PATH` for its `ptxas` executable if needed. Explicit
compiled execution requires CUDA and does not silently fall back to eager execution
if compilation fails. The intentional dense-cache behavior described above is
unchanged.

A signature-keyed LRU cache retains up to 32 static tile variants without changing
global Dynamo configuration. Short tail tiles, new shapes, and different gradient
or inference modes can compile additional kernels. First-use JIT cost depends on
compiler caches; it is not a steady-state timing or necessarily a cold compile.

The compiled path requires float32 geometry/atomic parameters and float32 or
complex64 scattering factors. Frozen complex-output and vector-Jacobian product
(VJP) tests distinguish CPU arithmetic checks (`compile_kernel=False` in the
low-level core) from actual CUDA Inductor execution, enabled with
`SFC_TEST_COMPILED_CUDA=1`. The final H100/PyTorch 2.7.1 suite passed 282 tests,
including all 52 compiled-backend tests; one network-dependent test was excluded.
Frozen fixtures and tolerances were unchanged.

For the measured public P1/isotropic workload, 8,192-reflection compiled tiles
provided a useful speed/memory tradeoff; the default tile size remains 2,048.
See [measured results and limitations](benchmarks/RESULTS.md), including first-use
compilation overhead and timing variability. Full-refinement validation is still
required; these isolated measurements do not establish whole-pipeline performance.

See [the benchmark guide](benchmarks/README.md) for test commands, compiler setup,
separate first-call/warmup measurements, and dense/streamed/compiled comparisons.

### Citing

```
@article{li2025sfcalculator,
  title={SFCalculator: connecting deep generative models and crystallography},
  author={Li, Minhuan and Dalton, Kevin M and Hekstra, Doeke Romke},
  journal={bioRxiv},
  pages={2025--01},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

