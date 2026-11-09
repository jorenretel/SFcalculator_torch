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

The legacy `fullsf_tensor` attribute remains available as a **lazy, cached dense
table**. Reading or assigning it intentionally restores dense execution, preserving
existing in-place edits. Avoid accessing it when memory-bounded execution is
needed. The batched-model API retains its legacy dense behavior. Higher-order
reverse-mode gradients are supported, but their graphs do not have the same
first-order memory bound; `torch.func`/forward-mode transforms are not qualified.
Do not mutate inputs before backward finishes. This path requires a PyTorch
version supporting non-reentrant checkpointing (tested with PyTorch 2.7 and 2.14).

See [the benchmark guide](benchmarks/README.md) for frozen output/gradient tests,
CUDA memory measurements, and repeatable dense-versus-streamed comparisons.

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

