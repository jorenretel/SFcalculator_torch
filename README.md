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

### Memory-bounded structure factors

The existing dense backend remains the default. For large single-model calculations,
select reflection tiles at construction instead of full atom-by-reflection intermediates:

```python
from SFC_Torch import SFcalculator

calculator = SFcalculator(
    "model.pdb", dmin=3.0, set_experiment=False, device="cuda:0",
    fprotein_backend="compiled", reflection_chunk_size=8192,
)
fprotein = calculator.calc_fprotein(Return=True)
```

`"streamed"` uses the original arithmetic with non-reentrant checkpointing on CPU
or CUDA, trading recomputation for memory. `"compiled"` additionally fuses tile
arithmetic with `torch.compile`/Inductor (tested with PyTorch 2.7); it requires CUDA
development headers and `ptxas`, incurs first-use compilation, and supports only
first-order gradients. Use `"streamed"` for higher-order derivatives. Larger tiles
use more memory; compact inputs and outputs still scale with reflection count.
Do not mutate inputs before backward finishes. Reading or assigning the legacy
`fullsf_tensor` intentionally restores dense execution to preserve in-place edits;
batched calculations also remain dense.

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

