# DLRA-CG

Dynamical low-rank approximation with conjugate search directions (DLRA-CG) for a Tikhonov-regularized elliptic inverse source problem. This project is based on my master's thesis, [masters-thesis](https://github.com/EliasSev/masters-thesis), split off to focus on the DLRA-CG contribution and turn it into a paper.

## Requirements
- Conda / Miniforge / Mambaforge
- Linux or macOS (recommended for FEniCS)
- Python 3.9 (required by the FEniCS conda packages)


## Environment Setup
To create and activate the environment, run the following commands in a terminal:
```shell
conda env create -f environment.yml
conda activate dlra_env
python -m ipykernel install --user --name dlra_env
```
If you change `pyproject.toml` or any package names, you need to re-build the package:
```shell
conda activate dlra_env
pip install -e .
```

## Module Design

All code is implemented in the `dlra` package under `src/`, installed editable
(distribution name `dlra-cg`, importable as `dlra`):

```
src/dlra/
├── solvers/
├── problem/
├── evaluation/
├── viz/
├── rsvd.py
└── io.py
```

- **`dlra/solvers/`** — the paper's contribution: DLRA, DLRA-CG, DLRA-PCG,
  and the baselines (full-rank CG, Riemannian CG, DLRA-SD) used to validate
  them, plus the Hessian preconditioners.
- **`dlra/problem/`** — the inverse problem itself: FE assembly of the
  forward operator (`fem.py`), different test problems
  (`test_problems.py`), and mesh/source generation (`meshes.py`,
  `sources.py`).
- **`dlra/evaluation/`** — reconstruction metrics (Euclidean, EMD,
  thresholded IoU/AUC, SSIM).
- **`dlra/viz/`** — plotting helpers and the paper's color scheme.
- **`dlra/io.py`** — a progress bar and a disk-cache decorator for
  expensive experiment results.
- **`dlra/rsvd.py`** — a matrix-free randomized SVD, copied over from the thesis, but not the focus of this project.

See [`docs/PROBLEM_SETUP.md`](docs/PROBLEM_SETUP.md) for the mathematical
reference (the inverse problem, weights, and the DLRA / DLRA-CG / DLRA-PCG
algorithms) and how it maps to the code above.

## Usage

```python
from dlra import DynamicalLowRankCG
from dlra.problem import TestProblemSetup

# Generate an inverse problem (K, M_dx, M_ds, Vh, etc.)
setup = TestProblemSetup(n=32, sigma=1, c=1)
pb = setup.build(name='II')  # 'I', 'II', etc.

# DLRA-CG solver
solver = DynamicalLowRankCG(operator=pb.operator)
x_hat = solver.solve(
    y=pb.y,
    w=pb.weights,
    lambda_=1e-4,
    max_rank=2
)

# Plot the reconstruction
from fenics import Function, plot
f = Function(pb.Vh)  # pb.Vh: FunctionSpace
f.vector()[:] = x_hat
plot(f)
```

`x_hat` is the reconstructed source as a flat FE coefficient vector; see
`notebooks/01_test.ipynb` and `notebooks/02_dlra_cg_test.ipynb` for full
examples, including plotting and comparisons against other solvers.