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