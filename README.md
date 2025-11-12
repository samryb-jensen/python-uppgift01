# Python Assignment 01

Interactive MNIST playground powered by PyTorch and managed with [uv](https://docs.astral.sh/uv/). Notebooks and Python percent scripts stay in sync via Jupytext, and you can either explore the project inside JupyterLab or run the CLI directly with `python` through uv.

## Requirements

- Python 3.12 or newer
- uv (install via `curl -LsSf https://astral.sh/uv/install.sh | sh` or follow the official instructions for your OS)

## Setup

```bash
# install all runtime and development dependencies
uv sync

# (optional) register the kernel so it shows up in JupyterLab
uv run python -m ipykernel install --user --name "$(basename "$PWD")"
```

`uv sync` creates a uv-managed virtual environment, installs the packages pinned in `uv.lock`, and ensures JupyterLab, Jupytext, torch, and testing tools are available.

## Running the project

There are two supported workflows—pick whichever fits how you want to interact with the MNIST demo.

### 1. JupyterLab + notebooks (exploratory)

```bash
uv run jupyter lab
```

- Open `main.ipynb`, `mnist.ipynb` and select the `python-uppgift01` kernel (or whatever name you chose during `ipykernel install`).
- Notebook files (`.ipynb`) mirror their `.py` counterparts via Jupytext’s [`py:percent`](https://jupytext.readthedocs.io) format, so edits in either place stay synchronized.
- Use `uv run jupytext --sync path/to/notebook.ipynb` if you need to force a manual resync.

### 2. Direct CLI (training + inference without notebooks)

```bash
uv run python main.py
```

- The script asks whether to reuse existing weights in `mnist_cnn.pt` or retrain the CNN from scratch (default 10 epochs).
- On the first run, the MNIST dataset downloads to `data/`; subsequent runs reuse the cached files.
- After training/loading, you can repeatedly classify test samples, optionally plotting the digit with Matplotlib as you go.

## Notebook/script pairing details

- Pairing rules live in `pyproject.toml` under `[tool.jupytext]` with `formats = "ipynb,py:percent"`.
- Any `.ipynb` saved in JupyterLab automatically updates its sibling `.py` file, keeping the repo friendly for reviews and diffs.

## Testing

Unit tests cover the MNIST helper module. Run them with:

```bash
uv run pytest
```

## Collaboration tips

- JupyterLab 4 ships with the `jupyter-collaboration` extension, so you can share a running Lab URL for real-time editing.
- Collaboration state stays local; you only need to commit the synced notebooks/scripts.

## Notes

- Always invoke tooling through `uv run …` to ensure you use the managed environment.
- The registered kernel name defaults to your project folder (for example, `python-uppgift01`).
