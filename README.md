# Python Assignment 01

This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies and environments for JupyterLab, Jupytext, and real-time collaboration.

## Requirements

- Python 3.12 or newer
- uv (install via `curl -LsSf https://astral.sh/uv/install.sh | sh` or follow the official instructions for your OS)

## Quick start

```bash
uv sync
uv run python -m ipykernel install --user --name "$(basename "$PWD")"
uv run jupyter lab
```

## Environment setup

```bash
# create and activate a uv-managed virtual environment
uv venv
source .venv/bin/activate

# install all runtime and development dependencies
uv sync
```

`uv sync` ensures that JupyterLab, Jupytext, ipykernel, and related tools are installed according to the versions pinned in `uv.lock`.

## Working with JupyterLab and Jupytext

- Launch the notebook environment:

  ```bash
  uv run jupyter lab
  ```

- This project stores notebooks in two synchronized formats:

  - Jupyter notebook files (`.ipynb`)
  - Python percent scripts (`.py:percent`)

- Pairing and sync behavior are configured in `pyproject.toml` under `[tool.jupytext]`.
- To manually synchronize notebook/script pairs, run:

  ```bash
  uv run jupytext --sync path/to/notebook.ipynb
  ```

When you edit or save notebooks in JupyterLab, both representations stay synchronized automatically.

## Collaboration

JupyterLab 4+ supports real-time collaboration through the `jupyter-collaboration` extension. When running the lab server, multiple users can connect to the same notebook URL and edit simultaneously. All collaboration state is stored locally and does not need to be version controlled.

## Notes

- The registered kernel name is derived from your project directory.
- Always select this project’s kernel (for example, `python-uppgift01`) inside JupyterLab to ensure notebooks execute in the correct uv environment.
- Use `uv run` for all Jupyter-related commands to guarantee consistent dependency resolution.
