# Week 2 Data Work Project

This project is set up for the week 2 data work assignments. It uses `uv` for environment and dependency management.

## Project Structure

```
week2-data-work/
  data/
    raw/
    cache/
    processed/
    external/
  reports/figures/
  scripts/
  src/bootcamp_data/
    __init__.py
    config.py
    io.py
    transforms.py
  tests/
  README.md
  pyproject.toml
```

The Python source code for this project is located in the `src` directory. The project is configured to allow direct imports from `src` without prefixing with `src.`. For example, you can import `bootcamp_data` as `import bootcamp_data`.

## Setup and Installation

1.  **Install `uv`**:
    If you don't have `uv` installed, follow the official installation instructions:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a virtual environment**:
    Navigate to the project root directory and create a virtual environment.
    ```bash
    uv venv
    ```
    This will create a `.venv` directory in your project folder.

3.  **Activate the virtual environment**:
    On macOS and Linux:
    ```bash
    source .venv/bin/activate
    ```
    On Windows:
    ```bash
    .venv\Scripts\activate
    ```

4.  **Install dependencies**:
    With the virtual environment activated, install the project dependencies in editable mode. This allows you to make changes to the source code and have them reflected immediately without reinstalling.
    ```bash
    uv pip install -e ".[dev]"
    ```
    This command installs both the project dependencies and the development dependencies (like `pytest`).

## Running the Code

With the environment set up and dependencies installed, you can run Python scripts. For example, if you have a script in the `scripts` directory that uses the `bootcamp_data` package, you can run it directly:

```bash
python scripts/your_script.py
```

Inside `your_script.py`, you can import modules from `src` without the `src.` prefix:

```python
import bootcamp_data
from bootcamp_data.transforms import enforce_schema

# Your code here
```

## Running Tests

To run the tests, use `pytest`:
```bash
pytest
```
Or, to ensure you are using the `pytest` from the virtual environment, you can also run:
```bash
uv run pytest
```
