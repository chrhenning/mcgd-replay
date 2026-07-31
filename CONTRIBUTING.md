# Contributing

```bash
pip install -r requirements-dev.txt
```

## Tests

```bash
pytest
```

## Linting & formatting

Linting and formatting run via [ruff](https://github.com/astral-sh/ruff) through pre-commit (ruff's version is pinned in `.pre-commit-config.yaml`):

```bash
pre-commit install          # enable the hooks on every commit
pre-commit run --all-files  # run them manually
```

## Docstrings

Use [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) (`Args:` / `Returns:` / `Raises:` sections).
