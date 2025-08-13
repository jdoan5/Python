# python-cli-skeleton

A clean, dependency-free starter for Python CLIs using the standard library.

## Quick start

```bash
# 1) Install in editable mode
pip install -e .

# 2) Use it
mycli greet Alice
mycli -v sum 1 2 3
mycli version --json
```

Or run without installing:

```bash
python -m cli_app greet Alice
```

## Commands

- `greet <name> [--yell]` — prints a greeting
- `sum <numbers...>` — prints the sum of given numbers
- `version [--json]` — prints the CLI version

## Packaging

Build a wheel:

```bash
python -m build
```

(Requires `pip install build`.)
