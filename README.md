# StructLensq

## Development

### Setup

Install dependencies:

```bash
uv sync
```

Prepare pre-commit:

```bash
uv run lefthook install
```

### Running Tests

```bash
uv run pytest
```

### Formatting & Linting

```bash
uv run ruff check --fix
```

### Type Checking

```bash
uv run ty check
```
