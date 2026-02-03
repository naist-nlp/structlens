# StructLens

Visualizer for StructLens: [https://naist-nlp.github.io/structlens/](https://naist-nlp.github.io/structlens/)

## Installation

```bash
# uv
uv add git+https://github.com/naist-nlp/structlens.git
# pip
# pip install git+https://github.com/naist-nlp/structlens.git
```

## Usage

```python
import torch
from structlens import L2DistanceSimilarityFunction, StructLens, create_masks

num_tokens_per_layer = 5
num_layers = 3
hidden_size = 128

representations = torch.randn(num_layers, num_tokens_per_layer, hidden_size)

struct_lens = StructLens()
st_list = struct_lens(representations)
for i, st in enumerate(st_list):
    print("layer: ", i)
    print("max_score: ", st["max_score"])
    print("argmax_heads: ", st["argmax_heads"])
```

See [examples/structlens-sample](examples/structlens-sample) for more details.

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
