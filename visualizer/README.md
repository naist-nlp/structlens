# StructLens Visualizer

Interactive dependency tree visualizer for StructLens output. Drop a JSON file and explore attention head dependency structures across layers with animated arc diagrams.

## Usage

```bash
cd visualizer
bun install
bun run dev
```

1. Open the app in your browser
2. Drop a StructLens JSON file (or click to browse)
3. Use the layer slider to scrub through layers, or press play to animate
4. Adjust the token range (From / To) to focus on a subset of tokens

## Input Format

```json
{
  "tokens": ["Language", "exhibits", "inherent", "structures", ",", "..."],
  "layers": [
    { "layer": 0, "argmax_heads": [0, 0, 1, 0, 3, "..."] },
    { "layer": 1, "argmax_heads": [2, 1, 0, 2, 3, "..."] }
  ]
}
```

- `argmax_heads[i] = j` — token `i` depends on token `j` (drawn as arc `i` -> `j`)
- `argmax_heads[i] = i` — root node
- `argmax_heads[i] = -1` — padding (hidden)

## Build

```bash
bun run build
```

Output goes to `dist/`.
