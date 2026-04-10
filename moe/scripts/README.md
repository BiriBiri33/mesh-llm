# MoE Scripts

## `analyze_and_publish.py`

Downloads a GGUF distribution from Hugging Face, runs `llama-moe-analyze`, writes artifacts in the canonical dataset layout from [../MOE_ANALYZE_STORAGE_SPEC.md](../MOE_ANALYZE_STORAGE_SPEC.md), and can upload those artifacts to a dataset repo.

Run it with `uv`:

```bash
uv run moe/scripts/analyze_and_publish.py \
  --source-repo unsloth/GLM-5.1-GGUF \
  --source-revision main \
  --distribution-id GLM-5.1-UD-IQ2_M \
  --analyzer-source local \
  --analyzer-bin /absolute/path/to/llama-moe-analyze \
  --analyzer-id micro-v1 \
  --dataset-repo your-org/moe-rankings
```

Bootstrap `llama-moe-analyze` from GitHub releases:

```bash
uv run moe/scripts/analyze_and_publish.py \
  --source-repo unsloth/GLM-5.1-GGUF \
  --distribution-id GLM-5.1-UD-IQ2_M \
  --analyzer-source release \
  --release-repo michaelneale/mesh-llm \
  --release-tag latest \
  --analyzer-id micro-v1 \
  --dataset-repo your-org/moe-rankings
```

Dry run:

```bash
uv run moe/scripts/analyze_and_publish.py \
  --source-repo unsloth/GLM-5.1-GGUF \
  --distribution-id GLM-5.1-UD-IQ2_M \
  --analyzer-source release \
  --dry-run
```

Notes:

- `micro-v1` runs one `llama-moe-analyze` pass per prompt and combines the resulting CSVs.
- The built-in analyzer ids do not accept `--prompt-file`.
- `micro-v1` is bound to the built-in canonical prompt set.
- `full-v1` runs a single `llama-moe-analyze` pass.
- The script forces CPU execution with `-ngl 0`.
- `--analyzer-source local` uses a locally built `llama-moe-analyze`.
- `--analyzer-source release` downloads a release bundle from GitHub and extracts `llama-moe-analyze` into `.moe-cache/tools/`.
- The canonical artifact directory shape is:

```text
data/<namespace>/<repo>/<revision>/gguf/<distribution_id>/<analyzer_id>/
```
