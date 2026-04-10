# MoE

This directory contains the MoE ranking work for `mesh-llm`.

## Contents

- [`MOE_ANALYZE_STORAGE_SPEC.md`](MOE_ANALYZE_STORAGE_SPEC.md)
  - Defines the canonical Hugging Face dataset layout for published `moe-analyze` artifacts.
  - Defines the optional colocated model-repo sidecar layout for `moe-analyze/` metadata next to GGUF files.
- [`scripts/`](scripts/)
  - Contains the dataset-generation tooling for downloading GGUF distributions, running `llama-moe-analyze`, and publishing artifacts.

## Current Scope

- GGUF source models
- `micro-v1` and `full-v1` analyzer ids
- Canonical publication to the `meshllm/moe-rankings` Hugging Face dataset

## Entry Points

- Read the storage contract in [`MOE_ANALYZE_STORAGE_SPEC.md`](MOE_ANALYZE_STORAGE_SPEC.md).
- Use [`scripts/analyze_and_publish.py`](scripts/analyze_and_publish.py) to generate and publish canonical dataset artifacts.
- See [`scripts/README.md`](scripts/README.md) for usage examples.

## Notes

- The Hugging Face dataset is the immutable system of record.
- Model-repo sidecars are documented, but not automatically generated yet.
- `micro-v1` is bound to the built-in canonical prompt set.
