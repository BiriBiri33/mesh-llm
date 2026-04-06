Implements the runtime-surface and GPU-inspection portion of #184.

## Summary

This PR makes the local runtime entrypoints explicit and adds a local GPU inspection command.

Users can now run:

```bash
mesh-llm serve ...
mesh-llm client ...
mesh-llm gpus
```

This keeps the local startup surface clearer without changing the later config work in #184.

`mesh-llm serve` is also necessary groundwork for the next config change. Once local config is introduced, `mesh-llm serve` will be the command that loads startup model configuration from `~/.mesh-llm/config.toml`. Splitting the runtime surface now gives that later config work a clear owner instead of continuing to hang model startup behavior off ambiguous top-level flags.

## New Commands

The new canonical runtime commands are:

```bash
mesh-llm serve --auto --model Qwen3-8B-Q4_K_M
mesh-llm serve --join <token>
mesh-llm client --auto
mesh-llm client --join <token>
mesh-llm gpus
```

`mesh-llm gpus` also has a `gpu` alias:

```bash
mesh-llm gpu
```

## Deprecated Top-Level Forms

These older top-level runtime forms still work in this PR, but they are now migration shims:

```bash
mesh-llm --auto --model Qwen3-8B-Q4_K_M
mesh-llm --gguf /path/to/model.gguf
mesh-llm --auto --client
mesh-llm --client --join <token>
```

They now warn and route to the new command surface.

Example:

```text
⚠️ top-level serving flags now map to `mesh-llm serve`.
  Please use: mesh-llm serve --auto --model Qwen3-8B-Q4_K_M
```

```text
⚠️ top-level `--client` now maps to `mesh-llm client`.
  Please use: mesh-llm client --auto
```

Mixed legacy usage such as top-level `--client` plus serving flags is rejected.

## Example Output

Client startup now uses the new client-oriented ready output:

```text
📡 Client ready:
  API: http://localhost:9337
  Console: http://localhost:3131
```

GPU inspection prints local GPU identity, backend device, VRAM, and cached bandwidth when available:

```text
🖥️ GPU 0
  Name: NVIDIA RTX 4090
  Stable ID: pci:0000:65:00.0
  Backend device: CUDA0
  VRAM: 24.0 GiB
  Bandwidth: 1008.0 GB/s
  Unified memory: no
```

If no GPUs are detected:

```text
⚠️ No GPUs detected on this node.
```

## Included In This PR

- `serve` and `client` as explicit local runtime entrypoints
- legacy top-level runtime rewrites with `⚠️` migration warnings
- `mesh-llm gpus` and `mesh-llm gpu`
- additive `GpuFacts` support in local hardware survey
- cached benchmark bandwidth attached to `mesh-llm gpus` output when available
- script and test updates to the new runtime surface
- README and docs updates showing the new commands

## Not Included In This PR

- unified local `~/.mesh-llm/config.toml` serve config
- pinned `gpu_id` assignment
- mesh-wide config reconciliation/editor work

## Validation

```bash
cargo test -p mesh-llm
```
