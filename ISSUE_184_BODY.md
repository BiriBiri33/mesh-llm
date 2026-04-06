## Summary

The current config work is still mixing two different problems:

1. local node startup configuration
2. mesh-wide desired state and editor reconciliation

This issue should stay focused on the first problem first.

The rollout should land explicit local `serve` / `client` commands together with GPU identity and inspection before expanding local config. Mesh-wide desired state can come later through editor-side reconciliation of per-node config rather than by making every node own a full-mesh TOML on disk.

## Problem

The main issue with the earlier `mesh.toml` direction is that it puts the authored file at the wrong level too early.

Operators mostly want to say:

- what should this node do on startup?
- should it serve or act as a client?
- what models should it load locally?
- what GPU should a model target once pinning exists?

They do not usually want every machine to carry a full mesh document describing every other machine.

That creates two problems immediately:

- the user-facing file is shaped around mesh-global state instead of local startup behavior
- it creates a distributed state problem before we have conflict-resolution, versioning, or reconciliation semantics

There is also a second issue: parts of the schema got ahead of the runtime surface. Things like `placement_mode` and `gpu_index` appeared before the CLI and runtime had a real local serve model that could own those concepts.

## Proposed rollout

### Phase 1: CLI migration and GPU inspection

Add explicit local entrypoints and a local GPU inspection surface:

```bash
mesh-llm serve ...
mesh-llm client ...
mesh-llm gpus
```

Goals:

- make serving and client behavior explicit
- keep top-level `--model` / `--gguf` / `--client` working temporarily as compatibility shims
- warn on legacy top-level usage and print the preferred replacement command
- reject invalid mixed legacy usage such as top-level `--client` plus serving flags
- expose friendly local GPU inspection backed by stable GPU identity
- keep the new CLI surfaces visually consistent with existing mesh-llm output by using emoji-prefixed status lines

Examples:

```bash
mesh-llm --auto --model Qwen3-8B-Q4_K_M
# warns, then behaves like:
mesh-llm serve --auto --model Qwen3-8B-Q4_K_M

mesh-llm --auto --client
# warns, then behaves like:
mesh-llm client --auto
```

Example warning text:

```text
⚠️ top-level serving flags now map to `mesh-llm serve`.
  Please use: mesh-llm serve ...
```

```text
⚠️ top-level `--client` now maps to `mesh-llm client`.
  Please use: mesh-llm client ...
```

CLI presentation direction for the new surfaces:

- `⚠️` warnings and migration notices
- `✅` successful setup / ready state
- `📡` client-mode status
- `🖥️` GPU headings in `mesh-llm gpus`
- `⛔` invalid combinations or fail-closed config errors

Examples:

```text
⚠️ top-level serving flags now map to `mesh-llm serve`.
  Please use: mesh-llm serve --auto --model Qwen3-8B-Q4_K_M
```

```text
📡 Client ready
  API:     http://localhost:9337
  Console: http://localhost:3131
```

Phase 1 also expands `GpuFacts` so `mesh-llm gpus` can show friendly details and a stable config-facing GPU identity:

- local index
- display name
- backend device name when applicable (`CUDA0`, `HIP0`, `MTL0`, etc.)
- VRAM
- bandwidth if benchmarked
- unified-memory marker
- stable ID
- platform-specific identity fields where available:
  - PCI BDF
  - vendor UUID
  - Metal registry ID
  - DXGI LUID
  - PnP instance ID

Example:

```text
🖥️ GPU 0
  Name: NVIDIA RTX 4090
  Stable ID: pci:0000:65:00.0
  Backend device: CUDA0
  VRAM: 24 GB
  Bandwidth: 1008 GB/s
```

This phase is intentionally local-only. No new config file shape is required yet.

### Phase 2: unified local config in `~/.mesh-llm/config.toml`

Keep config local-node only and reuse the existing local config path:

- default path: `~/.mesh-llm/config.toml`
- override: `--config /path/to/config.toml`

This file should become the local node config, not a mesh-global authored document.

Initial local config scope:

- `gpu.assignment = "auto"` only
- repeated startup models
- optional `mmproj` for GGUF multimodal models
- per-model `ctx_size`
- existing plugin config remains in the same file

Example:

```toml
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "apple"

[[models]]
model = "orange"
ctx_size = 8192

[[models]]
model = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/qwen2.5-vl-7b-instruct-q4_k_m.gguf"
mmproj = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-f16.gguf"
ctx_size = 8192

[[plugin]]
name = "blackboard"
enabled = true
```

Override rule:

- explicit CLI args override config completely for that concern
- explicit CLI models ignore configured models completely
- explicit CLI context size ignores configured context sizes completely

This is still local-node config, not mesh-global config.

### Phase 3: pinned GPU assignment

Once stable GPU identity is real and inspectable, add pinned placement to local config.

Example:

```toml
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
gpu_id = "pci:0000:65:00.0"
ctx_size = 8192
```

Pinned assignment should fail closed if the configured GPU identity no longer resolves.

### Phase 4: mesh-wide desired state via editor-side reconciliation

We do want mesh-wide desired state, but that does not mean every node should own a full-mesh TOML on disk.

A better model is:

- each node owns and serves its own local config
- nodes do not send full config on peer announce
- the console/editor explicitly asks each node for its current local config and GPU info
- the editor reconciles that into a full mesh view for editing
- applying changes writes updated per-node config back out to the nodes

That still gives us a real mesh-wide configuration editor, but the full-mesh state is reconstructed and reconciled in the editor flow instead of being a local authored file that every node is expected to keep in sync all the time.

## Design notes

### Config direction

The local config should stay user-facing and local-node oriented.

Prefer:

- `gpu.assignment = "auto"` as the explicit default
- later `gpu.assignment = "auto" | "pinned"`
- `[[models]].model = ...`
- optional `[[models]].mmproj = ...`

Avoid early internal vocabulary such as:

- `placement_mode = "pooled" | "separate"`
- `gpu_index` as long-lived identity

`gpu_index` can exist internally as a transient enumeration detail, but it should not be the primary stable config-facing GPU identity.

### Model reference shape

`[[models]].model` should intentionally accept the same broad model references that the resolver already supports:

- catalog name
- local path
- Hugging Face shorthand like `org/repo/file.gguf`
- full Hugging Face URL

`mmproj` should be optional and GGUF-oriented:

- explicit `mmproj` wins over auto-discovery
- omitted `mmproj` still allows current auto-discovery behavior
- non-GGUF backends should reject `mmproj` clearly when backend-aware local serving arrives

### `GpuFacts`

`GpuFacts` should become the local hardware backbone rather than keeping GPU state split across summarized name/count/VRAM fields.

This should power:

- `mesh-llm gpus`
- benchmark result attachment
- user-facing display
- later stable GPU matching for pinned config

That display should follow the same emoji-first CLI style rather than introducing a separate plain-text presentation style for the new surfaces.

## Compatibility

This is not intended to break existing users immediately.

Compatibility rules:

- existing top-level serve/client flows continue to work during the migration window
- they emit warnings pointing users to `mesh-llm serve` or `mesh-llm client`
- plugin-only `config.toml` files remain valid
- missing serve sections in `config.toml` mean “no configured startup models”

## Non-goals for the local config phases

- no mesh-global `[[nodes]]` file on every machine
- no distributed desired-state reconciliation in startup config
- no `gpu_index`-based long-lived identity
- no full-mesh authored shape before the local runtime surface is real
- no startup sections that do not map directly to `mesh-llm serve` / `mesh-llm client` behavior
