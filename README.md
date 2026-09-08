# novel-to-short-drama

Data-driven production pipeline for adapting TXT / text-based PDF stories into structured short-drama shots and deterministic MiniMax H3 generation requests.

## v1.0.0 freezes

- `shot.yaml` as shot source of truth
- canonical asset / character / location registries
- deterministic reference selection and numbering
- versioned H3 prompt rendering
- deterministic seed policy
- seed-only retry semantics
- manual QC state evaluation
- simple episode concat planning

v1.1.0 adds flat API-workflow construction, asset staging, ComfyUI `/prompt` submission, `/history` polling, output video discovery, and `QC_PENDING` result records.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
```

## Skill deployment

Keep this Git repository as the canonical source. Pi / Hermes should point to it via symlink rather than maintaining copies.


## ComfyUI local runtime

Copy the versioned example to the ignored local profile and edit it for the machine. `profiles/comfyui_local.yaml` is intentionally ignored. Native Windows or mirrored networking can use `127.0.0.1`; WSL NAT should use the current Windows vEthernet (WSL) address.

`build-api-workflow` stages canonical assets into the configured ComfyUI input directory and writes the deterministic flat API-workflow artifact. `generate --workflow` executes that exact JSON artifact, queues `/prompt`, waits on `/history/{prompt_id}`, and resolves the saved video under the configured output directory.

```bash
cp profiles/comfyui_local.example.yaml profiles/comfyui_local.yaml
short-drama comfyui-check profiles/comfyui_local.yaml
short-drama build-api-workflow request.json --input-dir /path/to/ComfyUI/input -o workflow_api.json
short-drama generate request.json profiles/comfyui_local.yaml --workflow workflow_api.json --result result.json
```
