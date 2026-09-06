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

Live ComfyUI HTTP submission is intentionally deferred to v1.1.0.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
```

## Skill deployment

Keep this Git repository as the canonical source. Pi / Hermes should point to it via symlink rather than maintaining copies.
