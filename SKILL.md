# novel-to-short-drama

## Purpose
Convert TXT or text-based PDF stories into structured, reviewable short-drama production data and deterministic MiniMax H3 R2V requests.

`shot.yaml` is the source of truth. Prompts, reference numbering, seeds, runtime requests, QC records, and retries are derived artifacts.

## Hard rules
1. Never generate canonical character or scene images unless explicitly requested by the user.
2. Never treat an H3 prompt as primary story data.
3. Never let the LLM assign `<Picture N>` / `<Audio N>` indices.
4. Missing important canonical camera coverage blocks production; do not ask H3 to invent it.
5. v1 supports at most 2 named characters and 2 speakers per shot.
6. Dialogue shots default to locked camera.
7. Attempts 1-3 are seed-only retries.
8. After 3 failed attempts stop at `MANUAL_INTERVENTION`.
9. Adaptation and generation preflight require human approval.
10. Manual QC is authoritative in v1.

## Frozen H3 baseline
- diffusion: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- 5 s, 0.4 MP, 16:9, 24 fps
- 20 steps, `res_multistep`, `simple`
- `ref_image_size=match`
- Turbo/Lightning off
- `control_after_generate=fixed`

## Character reference policy
- closeup: front + 3/4
- medium: front + 3/4 + full body
- full_body: front + 3/4 + full body

Two-character medium baseline = 3 + 3 + 1 scene = 7 image refs.

## Deterministic ordering
Character order follows `shot.characters`. Within each character: front -> 3/4 -> full body. Scene follows all character refs. Audio refs follow first speaking order.

## Retry invariant
Retries preserve shot, refs, rendered prompt, prompt-template version, profile, and settings. Only seed and output prefix change unless a human edits source data.

## Role split
Pi/Qwen: story understanding, adaptation, scene decomposition, shot planning, action/dialogue writing.

Deterministic Python: validation, registry lookup, reference selection/numbering, prompt rendering, seeds, request construction, QC transitions, assembly plan.

Hermes: orchestration, approval gates, Pi delegation, deterministic command invocation, retries, episode progression.


## ComfyUI Adapter v1.1

The runtime adapter must use ComfyUI API format, never UI workflow format. H3 autogrow inputs are emitted as flat dotted keys (`ref_images.ref_image_0`, `ref_audios.ref_audio_0`). Asset files are staged into ComfyUI input before queueing. The adapter must validate required node classes, POST `/prompt`, poll `/history/{prompt_id}`, record the saved video reference, and transition the attempt to `QC_PENDING`.
