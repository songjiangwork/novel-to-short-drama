# Architecture

## Source-of-truth hierarchy
1. Semantic YAML: project / character / location / scene / shot
2. Derived runtime data: resolved refs / prompt / request / QC metadata
3. Generated media: shot videos / episode output

## v1.0.0 runtime boundary
v1.0.0 ends at a deterministic generation request. v1.1.0 adds the ComfyUI adapter.

The adapter may patch only: prompt, image refs, audio refs, seed, width, height, length, output prefix.

Diffusion, CLIP and VAE model filenames, sampler, scheduler and reference policy remain profile-controlled. v1.1 supports the frozen LoRA-disabled workflow only; `lora.turbo_lightning: true` is rejected until an explicit LoRA workflow is implemented.


## H3 API dynamic-input rule

`MiniMaxH3ReferenceToVideo` autogrow inputs are serialized as flat dotted keys, for example `ref_images.ref_image_0` and `ref_audios.ref_audio_0`. The adapter must not emit nested `ref_images` / `ref_audios` objects.


## Execution artifact rule

`build-api-workflow` stages canonical assets and writes the deterministic API-workflow artifact. `generate --workflow` must execute that exact JSON artifact; it must not silently rebuild or overwrite it.
