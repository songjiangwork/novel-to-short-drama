# Architecture

## Source-of-truth hierarchy
1. Semantic YAML: project / character / location / scene / shot
2. Derived runtime data: resolved refs / prompt / request / QC metadata
3. Generated media: shot videos / episode output

## v1.0.0 runtime boundary
v1.0.0 ends at a deterministic generation request. v1.1.0 adds the ComfyUI adapter.

The adapter may patch only: prompt, image refs, audio refs, seed, width, height, length, output prefix.

Model, VAE, CLIP, sampler, scheduler, LoRA state and reference policy remain profile-controlled.
