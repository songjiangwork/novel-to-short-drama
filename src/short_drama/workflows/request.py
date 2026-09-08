from ..io import load_yaml, load_json
from ..seeds import deterministic_seed

def build_generation_request(shot_path, refs_path, project_id, attempt, prompt_text):
    if not 1 <= attempt <= 3:
        raise ValueError("attempt must be between 1 and 3")
    shot = load_yaml(shot_path)
    refs = load_json(refs_path)
    return {
        "schema_version": 1,
        "project_id": project_id,
        "shot_id": shot["shot_id"],
        "attempt": attempt,
        "profile": "h3_v1",
        "prompt_template_version": "h3_prompt_v1",
        "duration_seconds": shot["duration_seconds"],
        "seed": deterministic_seed(project_id, shot["shot_id"], attempt),
        "control_after_generate": "fixed",
        "references": refs,
        "prompt": prompt_text,
        "runtime_patch_whitelist": ["prompt","image_refs","audio_refs","seed","width","height","length","output_prefix"],
        "output_prefix": f"{shot['shot_id']}_attempt_{attempt:03d}",
        "adapter_status": "comfyui_adapter_v1",
    }
