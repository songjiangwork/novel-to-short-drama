from __future__ import annotations
from datetime import datetime, timezone
from typing import Any

from ..assets.staging import host_path
from ..io import dump_json, load_yaml
from .api_workflow import REQUIRED_NODE_CLASSES
from .client import ComfyUIClient
from .media import extract_video_refs, resolve_local_output


def generate_with_comfyui(
    request: dict[str, Any],
    runtime_config_path: str,
    workflow: dict[str, Any],
    result_path: str | None = None,
) -> dict[str, Any]:
    """Execute the exact API-workflow artifact supplied by the caller."""
    cfg = load_yaml(runtime_config_path)
    client = ComfyUIClient(cfg["server_url"])
    missing = client.check_nodes(REQUIRED_NODE_CLASSES)
    if missing:
        raise RuntimeError(
            f"ComfyUI is missing required node classes: {', '.join(missing)}"
        )

    queued_at = datetime.now(timezone.utc).isoformat()
    queued = client.queue_prompt(workflow)
    prompt_id = queued["prompt_id"]
    history = client.wait_for_history(
        prompt_id,
        float(cfg.get("timeout_seconds", 7200)),
        float(cfg.get("poll_interval_seconds", 2.0)),
    )
    media = extract_video_refs(history)
    if not media:
        raise RuntimeError(
            f"ComfyUI completed prompt {prompt_id} but no saved video was found in history"
        )

    output_root = host_path(cfg["output_dir"])
    outputs = []
    for ref in media:
        local_path = resolve_local_output(ref, output_root)
        outputs.append({
            **ref,
            "local_path": str(local_path),
            "exists": local_path.is_file(),
        })

    if not any(item["exists"] for item in outputs):
        paths = ", ".join(item["local_path"] for item in outputs)
        raise RuntimeError(
            f"ComfyUI completed prompt {prompt_id}, but no saved video is locally accessible: {paths}"
        )

    result = {
        "schema_version": 1,
        "project_id": request["project_id"],
        "shot_id": request["shot_id"],
        "attempt": request["attempt"],
        "prompt_id": prompt_id,
        "state": "QC_PENDING",
        "queued_at": queued_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs,
    }
    if result_path:
        dump_json(result, result_path)
    return result
