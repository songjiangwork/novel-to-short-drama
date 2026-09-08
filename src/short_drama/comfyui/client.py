from __future__ import annotations
import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

class ComfyUIError(RuntimeError):
    pass

class ComfyUIClient:
    def __init__(self, server_url: str, timeout_seconds: float = 30.0):
        self.server_url = server_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _json(self, method: str, path: str, payload: dict | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.server_url + path,
            data=data,
            method=method,
            headers={"Content-Type":"application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ComfyUIError(f"ComfyUI HTTP {exc.code} for {path}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ComfyUIError(f"Cannot reach ComfyUI at {self.server_url}: {exc.reason}") from exc
        return json.loads(raw) if raw else {}

    def object_info(self) -> dict[str, Any]:
        return self._json("GET", "/object_info")

    def check_nodes(self, required: set[str]) -> list[str]:
        info = self.object_info()
        return sorted(required.difference(info.keys()))

    def queue_prompt(self, workflow: dict[str, Any]) -> dict[str, Any]:
        payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
        response = self._json("POST", "/prompt", payload)
        if not response.get("prompt_id"):
            raise ComfyUIError(f"ComfyUI did not return prompt_id: {response}")
        node_errors = response.get("node_errors") or {}
        if node_errors:
            raise ComfyUIError(f"ComfyUI rejected workflow nodes: {node_errors}")
        return response

    def history(self, prompt_id: str) -> dict[str, Any]:
        return self._json("GET", f"/history/{prompt_id}")

    def wait_for_history(self, prompt_id: str, timeout_seconds: float, poll_interval_seconds: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            payload = self.history(prompt_id)
            entry = payload.get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                status_str = str(status.get("status_str", "")).lower()
                if status_str in {"error", "failed"}:
                    raise ComfyUIError(f"ComfyUI execution failed: {status}")
                if status.get("completed") is True or entry.get("outputs"):
                    return entry
            time.sleep(poll_interval_seconds)
        raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")
