from __future__ import annotations

import json
import pytest
from pathlib import Path

from short_drama.comfyui.api_workflow import build_api_workflow
from short_drama.comfyui.dimensions import h3_length_for_seconds, resolution_from_profile
from short_drama.comfyui.media import extract_video_refs, resolve_local_output
from short_drama.assets.staging import host_path, stage_references
from short_drama.prompting.builder import build_prompt
from short_drama.assets.selector import resolve_references
from short_drama.workflows.request import build_generation_request

FIX = Path(__file__).parent / "fixtures"


def _request(tmp_path):
    refs = resolve_references(
        str(FIX / "shot.yaml"),
        str(FIX / "assets.yaml"),
        str(FIX / "characters"),
        str(FIX / "locations"),
    )
    for ref in refs["images"] + refs["audios"]:
        ref["runtime_filename"] = f"short_drama/test/{ref['asset_id'].replace('.', '__')}.dat"
    refs_path = tmp_path / "refs.json"
    refs_path.write_text(json.dumps(refs, ensure_ascii=False), encoding="utf-8")
    prompt = build_prompt(str(FIX / "shot.yaml"), str(refs_path))
    return build_generation_request(
        str(FIX / "shot.yaml"), str(refs_path), "classroom-demo", 1, prompt
    )


def test_resolution_matches_comfyui_core_baseline():
    assert resolution_from_profile(0.4, "16:9", 32) == (864, 480)


def test_h3_five_second_length_matches_baseline():
    assert h3_length_for_seconds(5, 24) == 124


def test_api_workflow_uses_flat_dynamic_reference_keys(tmp_path):
    workflow = build_api_workflow(_request(tmp_path))
    h3 = workflow["136"]["inputs"]
    assert h3["width"] == 864
    assert h3["height"] == 480
    assert h3["length"] == 124
    assert h3["ref_image_size"] == "match"
    assert "ref_images" not in h3
    assert "ref_audios" not in h3
    assert [f"ref_images.ref_image_{i}" in h3 for i in range(7)] == [True] * 7
    assert h3["ref_audios.ref_audio_0"] == ["301", 0]
    assert h3["ref_audios.ref_audio_1"] == ["302", 0]


def test_api_workflow_preserves_frozen_sampling_profile(tmp_path):
    request = _request(tmp_path)
    workflow = build_api_workflow(request)
    assert workflow["129"]["inputs"]["noise_seed"] == request["seed"]
    assert workflow["123"]["inputs"]["sampler_name"] == "res_multistep"
    assert workflow["124"]["inputs"]["scheduler"] == "simple"
    assert workflow["124"]["inputs"]["steps"] == 20
    assert workflow["92"]["inputs"]["format"] == "auto"
    assert workflow["92"]["inputs"]["codec"] == "auto"


def test_history_video_extraction_is_key_name_agnostic():
    history = {
        "outputs": {
            "92": {
                "anything": [
                    {"filename": "E01_00001_.mp4", "subfolder": "video/short_drama", "type": "output"}
                ]
            }
        }
    }
    assert extract_video_refs(history) == [
        {"filename": "E01_00001_.mp4", "subfolder": "video/short_drama", "type": "output"}
    ]


def test_windows_history_subfolder_resolves_under_wsl(tmp_path):
    output_root = tmp_path / "output"
    expected = (
        output_root
        / "video"
        / "short_drama"
        / "classroom-demo"
        / "E01_S003_SH002_attempt_001_00001_.mp4"
    )
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"video")

    resolved = resolve_local_output(
        {
            "filename": "E01_S003_SH002_attempt_001_00001_.mp4",
            "subfolder": r"video\short_drama\classroom-demo",
            "type": "output",
        },
        output_root,
    )
    assert resolved == expected
    assert resolved.is_file()


def test_windows_canonical_path_maps_to_wsl():
    p = host_path(r"C:\Users\jiang\Downloads\Song_Canonical\Song_Face_100.jpg")
    if str(p).startswith("/mnt/"):
        assert str(p) == "/mnt/c/Users/jiang/Downloads/Song_Canonical/Song_Face_100.jpg"


def test_comfyui_client_protocol_roundtrip():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from short_drama.comfyui.client import ComfyUIClient
    from short_drama.comfyui.api_workflow import REQUIRED_NODE_CLASSES

    prompt_id = "test-prompt-id"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, payload, code=200):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/object_info":
                self._send({name: {} for name in REQUIRED_NODE_CLASSES})
            elif self.path == f"/history/{prompt_id}":
                self._send({prompt_id: {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {"92": {"videos": [{
                        "filename": "test_00001_.mp4",
                        "subfolder": "video/short_drama",
                        "type": "output",
                    }]}}
                }})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/prompt":
                self._send({"error": "not found"}, 404); return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            assert "prompt" in body
            assert "client_id" in body
            self._send({"prompt_id": prompt_id, "number": 0, "node_errors": {}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ComfyUIClient(f"http://127.0.0.1:{server.server_port}")
        assert client.check_nodes(REQUIRED_NODE_CLASSES) == []
        queued = client.queue_prompt({"1": {"class_type": "X", "inputs": {}}})
        assert queued["prompt_id"] == prompt_id
        history = client.wait_for_history(prompt_id, timeout_seconds=1, poll_interval_seconds=0.01)
        assert history["status"]["completed"] is True
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=1)


def test_stage_references_copies_assets_and_sets_runtime_names(tmp_path):
    request = _request(tmp_path)
    refs = request["references"]
    for ref in refs["images"] + refs["audios"]:
        suffix = ".wav" if ref["role"] == "voice" else ".jpg"
        source = tmp_path / f"{ref['asset_id'].replace('.', '_')}{suffix}"
        source.write_bytes(b"test")
        ref["source_path"] = str(source)
        ref.pop("runtime_filename", None)

    input_dir = tmp_path / "comfy-input"
    staged = stage_references(refs, input_dir, "classroom-demo", "E01_S003_SH002")
    first = staged["images"][0]
    assert first["runtime_filename"].startswith(
        "short_drama/classroom-demo/E01_S003_SH002/assets/"
    )
    assert (input_dir / first["runtime_filename"]).is_file()


def test_adapter_executes_exact_workflow_and_returns_qc_pending(tmp_path, monkeypatch):
    import yaml
    import short_drama.comfyui.adapter as adapter_module

    request = _request(tmp_path)
    workflow = build_api_workflow(request)
    output_dir = tmp_path / "comfy-output"
    saved = output_dir / "video" / "short_drama" / "classroom-demo" / "result_00001_.mp4"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"video")

    cfg = tmp_path / "runtime.yaml"
    cfg.write_text(yaml.safe_dump({
        "server_url": "http://fake",
        "input_dir": str(tmp_path / "unused-input"),
        "output_dir": str(output_dir),
        "poll_interval_seconds": 0.01,
        "timeout_seconds": 1,
    }), encoding="utf-8")

    captured = {}
    class FakeClient:
        def __init__(self, server_url): captured["server_url"] = server_url
        def check_nodes(self, required): return []
        def queue_prompt(self, queued_workflow):
            captured["workflow"] = queued_workflow
            return {"prompt_id": "pid", "node_errors": {}}
        def wait_for_history(self, prompt_id, timeout, interval):
            return {"status": {"completed": True}, "outputs": {"92": {"videos": [{
                "filename": saved.name,
                "subfolder": r"video\short_drama\classroom-demo",
                "type": "output",
            }]}}}

    monkeypatch.setattr(adapter_module, "ComfyUIClient", FakeClient)
    result = adapter_module.generate_with_comfyui(request, str(cfg), workflow)
    assert captured["workflow"] is workflow
    assert result["state"] == "QC_PENDING"
    assert result["prompt_id"] == "pid"
    assert result["outputs"][0]["exists"] is True


def test_adapter_fails_closed_when_saved_video_is_not_local(tmp_path, monkeypatch):
    import yaml
    import short_drama.comfyui.adapter as adapter_module

    request = _request(tmp_path)
    workflow = build_api_workflow(request)
    cfg = tmp_path / "runtime.yaml"
    cfg.write_text(yaml.safe_dump({
        "server_url": "http://fake",
        "output_dir": str(tmp_path / "comfy-output"),
        "poll_interval_seconds": 0.01,
        "timeout_seconds": 1,
    }), encoding="utf-8")

    class FakeClient:
        def __init__(self, server_url): pass
        def check_nodes(self, required): return []
        def queue_prompt(self, queued_workflow): return {"prompt_id": "pid", "node_errors": {}}
        def wait_for_history(self, prompt_id, timeout, interval):
            return {"status": {"completed": True}, "outputs": {"92": {"videos": [{
                "filename": "missing.mp4",
                "subfolder": r"video\short_drama\classroom-demo",
                "type": "output",
            }]}}}

    monkeypatch.setattr(adapter_module, "ComfyUIClient", FakeClient)
    with pytest.raises(RuntimeError, match="no saved video is locally accessible"):
        adapter_module.generate_with_comfyui(request, str(cfg), workflow)


def test_api_workflow_model_filenames_come_from_profile(tmp_path):
    workflow = build_api_workflow(_request(tmp_path))
    assert workflow["127"]["inputs"]["unet_name"] == "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    assert workflow["128"]["inputs"]["clip_name"] == "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    assert workflow["119"]["inputs"]["vae_name"] == "minimax_h3_video_vae_fp16.safetensors"
    assert workflow["120"]["inputs"]["vae_name"] == "minimax_h3_audio_vae_fp32.safetensors"


def test_api_workflow_rejects_enabled_turbo_lora(tmp_path, monkeypatch):
    import short_drama.comfyui.api_workflow as workflow_module

    request = _request(tmp_path)
    profile = workflow_module._profile(request)
    profile["lora"]["turbo_lightning"] = True
    monkeypatch.setattr(workflow_module, "_profile", lambda _: profile)

    with pytest.raises(ValueError, match="LoRA-disabled"):
        workflow_module.build_api_workflow(request)
