"""Offline guards for the explicit A6G Alice acceptance script."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from short_drama.llm.config import RuntimeConfig
from short_drama.paths import REPO_ROOT


_SPEC = importlib.util.spec_from_file_location(
    "a6g_alice_acceptance", REPO_ROOT / "scripts" / "a6g_alice_acceptance.py"
)
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


class _Ref:
    def __init__(self, name: str):
        self.name = name

    def to_dict(self):
        return {"ref": self.name}


def _args(*, verification_only=True):
    return SimpleNamespace(
        project="project.yaml", runs_root="runs", consolidation_profile_id="consolidation-v1",
        story_analysis_profile="a6.yaml", runtime_config="runtime.yaml",
        llm_profile="semantic.yaml", verification_only=verification_only,
    )


def _runtime():
    return RuntimeConfig(
        schema_version=2, transport_id="test", base_url="http://127.0.0.1:8080/v1",
        request_model="test", provider_family="test", credential_environment_name=None,
        timeout_seconds=1,
    )


def _verified_context():
    alice = SimpleNamespace(
        canonical_id="char_0042", display_name_original="Alice Liddell",
        aliases_original=("Alice",),
    )
    interpretation = SimpleNamespace(text_zh="测试")
    chars = SimpleNamespace(analyses=(SimpleNamespace(
        character_ref="char_0042", role=interpretation, arc_summary=interpretation,
    ),))
    events = SimpleNamespace(windows=(SimpleNamespace(owned_event_refs=("evt_000001",)),))
    structure = SimpleNamespace(
        main_conflict=interpretation, main_plot=interpretation, ending_state=interpretation,
        major_turning_point_refs=(), major_reveal_refs=(),
    )
    bible = SimpleNamespace(
        unresolved_summary=interpretation, source_conflict_summary=interpretation,
        premise_zh=interpretation, themes=(),
    )
    snapshot = SimpleNamespace(
        canonical_characters=(alice,), unresolved_entities=(), conflicts=(),
        consolidation_manifest_ref=_Ref("a5"),
    )
    plan = SimpleNamespace(event_stream=("evt_000001",))
    publication = SimpleNamespace(
        story_analysis_manifest_ref=_Ref("a6"), current_pointer_ref=_Ref("pointer"),
    )
    report = SimpleNamespace(summary=SimpleNamespace(to_dict=lambda: {"result": "PASS"}))
    return (Path("project.yaml"), {"project_id": "a3e-real-novel"}, SimpleNamespace(),
            snapshot, plan, publication, report, chars, events, SimpleNamespace(), structure, bible)


def test_verification_only_uses_no_provider_and_no_writes(monkeypatch):
    seen = []
    context = _verified_context()
    monkeypatch.setattr(gate, "_load_verified_current", lambda args: context)
    monkeypatch.setattr(gate, "load_runtime_config", lambda path: _runtime())
    monkeypatch.setattr(gate, "_files", lambda root: {"unchanged": "hash"})
    monkeypatch.setattr(gate, "OpenAICompatibleLLMClient", lambda runtime: pytest.fail("must not construct provider client"))

    def reuse(project, **kwargs):
        seen.append(kwargs["llm_client"])
        return SimpleNamespace(reused=True, a6_manifest_ref=context[5].story_analysis_manifest_ref)

    monkeypatch.setattr(gate, "build_story_bible_project", reuse)
    result = gate.run(_args())

    assert result["mode"] == "verification-only"
    assert result["fresh"] == {"status": "NOT_RUN"}
    assert result["exact_rerun"] == {"reused": True, "provider_calls": 0, "writes": 0, "same_manifest": True}
    assert result["runtime_only_reuse"] == {"reused": True, "provider_calls": 0, "writes": 0}
    assert all(isinstance(client, gate.NoProviderLLMClient) and client.calls == 0 for client in seen)
    assert result["structural"]["reference_validation"]["status"] == "PASS"
    assert "invalid_canonical_or_a6_refs" not in result["structural"]
    assert result["manual_review_packet"]["status"] == "PENDING_MANUAL_REVIEW"
    assert result["manual_review_packet"]["alice_dossier"]["canonical_character_ref"] == "char_0042"


def test_fresh_mode_still_rejects_existing_current(monkeypatch):
    pointers = SimpleNamespace(resolve_current=lambda pointer_id: object())
    monkeypatch.setattr(gate, "_load_project", lambda path: (Path(path), {"project_id": "a3e-real-novel"}))
    monkeypatch.setattr(gate, "load_story_analysis_profile", lambda path: SimpleNamespace(profile_id="global-story-analysis-v1"))
    monkeypatch.setattr(gate, "_stores", lambda runs_root, project_id: (object(), pointers))
    monkeypatch.setattr(gate, "build_story_bible_project", lambda *args, **kwargs: pytest.fail("fresh execution must not start"))

    with pytest.raises(gate.AcceptanceError, match="already exists"):
        gate.run(_args(verification_only=False))


def test_alice_identity_without_single_canonical_match_remains_pending():
    snapshot = SimpleNamespace(canonical_characters=())
    chars = SimpleNamespace(analyses=())
    assert gate._alice_dossier(snapshot, chars) == {
        "status": "PENDING_CANONICAL_IDENTITY_CONFIRMATION",
        "matching_canonical_character_count": 0,
    }
