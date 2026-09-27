"""A5G3 real-novel acceptance; this is reporting instrumentation, not A5 logic."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from short_drama.foundation import ValidationResult, load_validation_report
from short_drama.llm import LLMClient, LLMRetryExhaustedError, OpenAICompatibleLLMClient, load_runtime_config, load_semantic_profile
from short_drama.paths import REPO_ROOT
from short_drama.story import ConsolidationCurrentMissingError, ConsolidationSemanticGenerationError, consolidate_evidence_project
from short_drama.story.consolidation_persistence import (
    ConsolidationPersistenceService, a5_base_artifact_id, canonical_event_set_artifact_id,
    canonical_fact_set_artifact_id, canonical_relationship_set_artifact_id,
    consolidation_candidate_index_artifact_id, load_canonical_event_set,
    load_canonical_fact_set, load_canonical_relationship_set,
    load_consolidation_candidate_index, load_story_conflict_set,
    story_conflict_set_artifact_id,
)
from short_drama.story.reconciliation_persistence import (
    ReconciliationPersistenceService, load_canonical_character_registry,
    load_canonical_location_registry, load_unresolved_entity_set,
)
from short_drama.story.service import DOCUMENT_ID, _load_project, _stores

PROJECT = REPO_ROOT / "examples" / "a3e_real_novel" / "project.yaml"
RUNS_ROOT = REPO_ROOT / "runs" / "a4e_real_novel"
CONSOLIDATION_PROFILE = REPO_ROOT / "profiles" / "consolidation_v1.yaml"
RUNTIME_CONFIG = REPO_ROOT / "profiles" / "llm_local.yaml"
SEMANTIC_PROFILE = REPO_ROOT / "profiles" / "consolidation_llm_v1.yaml"
RECONCILIATION_PROFILE_ID = "entity-reconciliation-v2"
EXPECTED_PROJECT_ID = "a3e-real-novel"


class AcceptanceError(RuntimeError):
    pass


class CountingLLMClient(LLMClient):
    """Count semantic calls separately from transport attempts."""
    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.semantic_generation_calls = 0
        self.provider_attempts = 0

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
        self.semantic_generation_calls += 1
        try:
            result = self._inner.generate_structured(rendered_prompt, output_schema, semantic_profile)
        except LLMRetryExhaustedError as exc:
            self.provider_attempts += exc.attempts
            raise
        self.provider_attempts += result.attempts
        return result


def _ref(ref: Any) -> dict[str, Any]:
    return ref.to_dict() if hasattr(ref, "to_dict") else dict(ref)


def _evidence_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return tuple(item[key] for key in ("paragraph_id", "role", "strength", "excerpt"))  # type: ignore[return-value]


def require_upstream_identity(a4_current: Any, *, project_id: str, reconciliation_profile_id: str) -> dict[str, Any]:
    """Return the exact A4-pinned A3 identity, or fail before provider use."""
    if project_id != EXPECTED_PROJECT_ID or DOCUMENT_ID != "src_001":
        raise AcceptanceError("A5G3 requires the fixed Alice project/document identity")
    entity_map = a4_current.entity_map
    a3 = entity_map.a3_input
    refs = tuple(a3.candidate_extraction_refs)
    if len(refs) != 12:
        raise AcceptanceError(f"Alice A4 EntityMap pins {len(refs)} A3 refs, expected 12")
    return {
        "project_id": project_id,
        "document_id": DOCUMENT_ID,
        "reconciliation_profile_id": reconciliation_profile_id,
        "entity_map_ref": _ref(a4_current.entity_map_ref),
        "validation_report_ref": _ref(a4_current.validation_report_ref),
        "current_pointer_ref": _ref(a4_current.current_pointer_ref),
        "source_document_ref": _ref(a3.source_document_ref),
        "chunk_manifest_ref": _ref(a3.chunk_manifest_ref),
        "candidate_extraction_refs": [_ref(ref) for ref in refs],
    }


def _bound_refs(index: dict[str, Any], facts: dict[str, Any], events: dict[str, Any], relationships: dict[str, Any]) -> tuple[list[str], list[str]]:
    indexed: list[str] = []
    for item in index["facts"]:
        indexed.extend(item["subject_refs"] + item["object_refs"])
    for item in index["events"]:
        indexed.extend(item["participants"] + item["locations"])
    for item in index["relationships"]:
        indexed.extend((item["source_entity_ref"], item["target_entity_ref"]))
    canonical: list[str] = []
    for item in facts["facts"]:
        canonical.extend(item["subject_refs"] + item["object_refs"])
    for item in facts["state_transitions"]:
        canonical.extend(item["subject_refs"])
    for item in events["events"]:
        canonical.extend(item["participants"] + item["locations"])
    for item in relationships["relationships"]:
        canonical.extend((item["source_entity_ref"], item["target_entity_ref"]))
    return indexed, canonical


def audit_persisted_bundle(*, index: dict[str, Any], facts: dict[str, Any], events: dict[str, Any], relationships: dict[str, Any], conflicts: dict[str, Any], allowed_entity_ids: set[str], unresolved_ids: set[str]) -> dict[str, Any]:
    """Read-only issue-56 audit over typed persisted payload projections."""
    expected = {"fact": {x["global_candidate_ref"] for x in index["facts"]}, "event": {x["global_candidate_ref"] for x in index["events"]}, "relationship": {x["global_candidate_ref"] for x in index["relationships"]}}
    actual_lists = {
        "fact": [ref for item in facts["facts"] for ref in item["candidate_fact_refs"]],
        "event": [ref for item in events["events"] for ref in item["candidate_event_refs"]],
        "relationship": [ref for item in relationships["relationships"] for ref in item["candidate_relationship_refs"]],
    }
    actual = {domain: set(refs) for domain, refs in actual_lists.items()}
    failures: list[str] = []
    accounting = {}
    for domain in expected:
        missing, extra = expected[domain] - actual[domain], actual[domain] - expected[domain]
        accounting[domain] = {"accounted": len(expected[domain] - missing), "total": len(expected[domain]), "missing": len(missing), "extra": len(extra)}
        if missing or extra or len(actual_lists[domain]) != len(actual[domain]):
            failures.append(f"{domain} candidate accounting missing={len(missing)} extra={len(extra)}")
    indexed_bound, canonical_bound = _bound_refs(index, facts, events, relationships)
    indexed_unresolved = {x for x in indexed_bound if x.startswith("unres_")}
    canonical_unresolved = {x for x in canonical_bound if x.startswith("unres_")}
    dangling = {x for x in canonical_bound if x.startswith(("char_", "loc_", "unres_")) and x not in allowed_entity_ids}
    if dangling:
        failures.append(f"dangling bound refs={sorted(dangling)}")
    if not canonical_unresolved <= unresolved_ids:
        failures.append("canonical unresolved ids are not in exact A4 UnresolvedEntitySet")
    canonical_ids = {
        "fact_id": [x["fact_id"] for x in facts["facts"]],
        "transition_id": [x["transition_id"] for x in facts["state_transitions"]],
        "event_id": [x["event_id"] for x in events["events"]],
        "relationship_id": [x["relationship_id"] for x in relationships["relationships"]],
        "conflict_id": [x["conflict_id"] for x in conflicts["conflicts"]],
    }
    duplicate_namespaces = [
        namespace for namespace, ids in canonical_ids.items()
        if len(ids) != len(set(ids))
    ]
    fact_ids = canonical_ids["fact_id"]
    relationship_ids = canonical_ids["relationship_id"]
    dangling_facts = {ref for item in facts["state_transitions"] for ref in (item["from_fact_id"], item["to_fact_id"])} - set(fact_ids)
    dangling_conflict_facts = {ref for item in conflicts["conflicts"] for ref in item["fact_ids"]} - set(fact_ids)
    dangling_relationships = {ref for item in conflicts["conflicts"] for ref in item["relationship_ids"]} - set(relationship_ids)
    if duplicate_namespaces or dangling_facts or dangling_conflict_facts or dangling_relationships:
        failures.append("canonical graph consistency failure")
    tuples = [x["evidence_refs"] for x in facts["facts"]] + [x["evidence_refs"] for x in facts["state_transitions"]] + [x["evidence_refs"] for x in events["events"]] + [x["evidence_refs"] for x in conflicts["conflicts"]] + [s["evidence_refs"] for x in relationships["relationships"] for s in x["state_history"]]
    duplicate_evidence = sum(len(t) != len({_evidence_key(x) for x in t}) for t in tuples)
    if duplicate_evidence:
        failures.append(f"duplicate exact EvidenceRefs within tuples={duplicate_evidence}")
    if failures:
        raise AcceptanceError("; ".join(failures))
    return {"candidate_accounting": accounting, "unresolved": {"indexed_occurrences": sum(x.startswith("unres_") for x in indexed_bound), "indexed_unique_ids": len(indexed_unresolved), "canonical_unique_ids_used": len(canonical_unresolved), "dangling_unresolved_ids": 0}, "dangling_bound_refs": 0, "graph": {"duplicate_canonical_ids": 0, "dangling_fact_refs": 0, "dangling_relationship_refs": 0, "contradictions": 0}, "evidence": {"tuples_audited": len(tuples), "duplicate_exact_within_tuples": 0}}


def publication_refs(result: Any) -> dict[str, dict[str, Any]]:
    return {name: _ref(getattr(result, name)) for name in ("consolidation_candidate_index_ref", "consolidation_decision_set_ref", "canonical_fact_set_ref", "canonical_event_set_ref", "canonical_relationship_set_ref", "story_conflict_set_ref", "consolidation_manifest_ref", "validation_report_ref", "current_pointer_ref")}


def snapshot_a5_files(runs_root: str | Path, project_id: str) -> dict[str, str]:
    root = Path(runs_root) / project_id / "story"
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*")) if path.is_file() and ".a5." in str(path)}


def check_reuse_gate(first: Any, rerun: Any, *, semantic_delta: int, attempt_delta: int, before: dict[str, str], after: dict[str, str]) -> None:
    if not rerun.reused or rerun.semantic_generation_call_count != 0 or semantic_delta or attempt_delta or publication_refs(first) != publication_refs(rerun) or before != after:
        raise AcceptanceError("exact rerun did not prove zero-provider immutable reuse")


def check_invalidation_gate(*, old_hash: str, new_hash: str, result: Any, semantic_calls: int, provider_attempts: int, old_manifest: dict[str, Any], temp_current: Any) -> None:
    if old_hash == new_hash or result.reused or result.semantic_generation_call_count < 1 or semantic_calls < 1 or provider_attempts < 1 or _ref(result.consolidation_manifest_ref) == old_manifest or temp_current.manifest_ref != result.consolidation_manifest_ref:
        raise AcceptanceError("semantic identity invalidation gate failed")


def _semantic_domain(block_id: str) -> str:
    if block_id.startswith("a5fblk_"):
        return "fact"
    if block_id.startswith("a5eblk_"):
        return "event"
    if block_id.startswith("a5rblk_"):
        return "relationship"
    return "unknown"


def live_failure_payload(exc: Exception, client: CountingLLMClient | None) -> dict[str, Any]:
    """Machine-readable diagnostics from existing exceptions only."""
    payload: dict[str, Any] = {
        "valid": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if client is not None:
        payload["semantic_generation_calls_observed"] = client.semantic_generation_calls
        payload["provider_attempts_observed"] = client.provider_attempts
    if isinstance(exc, LLMRetryExhaustedError):
        payload["attempts"] = exc.attempts
    if isinstance(exc, ConsolidationSemanticGenerationError):
        payload.update({
            "domain": _semantic_domain(exc.block_id),
            "block_id": exc.block_id,
            "rounds_attempted": exc.rounds_attempted,
            "pair_count": len(exc.expected_pairs),
            "last_failure_details": exc.last_failure_details,
        })
    return payload


def _persisted_audit(store: Any, entity_map: Any, result: Any, profile_id: str) -> dict[str, Any]:
    base = a5_base_artifact_id(EXPECTED_PROJECT_ID, DOCUMENT_ID, profile_id)
    index = load_consolidation_candidate_index(store, result.consolidation_candidate_index_ref, expected_artifact_id=consolidation_candidate_index_artifact_id(base)).to_dict()
    facts = load_canonical_fact_set(store, result.canonical_fact_set_ref, expected_artifact_id=canonical_fact_set_artifact_id(base)).to_dict()
    events = load_canonical_event_set(store, result.canonical_event_set_ref, expected_artifact_id=canonical_event_set_artifact_id(base)).to_dict()
    relationships = load_canonical_relationship_set(store, result.canonical_relationship_set_ref, expected_artifact_id=canonical_relationship_set_artifact_id(base)).to_dict()
    conflicts = load_story_conflict_set(store, result.story_conflict_set_ref, expected_artifact_id=story_conflict_set_artifact_id(base)).to_dict()
    chars = load_canonical_character_registry(store, entity_map.canonical_character_registry_ref, expected_artifact_id=entity_map.canonical_character_registry_ref.artifact_id)
    locations = load_canonical_location_registry(store, entity_map.canonical_location_registry_ref, expected_artifact_id=entity_map.canonical_location_registry_ref.artifact_id)
    unresolved = load_unresolved_entity_set(store, entity_map.unresolved_entity_set_ref, expected_artifact_id=entity_map.unresolved_entity_set_ref.artifact_id)
    allowed = {x.canonical_id for x in chars.entities} | {x.canonical_id for x in locations.entities} | {x.unresolved_id for x in unresolved.entities}
    return audit_persisted_bundle(index=index, facts=facts, events=events, relationships=relationships, conflicts=conflicts, allowed_entity_ids=allowed, unresolved_ids={x.unresolved_id for x in unresolved.entities})


def run_acceptance(
    args: argparse.Namespace,
    *,
    failure_observer: dict[str, CountingLLMClient] | None = None,
) -> dict[str, Any]:
    project_file, project = _load_project(args.project)
    project_id = project["project_id"]
    if project_id != EXPECTED_PROJECT_ID:
        raise AcceptanceError(f"expected project_id {EXPECTED_PROJECT_ID!r}, got {project_id!r}")
    store, pointers = _stores(args.runs_root, project_id)
    a4 = ReconciliationPersistenceService(store, pointers).require_current_validated(project_id=project_id, document_id=DOCUMENT_ID, reconciliation_profile_id=args.reconciliation_profile_id)
    upstream = require_upstream_identity(a4, project_id=project_id, reconciliation_profile_id=args.reconciliation_profile_id)
    from short_drama.story import load_consolidation_profile
    profile = load_consolidation_profile(Path(args.consolidation_profile))
    a5 = ConsolidationPersistenceService(store, pointers)
    preexisting_a5_files = snapshot_a5_files(args.runs_root, project_id)
    try:
        existing = a5.require_current_validated(project_id=project_id, document_id=DOCUMENT_ID, consolidation_profile_id=profile.profile_id)
    except ConsolidationCurrentMissingError:
        existing = None
    if existing is not None:
        raise AcceptanceError(f"A5 CURRENT already exists; not fresh: {_ref(existing.manifest_ref)}")
    runtime = load_runtime_config(args.runtime_config)
    client = CountingLLMClient(OpenAICompatibleLLMClient(runtime))
    if failure_observer is not None:
        failure_observer["client"] = client
    started = time.monotonic()
    fresh = consolidate_evidence_project(project_file, runs_root=args.runs_root, reconciliation_profile_id=args.reconciliation_profile_id, consolidation_profile_path=args.consolidation_profile, semantic_profile_path=args.llm_profile, llm_client=client)
    fresh_elapsed = time.monotonic() - started
    if fresh.reused or fresh.semantic_generation_call_count < 1 or client.semantic_generation_calls != fresh.semantic_generation_call_count or fresh.entity_map_ref != a4.entity_map_ref or fresh.candidate_extraction_refs != a4.entity_map.a3_input.candidate_extraction_refs:
        raise AcceptanceError("fresh production result failed identity/provider gates")
    validated = a5.require_current_validated(project_id=project_id, document_id=DOCUMENT_ID, consolidation_profile_id=profile.profile_id)
    if validated.manifest_ref != fresh.consolidation_manifest_ref or validated.validation_report_ref != fresh.validation_report_ref or validated.current_pointer_ref != fresh.current_pointer_ref or load_validation_report(store, validated.validation_report_ref).summary.result is not ValidationResult.PASS:
        raise AcceptanceError("fresh A5 CURRENT validation gate failed")
    audit = _persisted_audit(store, a4.entity_map, fresh, profile.profile_id)
    before = snapshot_a5_files(args.runs_root, project_id)
    sem_before, attempts_before, started = client.semantic_generation_calls, client.provider_attempts, time.monotonic()
    rerun = consolidate_evidence_project(project_file, runs_root=args.runs_root, reconciliation_profile_id=args.reconciliation_profile_id, consolidation_profile_path=args.consolidation_profile, semantic_profile_path=args.llm_profile, llm_client=client)
    rerun_elapsed = time.monotonic() - started
    after = snapshot_a5_files(args.runs_root, project_id)
    check_reuse_gate(fresh, rerun, semantic_delta=client.semantic_generation_calls-sem_before, attempt_delta=client.provider_attempts-attempts_before, before=before, after=after)
    changed_runtime = dataclasses.replace(runtime, transport_id="a5g3-unreachable", base_url="http://127.0.0.1:9/v1", request_model="a5g3-unreachable", provider_family="acceptance", timeout_seconds=1)
    transport_client = CountingLLMClient(OpenAICompatibleLLMClient(changed_runtime))
    before_transport = snapshot_a5_files(args.runs_root, project_id)
    transport = consolidate_evidence_project(project_file, runs_root=args.runs_root, reconciliation_profile_id=args.reconciliation_profile_id, consolidation_profile_path=args.consolidation_profile, semantic_profile_path=args.llm_profile, llm_client=transport_client)
    after_transport = snapshot_a5_files(args.runs_root, project_id)
    check_reuse_gate(fresh, transport, semantic_delta=transport_client.semantic_generation_calls, attempt_delta=transport_client.provider_attempts, before=before_transport, after=after_transport)
    original_current = a5.require_current_validated(project_id=project_id, document_id=DOCUMENT_ID, consolidation_profile_id=profile.profile_id)
    semantic = load_semantic_profile(args.llm_profile)
    with tempfile.TemporaryDirectory(prefix="a5g3_invalidation_") as tmp:
        temp_root = Path(tmp) / "runs"; temp_root.mkdir()
        shutil.copytree(Path(args.runs_root) / project_id, temp_root / project_id)
        temp_profile = Path(tmp) / "consolidation_llm_changed.yaml"
        payload = json.loads(json.dumps(load_yaml(args.llm_profile)))
        payload["max_output_tokens"] += 1
        import yaml
        temp_profile.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        changed_semantic = load_semantic_profile(temp_profile)
        temp_client = CountingLLMClient(OpenAICompatibleLLMClient(runtime))
        invalidated = consolidate_evidence_project(project_file, runs_root=temp_root, reconciliation_profile_id=args.reconciliation_profile_id, consolidation_profile_path=args.consolidation_profile, semantic_profile_path=temp_profile, llm_client=temp_client)
        temp_store, temp_pointers = _stores(temp_root, project_id)
        temp_current = ConsolidationPersistenceService(temp_store, temp_pointers).require_current_validated(project_id=project_id, document_id=DOCUMENT_ID, consolidation_profile_id=profile.profile_id)
        check_invalidation_gate(old_hash=semantic.semantic_profile_hash, new_hash=changed_semantic.semantic_profile_hash, result=invalidated, semantic_calls=temp_client.semantic_generation_calls, provider_attempts=temp_client.provider_attempts, old_manifest=_ref(fresh.consolidation_manifest_ref), temp_current=temp_current)
        invalidation = {"temp_runs_root": str(temp_root), "changed_field": "max_output_tokens", "old_semantic_profile_hash": semantic.semantic_profile_hash, "new_semantic_profile_hash": changed_semantic.semantic_profile_hash, "reused": invalidated.reused, "semantic_generation_calls": temp_client.semantic_generation_calls, "provider_attempts": temp_client.provider_attempts, "old_manifest_ref": _ref(fresh.consolidation_manifest_ref), "new_manifest_ref": _ref(invalidated.consolidation_manifest_ref)}
    unchanged = a5.require_current_validated(project_id=project_id, document_id=DOCUMENT_ID, consolidation_profile_id=profile.profile_id)
    if unchanged.manifest_ref != original_current.manifest_ref or unchanged.current_pointer_ref != original_current.current_pointer_ref:
        raise AcceptanceError("temporary invalidation polluted original A5 CURRENT")
    summary = fresh.to_dict()
    return {"upstream": upstream, "preflight": {"a5_current_before_fresh": None, "historical_a5_file_count": len(preexisting_a5_files)}, "fresh": {"timing_seconds": fresh_elapsed, "semantic_generation_calls": client.semantic_generation_calls, "provider_attempts": client.provider_attempts, "result": summary, "artifact_refs": publication_refs(fresh), "validation_pass": True}, "persisted_audit": audit, "exact_rerun": {"reused": rerun.reused, "semantic_generation_call_count": rerun.semantic_generation_call_count, "provider_call_delta": 0, "no_writes": True, "timing_seconds": rerun_elapsed}, "transport_only_reuse": {"changed_runtime_fields": ["transport_id", "base_url", "request_model", "provider_family", "timeout_seconds"], "reused": transport.reused, "semantic_generation_calls": transport_client.semantic_generation_calls, "provider_attempts": transport_client.provider_attempts, "same_refs": publication_refs(fresh) == publication_refs(transport), "no_writes": True}, "semantic_invalidation": dict(invalidation, original_current_unchanged=True)}


def load_yaml(path: str | Path) -> dict[str, Any]:
    from short_drama.io import load_yaml as _load_yaml
    return _load_yaml(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="A5G3 Alice real-novel consolidation acceptance")
    parser.add_argument("--project", default=str(PROJECT)); parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--reconciliation-profile-id", default=RECONCILIATION_PROFILE_ID)
    parser.add_argument("--consolidation-profile", default=str(CONSOLIDATION_PROFILE)); parser.add_argument("--runtime-config", default=str(RUNTIME_CONFIG)); parser.add_argument("--llm-profile", default=str(SEMANTIC_PROFILE))
    args = parser.parse_args()
    failure_observer: dict[str, CountingLLMClient] = {}
    try:
        print(json.dumps(run_acceptance(args, failure_observer=failure_observer), ensure_ascii=False, indent=2))
    except Exception as exc:  # live failures are reported, never repaired here
        print(json.dumps(live_failure_payload(exc, failure_observer.get("client")), ensure_ascii=False, indent=2))
        print("A5G3 REAL-NOVEL ACCEPTANCE FAIL")
        return 2
    print("A5G3 REAL-NOVEL ACCEPTANCE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
