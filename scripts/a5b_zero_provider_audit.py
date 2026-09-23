"""v1.2 A5B — zero-provider input binding / indexing / corpus-shape audit.

This is the Phase A A5B delivery slice (issue #51). It resolves the exact
current-eligible A4 CURRENT EntityMap from a *pre-existing* run tree, loads the
exact pinned A3 input, binds every A3 local entity ref to its A4-bound A5
entity id, builds the source-ordered ``ConsolidationCandidateIndex``, and
reports a corpus-shape audit. It is **zero-provider** (no LLM / provider call)
and **read-only** (no artifact, pointer, CURRENT, or validation-report write).

The audit is a read-only gate: it exercises the deterministic A5B front half
against the Alice corpus (project ``a3e-real-novel``) and prints the input
binding, index shape, source-order authority, and coverage summary. It STOPs
here and does NOT implement blocking, semantic decisions, canonical ids, A5
persistence/CURRENT, or a CLI (those are later A5B-A5H slices).

Usage:
    python scripts/a5b_zero_provider_audit.py \
        [--runs-root PATH] [--project ID] [--document ID] [--profile ID]

Exit code 0 on a clean audit, 2 on any structural / binding / integrity failure.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    ConsolidationCurrentMissingError,
    ConsolidationPlanningResult,
    StoryIntegrityError,
    build_consolidation_planning,
)

DEFAULT_RUNS_ROOT = REPO_ROOT / "runs" / "a4e_real_novel"
DEFAULT_PROJECT = "a3e-real-novel"
DEFAULT_DOCUMENT = "src_001"
DEFAULT_PROFILE = "entity-reconciliation-v2"

# Field kinds (mirrors the A3 validation authority).
_PERSON_FIELDS = ("participants", "source_entity_ref", "target_entity_ref")
_LOCATION_FIELDS = ("locations",)
_ANY_FIELDS = ("subject_refs", "object_refs")


def _stores(runs_root: str | Path, project_id: str):
    root = Path(runs_root).expanduser() / project_id / "story"
    artifact_store = FileArtifactStore(root / "artifacts")
    pointer_store = FilePointerStore(root / "pointers", artifact_store)
    return artifact_store, pointer_store


def _namespace(bound_id: str) -> str:
    for prefix in ("char_", "loc_", "unres_"):
        if bound_id.startswith(prefix):
            return prefix[:-1]
    return "other"


def _field_namespaces(result: ConsolidationPlanningResult) -> dict[str, Counter]:
    """Count bound-ref namespaces per reference field across the whole index."""
    index = result.index
    counts: dict[str, Counter] = {
        "subject_refs": Counter(),
        "object_refs": Counter(),
        "participants": Counter(),
        "locations": Counter(),
        "source_entity_ref": Counter(),
        "target_entity_ref": Counter(),
    }
    for fact in index.facts:
        for ref in fact.subject_refs:
            counts["subject_refs"][_namespace(ref)] += 1
        for ref in fact.object_refs:
            counts["object_refs"][_namespace(ref)] += 1
    for event in index.events:
        for ref in event.participants:
            counts["participants"][_namespace(ref)] += 1
        for ref in event.locations:
            counts["locations"][_namespace(ref)] += 1
    for rel in index.relationships:
        counts["source_entity_ref"][_namespace(rel.source_entity_ref)] += 1
        counts["target_entity_ref"][_namespace(rel.target_entity_ref)] += 1
    return counts


def _source_order_check(candidates) -> tuple[bool, str]:
    """Return (ok, first_key) and assert a unique total source order."""
    if not candidates:
        return True, "<empty>"
    keys = [c.source_order_key for c in candidates]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        return False, keys[0]
    return True, keys[0]


def _audit_checks(result: ConsolidationPlanningResult) -> list[tuple[str, bool]]:
    """Evaluate the read-only A5B Phase A audit checks (each a (label, pass))."""
    index = result.index
    coverage = result.coverage
    counts = _field_namespaces(result)
    checks: list[tuple[str, bool]] = []

    # Coverage: the index must account for every A3 candidate exactly.
    checks.append(
        (
            "index counts match coverage (facts/events/relationships)",
            coverage.fact_candidate_count == len(index.facts)
            and coverage.event_candidate_count == len(index.events)
            and coverage.relationship_candidate_count == len(index.relationships),
        )
    )

    # Phase A: canonical / decision / conflict counts are all zero.
    checks.append(
        (
            "Phase A canonical/decision/conflict counts are zero",
            coverage.canonical_fact_count == 0
            and coverage.canonical_event_count == 0
            and coverage.canonical_relationship_count == 0
            and coverage.uncertain_decision_count == 0
            and coverage.story_conflict_count == 0,
        )
    )

    # Source-order authority: each category is a unique, total, sorted order.
    for label, cands in (
        ("facts source-ordered and unique", index.facts),
        ("events source-ordered and unique", index.events),
        ("relationships source-ordered and unique", index.relationships),
    ):
        ok, _ = _source_order_check(cands)
        checks.append((label, ok))

    # Field-kind legality: person fields never bind loc_*; location fields
    # never bind char_*; unres_* may appear in every field.
    person_ok = all(
        counts[field][_ns] == 0 for field in _PERSON_FIELDS for _ns in ("loc",)
    )
    checks.append(("person fields bind no loc_*", person_ok))
    location_ok = all(
        counts[field][_ns] == 0 for field in _LOCATION_FIELDS for _ns in ("char",)
    )
    checks.append(("location fields bind no char_*", location_ok))
    other_ok = all(
        counts[field][_ns] == 0
        for field in list(counts)
        for _ns in ("other",)
    )
    checks.append(("every bound id is in the char_/loc_/unres_ namespace", other_ok))

    return checks


def run_audit(
    store: FileArtifactStore,
    pointers: FilePointerStore,
    *,
    project_id: str,
    document_id: str,
    reconciliation_profile_id: str,
) -> int:
    result = build_consolidation_planning(
        store,
        pointers,
        project_id=project_id,
        document_id=document_id,
        reconciliation_profile_id=reconciliation_profile_id,
    )
    index = result.index
    coverage = result.coverage
    snapshot = result.snapshot

    print("== A5B zero-provider input binding / indexing / audit ==")
    print(f"project:                    {project_id}")
    print(f"document:                   {document_id}")
    print(f"reconciliation profile:     {reconciliation_profile_id}")
    print()

    print("-- A4 CURRENT (validated, read-only) --")
    em = snapshot.entity_map
    print(f"entity_map ref:             {snapshot.entity_map_ref.artifact_id}:"
          f"r{snapshot.entity_map_ref.revision}")
    print(f"validation report ref:      {snapshot.a4_validation_report_ref.artifact_id}:"
          f"r{snapshot.a4_validation_report_ref.revision}")
    print(f"current pointer ref:        {snapshot.a4_current_pointer_ref.artifact_id}")
    print(f"entity map entries:         {len(em.entries)}")
    resolved = sum(1 for e in em.entries if e.status == "resolved")
    unresolved = sum(1 for e in em.entries if e.status == "unresolved")
    print(f"  resolved / unresolved:    {resolved} / {unresolved}")
    a3 = snapshot.a3_input
    print(f"extraction profile:         {a3.extraction_profile_id}:"
          f"{a3.extraction_profile_hash[:12]}")
    print()

    print("-- exact A3 input --")
    print(f"source document ref:        {a3.source_document_ref.artifact_id}")
    print(f"chunk manifest ref:         {a3.chunk_manifest_ref.artifact_id}")
    print(f"chunks (chunk_refs):        {len(snapshot.chunk_manifest.chunk_refs)}")
    print(f"candidate extractions:      {len(snapshot.candidate_extractions)}")
    print(f"profile id:                 {snapshot.chunk_manifest.profile.profile_id}")
    char_total = sum(len(e.candidates.characters) for e in snapshot.candidate_extractions)
    loc_total = sum(len(e.candidates.locations) for e in snapshot.candidate_extractions)
    unres_total = sum(len(e.candidates.unresolved_mentions) for e in snapshot.candidate_extractions)
    print(f"A3 char/loc/unres candidates: {char_total}/{loc_total}/{unres_total} "
          f"(total {char_total + loc_total + unres_total})")
    print()

    print("-- ConsolidationCandidateIndex (source-ordered) --")
    print(f"facts:                      {len(index.facts)}")
    print(f"events:                     {len(index.events)}")
    print(f"relationships:              {len(index.relationships)}")
    total = len(index.facts) + len(index.events) + len(index.relationships)
    print(f"total consolidation candidates: {total}")
    print()

    print("-- source-order authority (first key per category) --")
    for label, cands in (
        ("facts", index.facts),
        ("events", index.events),
        ("relationships", index.relationships),
    ):
        _ok, first = _source_order_check(cands)
        print(f"  {label:14} first: {first}")
    print()

    print("-- bound-ref namespace distribution per field --")
    counts = _field_namespaces(result)
    for field, counter in counts.items():
        dist = " ".join(f"{ns}={n}" for ns, n in sorted(counter.items())) or "(none)"
        print(f"  {field:18} {dist}")
    print()

    print("-- coverage summary --")
    print(f"  fact_candidate_count:        {coverage.fact_candidate_count}")
    print(f"  event_candidate_count:       {coverage.event_candidate_count}")
    print(f"  relationship_candidate_count:{' ' * 1}{coverage.relationship_candidate_count}")
    print(f"  canonical_*_count:           {coverage.canonical_fact_count}/"
          f"{coverage.canonical_event_count}/{coverage.canonical_relationship_count} (Phase A: 0)")
    print(f"  uncertain_decision_count:    {coverage.uncertain_decision_count} (Phase A: 0)")
    print(f"  story_conflict_count:        {coverage.story_conflict_count} (Phase A: 0)")
    print()

    print("-- audit checks --")
    checks = _audit_checks(result)
    all_pass = True
    for label, passed in checks:
        all_pass = all_pass and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print()

    if not all_pass:
        print("A5B AUDIT RESULT: FAIL", file=sys.stderr)
        return 2
    print("A5B AUDIT RESULT: PASS (zero-provider, read-only)")
    print("Phase A complete: input binding + indexing + source order + coverage.")
    print("STOP: no blocking, provider call, canonical id, persistence, or CURRENT.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A5B zero-provider input binding / indexing / corpus-shape audit"
    )
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--document", default=DEFAULT_DOCUMENT)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    args = parser.parse_args(argv)

    try:
        store, pointers = _stores(args.runs_root, args.project)
        return run_audit(
            store,
            pointers,
            project_id=args.project,
            document_id=args.document,
            reconciliation_profile_id=args.profile,
        )
    except (ConsolidationCurrentMissingError, StoryIntegrityError) as exc:
        print(f"A5B AUDIT RESULT: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive (unexpected)
        print(f"A5B AUDIT RESULT: ERROR ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
