"""A6B tests: exact A5 binding, deterministic exact-ref planning, zero-provider.

Covers the A6B planning surface required by Issue #87:

- ``StoryAnalysisInputSnapshot`` binds the *exact* A5 CURRENT and the exact A4
  EntityMap through the manifest's ``entity_map_ref``, and requires the A3
  identity to be exactly equal across the A4 EntityMap and the A5 manifest
  (failing closed otherwise).
- 100% canonical-character coverage with deterministic exact-ref joins for the
  character evidence packages (facts / events / relationships / transitions /
  conflicts / unresolved).
- A deterministic narrative-order event stream and contiguous, non-overlapping
  plot windows with the exact-one event-ownership invariant (context overlap
  allowed but never confers ownership).
- A backend-neutral plan identity hash (no provider/model/GPU/credential),
  deterministic across rebuilds.
- Fail-closed budget enforcement: an overflowing package is never truncated or
  sampled; it raises.
- The production profile (created from the A6B Alice zero-provider audit) loads
  and plans.

A real synthetic A5 CURRENT is produced by reusing the A5F2 publication
helpers, so these tests exercise the exact persistence bindings (not mocks).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from test_story_a5f2_current_publication import (
    CONSOLIDATION_PROFILE_ID,
    DOCUMENT,
    PROJECT,
    _build_tree_with_a4,
    _planning,
    _publish_a5,
)

from short_drama.artifacts.canonical import canonical_json_bytes
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    ConsolidationCurrentMissingError,
    StoryAnalysisModelError,
    CanonicalEvent,
    StoryAnalysisPlanningError,
    StoryAnalysisPlanningPolicy,
    StoryIntegrityError,
    assert_full_character_coverage,
    build_character_evidence_packages,
    build_global_index_base,
    build_story_analysis_plan,
    build_story_analysis_plan_from_profile,
    build_story_analysis_snapshot,
    load_story_analysis_profile,
    ordered_event_stream,
    plan_plot_windows,
    planning_policy_ids_from_profile,
    validate_window_ownership,
)
from short_drama.story.chunking import TOKEN_COUNTER_ID

# The four versioned A6 planning policy IDs (bound into the plan hash, BLOCK 2).
_POLICY_IDS = {
    "character_analysis_policy_id": "a6-character-v1",
    "plot_window_policy_id": "a6-window-v1",
    "global_skeleton_policy_id": "a6-skeleton-v1",
    "story_bible_policy_id": "a6-bible-v1",
}


def _plan(snap, policy):
    """Build a plan with the default four versioned policy IDs + token counter."""
    return build_story_analysis_plan(snap, policy, planning_policy_ids=_POLICY_IDS)


_POLICY = StoryAnalysisPlanningPolicy(
    character_packet_max_estimated_tokens=100_000,
    plot_window_packet_max_estimated_tokens=100_000,
    plot_window_owned_event_target=2,
    plot_window_context_event_count=1,
    global_skeleton_packet_max_estimated_tokens=100_000,
    story_bible_packet_max_estimated_tokens=100_000,
)

# The exact frozen planning policy measured by the A6B Alice audit. The two
# whole-story ceilings are the A5-DERIVED BASE BOUND ONLY (BLOCK 1 sequencing
# contradiction: the full semantic ceiling is deferred to A6C/A6D/A6E/A6G).
_FROZEN_POLICY = {
    "character_packet_max_estimated_tokens": 70_000,
    "plot_window_packet_max_estimated_tokens": 46_000,
    "plot_window_owned_event_target": 1,
    "plot_window_context_event_count": 0,
    "global_skeleton_packet_max_estimated_tokens": 114_000,
    "story_bible_packet_max_estimated_tokens": 114_000,
}

_FORBIDDEN = (
    "provider", "model", "base_url", "gpu", "slot", "concurrency",
    "parallel", "timeout", "credential", "api_key", "llama", "openai",
    "anthropic",
)


def _mk_event(n: int, participants: tuple[str, ...] = ()) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=f"evt_{n:06d}",
        narrative_order=n,
        summary_zh=f"事件{n}",
        participants=participants,
        locations=(),
        temporal_mode="normal",
        candidate_event_refs=(f"CH001_C001:cand_evt_{n:06d}",),
        evidence_refs=(),
        first_source_order=f"CH001_P{n:04d}:1",
    )


def _snapshot(tmp_path: Path):
    tree, _ext, _a3, _idx = _build_tree_with_a4(tmp_path)
    planning = _planning(tree)
    pub = _publish_a5(tree, planning)
    snap = build_story_analysis_snapshot(
        tree.store,
        tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
    )
    return tree, planning, pub, snap


# --- exact A5 binding + A3 identity -----------------------------------------

def test_snapshot_binds_exact_a5_current_and_a4(tmp_path: Path) -> None:
    _tree, planning, pub, snap = _snapshot(tmp_path)

    # The snapshot binds the exact published A5 CURRENT manifest and pointers.
    assert snap.consolidation_manifest_ref == pub.consolidation_manifest_ref
    assert snap.a5_current_pointer_ref == pub.current_pointer_ref
    assert snap.a5_validation_report_ref == pub.validation_report_ref

    # The exact pinned A4 EntityMap, resolved through the manifest's ref (not
    # a re-resolved "current").
    em_ref = snap.consolidation_manifest.entity_map_ref
    assert em_ref.artifact_id == planning.snapshot.entity_map_ref.artifact_id
    assert em_ref.content_hash == planning.snapshot.entity_map_ref.content_hash

    # A3 identity is exactly equal across the A4 EntityMap and the A5 manifest.
    assert snap.entity_map.a3_input == snap.consolidation_manifest.upstream_identity.a3_input

    # Fixture shape: 2 chars, 1 loc, 1 unresolved, 1 fact, 1 event, 1
    # relationship, 0 conflicts, 0 transitions.
    assert len(snap.canonical_characters) == 2
    assert len(snap.canonical_locations) == 1
    assert len(snap.unresolved_entities) == 1
    assert len(snap.facts) == 1
    assert len(snap.events) == 1
    assert len(snap.relationships) == 1
    assert len(snap.conflicts) == 0
    assert len(snap.state_transitions) == 0


def test_snapshot_missing_current_fails_closed(tmp_path: Path) -> None:
    tree, _ext, _a3, _idx = _build_tree_with_a4(tmp_path)
    # No A5 CURRENT has been published yet.
    with pytest.raises(ConsolidationCurrentMissingError):
        build_story_analysis_snapshot(
            tree.store,
            tree.pointers,
            project_id=PROJECT,
            document_id=DOCUMENT,
            consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
        )


def test_snapshot_missing_a4_leaf_fails_closed(tmp_path: Path) -> None:
    """A pinned A4 leaf that is missing from the store fails closed (no fallback)."""
    tree, _planning, _pub, snap = _snapshot(tmp_path)
    ref = snap.entity_map.canonical_character_registry_ref
    leaf_path = tree.store._path(ref.artifact_type, ref.artifact_id, ref.revision)
    assert leaf_path.is_file()
    leaf_path.unlink()
    # The exact pinned A4 character registry is now gone; the snapshot must
    # refuse to fall back to "latest A4" and fail closed.
    with pytest.raises(StoryIntegrityError):
        build_story_analysis_snapshot(
            tree.store,
            tree.pointers,
            project_id=PROJECT,
            document_id=DOCUMENT,
            consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
        )


def test_snapshot_wrong_profile_fails_closed(tmp_path: Path) -> None:
    """A different consolidation profile has no A5 CURRENT -> fails closed."""
    tree, _planning, _pub, _snap = _snapshot(tmp_path)
    with pytest.raises(ConsolidationCurrentMissingError):
        build_story_analysis_snapshot(
            tree.store,
            tree.pointers,
            project_id=PROJECT,
            document_id=DOCUMENT,
            consolidation_profile_id="consolidation-other",
        )


# --- 100% character coverage + exact-ref joins ------------------------------

def test_character_coverage_is_100_percent_and_unique(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    packages = build_character_evidence_packages(snap)
    assert_full_character_coverage(snap, packages)

    assert len(packages) == len(snap.canonical_characters)
    refs = [p.character_ref for p in packages]
    assert len(set(refs)) == len(refs)
    assert set(refs) == {c.canonical_id for c in snap.canonical_characters}


def test_character_evidence_exact_ref_joins(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    packages = {p.character_ref: p for p in build_character_evidence_packages(snap)}

    fact = snap.facts[0]
    event = snap.events[0]
    rel = snap.relationships[0]
    subject = fact.subject_refs[0]
    participant = event.participants[0]

    # The fact's subject character has the fact.
    assert {f.fact_id for f in packages[subject].related_facts} == {fact.fact_id}
    # The event's participant character has the event.
    assert {e.event_id for e in packages[participant].participating_events} == {event.event_id}
    # The relationship is present at BOTH endpoints.
    assert rel.relationship_id in {r.relationship_id for r in packages[rel.source_entity_ref].relationships}
    assert rel.relationship_id in {r.relationship_id for r in packages[rel.target_entity_ref].relationships}
    # The relationship's target character does NOT have the fact (it is not a
    # subject/object of it) -- no fuzzy/superset matching.
    assert fact.fact_id not in {f.fact_id for f in packages[rel.target_entity_ref].related_facts}
    # The location is never treated as a character package.
    assert fact.object_refs[0] not in packages


def test_character_packet_deterministic_bytes_and_hash(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    p1 = build_character_evidence_packages(snap)
    p2 = build_character_evidence_packages(snap)
    for a, b in zip(p1, p2):
        assert a.character_ref == b.character_ref
        assert a.canonical_bytes() == b.canonical_bytes()
        assert a.estimated_tokens() == b.estimated_tokens()
        assert a.content_hash() == b.content_hash()
    # estimated_tokens is consistent with the canonical byte size.
    assert all(p.estimated_tokens() >= 1 for p in p1)


# --- event stream + plot-window exact-one ownership -------------------------

def test_event_stream_ordered_by_narrative_order() -> None:
    events = [
        _mk_event(3), _mk_event(1), _mk_event(2),
        _mk_event(5), _mk_event(4),
    ]
    # Build a lightweight snapshot via the real helper on a constructed set:
    # ordered_event_stream sorts by (narrative_order, event_id).
    stream = _sorted_stream(events)
    assert [e.event_id for e in stream] == [
        "evt_000001", "evt_000002", "evt_000003", "evt_000004", "evt_000005",
    ]


def test_plan_plot_windows_exact_one_ownership() -> None:
    n = 20
    events = [_mk_event(i) for i in range(1, n + 1)]
    all_ids = frozenset(e.event_id for e in events)

    for target in (1, 3, 5, 10, 20):
        for ctx in (0, 2, 4):
            windows = plan_plot_windows(events, owned_event_target=target, context_event_count=ctx)
            # Exact-one ownership invariant (no missing/duplicate/unknown).
            validate_window_ownership(windows, all_ids)
            owned = Counter(eid for w in windows for eid in w.owned_event_ids)
            assert set(owned) == all_ids
            assert all(c == 1 for c in owned.values())
            # Windows are sequential and uniquely ided.
            assert [w.window_ordinal for w in windows] == list(range(1, len(windows) + 1))
            ids = [w.window_id for w in windows]
            assert len(set(ids)) == len(ids)
            # Context refs are always known and never include owned events.
            for w in windows:
                assert set(w.context_event_ids) <= all_ids
                assert not (set(w.context_event_ids) & set(w.owned_event_ids))


def test_plot_window_context_overlap_does_not_confer_ownership() -> None:
    n = 10
    events = [_mk_event(i) for i in range(1, n + 1)]
    all_ids = frozenset(e.event_id for e in events)
    windows = plan_plot_windows(events, owned_event_target=3, context_event_count=2)

    # With context > 0 and multiple windows, boundary events appear as context
    # in a neighbour while owned by their own window.
    context_owners: dict[str, int] = {}
    for w in windows:
        for eid in w.context_event_ids:
            context_owners[eid] = context_owners.get(eid, 0) + 1
    assert len(windows) > 1
    overlapping = [eid for eid, c in context_owners.items() if c >= 1]
    assert overlapping, "expected boundary context overlap"
    # Yet every overlapping event is still owned by exactly one window.
    owned = Counter(eid for w in windows for eid in w.owned_event_ids)
    for eid in overlapping:
        assert owned[eid] == 1


def test_plan_plot_windows_rejects_unknown_and_empty() -> None:
    events = [_mk_event(1), _mk_event(2)]
    all_ids = frozenset(e.event_id for e in events)
    # An unknown ref in a window fails closed.
    bad = plan_plot_windows(events, owned_event_target=2, context_event_count=0)
    from short_drama.story.story_analysis_planning import PlotWindowPlan
    tampered = [
        PlotWindowPlan(window_id=w.window_id, window_ordinal=w.window_ordinal, owned_event_ids=w.owned_event_ids, context_event_ids=("evt_999999",))
        for w in bad
    ]
    with pytest.raises(StoryAnalysisPlanningError, match="unknown context event"):
        validate_window_ownership(tampered, all_ids)
    # An empty window plan with a non-empty stream is a coverage failure.
    with pytest.raises(StoryAnalysisPlanningError, match="no owned window"):
        validate_window_ownership([], all_ids)
    # A duplicate-owned event is a coverage failure (both events still covered,
    # so the missing check passes and the duplicate check fires).
    from short_drama.story.story_analysis_planning import PlotWindowPlan
    dup2 = (
        PlotWindowPlan(window_id="win_a", window_ordinal=1, owned_event_ids=("evt_000001",), context_event_ids=()),
        PlotWindowPlan(window_id="win_b", window_ordinal=2, owned_event_ids=("evt_000001", "evt_000002"), context_event_ids=()),
    )
    with pytest.raises(StoryAnalysisPlanningError, match="more than one window"):
        validate_window_ownership(dup2, all_ids)


def _sorted_stream(events: list[CanonicalEvent]) -> tuple[CanonicalEvent, ...]:
    from short_drama.story.story_analysis_planning import ordered_event_stream
    # Build a throwaway snapshot exposing only the events for ordering.
    import types
    snap = types.SimpleNamespace(events=tuple(events))
    return ordered_event_stream(snap)


# --- plan identity: deterministic + backend-neutral -------------------------

def test_plan_hash_deterministic_and_backend_neutral(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan1 = _plan(snap, _POLICY)
    plan2 = _plan(snap, _POLICY)
    assert plan1.plan_hash == plan2.plan_hash
    assert re.fullmatch(r"[0-9a-f]{64}", plan1.plan_hash)

    canon = canonical_json_bytes(plan1.plan_identity_payload()).decode("utf-8").lower()
    for forbidden in _FORBIDDEN:
        assert forbidden not in canon, f"plan identity leaks {forbidden!r}"

    # The plan binds the exact A5 CURRENT manifest ref and the policy.
    payload = plan1.plan_identity_payload()
    assert payload["consolidation_manifest_ref"]["artifact_id"] == snap.consolidation_manifest_ref.artifact_id
    assert payload["planning_policy"] == _POLICY.to_dict()
    assert len(payload["character_package_hashes"]) == len(snap.canonical_characters)
    assert payload["event_stream"] == [e.event_id for e in snap.events]
    assert len(payload["windows"]) == len(plan1.windows)


def test_plan_hash_binds_policy(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    h_a = _plan(
        snap, StoryAnalysisPlanningPolicy(100_000, 100_000, 2, 1, 100_000, 100_000)
    ).plan_hash
    h_b = _plan(
        snap, StoryAnalysisPlanningPolicy(100_000, 100_000, 3, 1, 100_000, 100_000)
    ).plan_hash
    assert h_a != h_b


def test_global_index_base_is_a5_derived_and_deterministic(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    g1 = build_global_index_base(snap)
    g2 = build_global_index_base(snap)
    assert g1.canonical_bytes() == g2.canonical_bytes()
    assert g1.content_hash() == g2.content_hash()
    assert g1.estimated_tokens() >= 1
    # It carries every canonical event exactly once (A5-derived event index).
    assert [e["event_id"] for e in g1.event_index] == [e.event_id for e in snap.events]


# --- fail-closed budget enforcement -----------------------------------------

def test_budget_overflow_fails_closed_not_truncated(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)

    # Character budget smaller than the real (non-trivial) character package.
    with pytest.raises(StoryAnalysisPlanningError, match="refusing to truncate"):
        _plan(
            snap,
            StoryAnalysisPlanningPolicy(1, 100_000, 1, 0, 100_000, 100_000),
        )
    # Window budget smaller than the real window packet.
    with pytest.raises(StoryAnalysisPlanningError, match="refusing to truncate"):
        _plan(
            snap,
            StoryAnalysisPlanningPolicy(100_000, 1, 1, 0, 100_000, 100_000),
        )
    # Global-skeleton budget smaller than the real A5-derived index base.
    with pytest.raises(StoryAnalysisPlanningError, match="refusing to truncate"):
        _plan(
            snap,
            StoryAnalysisPlanningPolicy(100_000, 100_000, 1, 0, 1, 100_000),
        )
    # A passing budget yields a full plan (no truncation of coverage).
    plan = _plan(snap, _POLICY)
    assert len(plan.character_packages) == len(snap.canonical_characters)
    assert list(plan.event_stream) == [e.event_id for e in snap.events]


def test_policy_validation_fail_closed() -> None:
    with pytest.raises(StoryAnalysisModelError):
        StoryAnalysisPlanningPolicy(
            character_packet_max_estimated_tokens=0,
            plot_window_packet_max_estimated_tokens=1,
            plot_window_owned_event_target=1,
            plot_window_context_event_count=0,
            global_skeleton_packet_max_estimated_tokens=1,
            story_bible_packet_max_estimated_tokens=1,
        )
    with pytest.raises(StoryAnalysisModelError):
        StoryAnalysisPlanningPolicy(
            character_packet_max_estimated_tokens=1,
            plot_window_packet_max_estimated_tokens=1,
            plot_window_owned_event_target=0,
            plot_window_context_event_count=0,
            global_skeleton_packet_max_estimated_tokens=1,
            story_bible_packet_max_estimated_tokens=1,
        )


# --- production profile (from the A6B audit) --------------------------------

def test_production_profile_loads_with_audit_frozen_policy() -> None:
    profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml")
    assert profile.profile_id == "global-story-analysis-v1"
    assert profile.planning_policy.to_dict() == _FROZEN_POLICY


def test_production_profile_plans_the_fixture(tmp_path: Path) -> None:
    profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml")
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = build_story_analysis_plan_from_profile(snap, profile)
    assert plan.plan_hash
    assert len(plan.character_packages) == len(snap.canonical_characters)
    validate_window_ownership(plan.windows, frozenset(plan.event_stream))
    # The profile's planning policy is what the plan recorded.
    assert plan.planning_policy.to_dict() == _FROZEN_POLICY


def test_production_profile_identity_is_stable() -> None:
    profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml")
    h = profile.content_hash()
    assert re.fullmatch(r"[0-9a-f]{64}", h)
    assert h == load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml").content_hash()


# --- BLOCK 2: the plan identity binds the versioned policy IDs + token counter

def test_plan_hash_binds_policy_ids(tmp_path: Path) -> None:
    """BLOCK 2: changing a versioned policy ID changes the plan hash."""
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    h_a = build_story_analysis_plan(
        snap, _POLICY, planning_policy_ids=dict(_POLICY_IDS)
    ).plan_hash
    h_b = build_story_analysis_plan(
        snap, _POLICY,
        planning_policy_ids={**_POLICY_IDS, "character_analysis_policy_id": "a6-character-v2"},
    ).plan_hash
    assert h_a != h_b
    # The plan identity payload binds the four versioned policy IDs.
    plan = build_story_analysis_plan(snap, _POLICY, planning_policy_ids=dict(_POLICY_IDS))
    payload = plan.plan_identity_payload()
    assert payload["planning_policy_ids"] == dict(_POLICY_IDS)
    assert set(payload["planning_policy_ids"].keys()) == set(_POLICY_IDS.keys())


def test_plan_hash_binds_token_counter_id(tmp_path: Path) -> None:
    """BLOCK 2: changing the token-counter ID changes the plan hash."""
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    h_a = build_story_analysis_plan(
        snap, _POLICY,
        planning_policy_ids=dict(_POLICY_IDS),
        token_counter_id=TOKEN_COUNTER_ID,
    ).plan_hash
    h_b = build_story_analysis_plan(
        snap, _POLICY,
        planning_policy_ids=dict(_POLICY_IDS),
        token_counter_id="utf8-bytes-div3-v2",
    ).plan_hash
    assert h_a != h_b
    plan = build_story_analysis_plan(
        snap, _POLICY, planning_policy_ids=dict(_POLICY_IDS)
    )
    assert plan.token_counter_id == TOKEN_COUNTER_ID
    assert plan.plan_identity_payload()["token_counter_id"] == TOKEN_COUNTER_ID


def test_policy_ids_validation_fail_closed(tmp_path: Path) -> None:
    """BLOCK 2: the four versioned policy IDs are validated (fail closed)."""
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    # Missing a policy ID.
    with pytest.raises(StoryAnalysisPlanningError, match="four versioned policy"):
        build_story_analysis_plan(
            snap, _POLICY,
            planning_policy_ids={k: v for k, v in _POLICY_IDS.items() if k != "story_bible_policy_id"},
        )
    # An extra (unknown) key.
    with pytest.raises(StoryAnalysisPlanningError, match="four versioned policy"):
        build_story_analysis_plan(
            snap, _POLICY, planning_policy_ids={**_POLICY_IDS, "extra_policy_id": "x"}
        )
    # An empty value.
    with pytest.raises(StoryAnalysisPlanningError, match="non-empty string"):
        build_story_analysis_plan(
            snap, _POLICY,
            planning_policy_ids={**_POLICY_IDS, "story_bible_policy_id": ""},
        )


def test_planning_policy_ids_from_profile(tmp_path: Path) -> None:
    """BLOCK 2: the four versioned policy IDs are extracted from the profile."""
    profile = load_story_analysis_profile(
        REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml"
    )
    assert planning_policy_ids_from_profile(profile) == _POLICY_IDS


# --- BLOCK 3: the audit is zero-write (before/after runs-tree equality) ------

def _load_audit_module():
    import importlib.util
    path = REPO_ROOT / "scripts" / "a6b_alice_zero_provider_audit.py"
    spec = importlib.util.spec_from_file_location("a6b_audit_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_zero_write() -> None:
    """BLOCK 3 regression: the audit does NOT mutate the story runs tree.

    Runs the full zero-provider audit against the real Alice corpus and asserts
    the before/after SHA-256 snapshot of the story runs tree is exactly equal.
    """
    audit = _load_audit_module()
    store, pointers = audit._stores(audit.DEFAULT_RUNS_ROOT, audit.DEFAULT_PROJECT)
    before = audit._snapshot_runs_tree(audit.DEFAULT_RUNS_ROOT, audit.DEFAULT_PROJECT)
    report, all_pass = audit.run_audit(
        store,
        pointers,
        project_id=audit.DEFAULT_PROJECT,
        document_id=audit.DEFAULT_DOCUMENT,
        consolidation_profile_id=audit.DEFAULT_CONSOLIDATION_PROFILE,
    )
    after = audit._snapshot_runs_tree(audit.DEFAULT_RUNS_ROOT, audit.DEFAULT_PROJECT)
    assert before == after, "audit mutated the story runs tree (not zero-write)"
    assert report["zero_write"] is True
    assert report["result"] == "PASS"
    assert all_pass


def test_audit_refuses_report_inside_runs_tree(tmp_path: Path) -> None:
    """BLOCK 3: a report path inside the story runs tree is refused (exit 2)."""
    audit = _load_audit_module()
    bad_report = audit.DEFAULT_RUNS_ROOT / audit.DEFAULT_PROJECT / "story" / "bad.json"
    rc = audit.main(
        [
            "--report", str(bad_report),
            "--runs-root", str(audit.DEFAULT_RUNS_ROOT),
            "--project", audit.DEFAULT_PROJECT,
        ]
    )
    assert rc == 2
    assert not bad_report.exists()


# --- BLOCK 4: corpus-derived candidate space + deterministic selection -------

def test_window_candidate_space_is_corpus_derived() -> None:
    """BLOCK 4: the candidate space is generated from the event universe."""
    audit = _load_audit_module()
    space = audit._window_candidate_space(166)
    # owned target 1..166 (exhaustive) x context 0..max(1, 166 // 10) = 0..16.
    assert len(space) == 166 * 17
    assert {t for t, _ in space} == set(range(1, 167))
    assert {c for _, c in space} == set(range(0, 17))
    # A smaller corpus: owned 1..5 x context 0..max(1, 5 // 10) = 0..1.
    space_small = audit._window_candidate_space(5)
    assert len(space_small) == 5 * 2
    assert {t for t, _ in space_small} == {1, 2, 3, 4, 5}
    assert {c for _, c in space_small} == {0, 1}
    # Empty corpus -> empty space.
    assert audit._window_candidate_space(0) == []


def test_window_policy_selection_is_deterministic() -> None:
    """BLOCK 4: the selection objective is an explicit deterministic min."""
    audit = _load_audit_module()
    sweep = [
        {"max_window_packet_estimated_tokens": 100, "window_count": 5, "overlap_cost_tokens": 10},
        {"max_window_packet_estimated_tokens": 80, "window_count": 10, "overlap_cost_tokens": 5},
        {"max_window_packet_estimated_tokens": 80, "window_count": 4, "overlap_cost_tokens": 5},
        {"max_window_packet_estimated_tokens": 80, "window_count": 4, "overlap_cost_tokens": 3},
    ]
    selected = audit._select_frozen_window_policy(sweep)
    # The lexicographic min of (max_packet, window_count, overlap) is the last.
    assert selected == sweep[3]
    # Deterministic: re-running gives the same result.
    assert audit._select_frozen_window_policy(sweep) == selected
    # Empty sweep fails closed.
    with pytest.raises(ValueError):
        audit._select_frozen_window_policy([])


# --- BLOCK 5: exact typed-leaf serialization --------------------------------

def test_leaf_bytes_measures_exact_typed_leaf(tmp_path: Path) -> None:
    """BLOCK 5: the leaf measurement serializes the exact typed leaf's to_dict()."""
    audit = _load_audit_module()
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    from short_drama.story.story_analysis_planning import _canonical_bytes

    leaf = snap.canonical_fact_set
    leaf_dict = leaf.to_dict()
    assert "schema_version" in leaf_dict
    # The measured leaf bytes == the canonical bytes of the typed leaf's to_dict().
    assert audit._leaf_bytes(leaf) == len(_canonical_bytes(leaf_dict))
    # It is NOT the raw array of children (which lacks the schema_version +
    # the set-level fields such as state_transitions).
    raw_array_bytes = len(_canonical_bytes([f.to_dict() for f in leaf.facts]))
    assert audit._leaf_bytes(leaf) != raw_array_bytes


# --- owned/context overlap invariant (additional) ---------------------------

def test_validate_window_ownership_rejects_owned_context_overlap() -> None:
    """The owned∩context overlap invariant fails closed (additional invariant).

    A window that lists the same event in both ``owned_event_ids`` and
    ``context_event_ids`` is a planning bug and must raise.
    """
    from short_drama.story.story_analysis_planning import PlotWindowPlan
    windows = (
        PlotWindowPlan(
            window_id="win_a",
            window_ordinal=1,
            owned_event_ids=("evt_000001", "evt_000002"),
            context_event_ids=("evt_000002",),
        ),
    )
    all_ids = frozenset({"evt_000001", "evt_000002"})
    with pytest.raises(StoryAnalysisPlanningError, match="both owned and context"):
        validate_window_ownership(windows, all_ids)
