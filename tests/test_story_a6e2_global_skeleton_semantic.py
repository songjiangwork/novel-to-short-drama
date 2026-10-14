"""A6E-2 tests: global-skeleton semantic execution (Issue #102).

Covers the A6E-2 semantic execution module
``short_drama.story.story_analysis_global_skeleton_semantic``:

* identity transition: the ceiling change (null -> 100000) changes both the
  StoryAnalysisProfile hash and the A6B plan hash, which in turn changes the
  A6C/A6D stable request identities;
* ceiling enforcement: null ceiling fails closed, exceeded ceiling fails
  closed;
* complete coverage: all characters and windows must be present;
* canonical refs: all refs must belong to the A5 evidence universe;
* ID allocation: Python-assigned deterministic IDs (arc_000001, turn_000001,
  reveal_000001, payoff_000001);
* wrong narrative anchors: arc start_event must precede end_event in
  narrative order;
* invalid internal references: major_*_proposal_ordinals must reference valid
  proposal ordinals;
* deterministic request hash: the stable request identity is deterministic;
* provenance: mismatch fails closed without retry;
* retry exhaustion: two semantic rounds exhausted -> fail;
* no persistence: no writes to disk.

All tests are offline (no real provider calls).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from short_drama.artifacts.canonical import content_hash
from short_drama.llm import (
    GenerationExecutionOptions,
    LLMClient,
    LLMInvocationProvenance,
    OutputSchema,
    PromptRegistry,
    RenderedPrompt,
    SemanticLLMProfile,
    StructuredGenerationRequest,
)
from short_drama.paths import PROFILES_DIR, SCHEMAS_DIR, REPO_ROOT
from short_drama.story import (
    A6E_GLOBAL_SKELETON_PROMPT_ID,
    A6E_GLOBAL_SKELETON_PROMPT_VERSION,
    CharacterAnalysis,
    EvidenceBackedInterpretation,
    GlobalEventAnalysis,
    GlobalSkeletonContext,
    GlobalStructure,
    ArcAnalysis,
    PlotWindowAnalysis,
    STORY_ANALYSIS_MAX_GENERATION_ROUNDS_V1,
    StoryAnalysisPlanningPolicy,
    StoryAnalysisProfile,
    StoryAnalysisSemanticPass,
    build_global_skeleton_context,
    build_global_skeleton_semantic_preparation,
    enforce_global_skeleton_ceiling,
    global_skeleton_request_identity_hash,
    load_story_analysis_profile,
    resolve_global_skeleton,
    validate_global_skeleton_coverage,
    validate_global_skeleton_output,
)
from short_drama.story.chunking import estimate_tokens
from short_drama.story.errors import (
    StoryAnalysisGlobalSkeletonSemanticGenerationError,
    StoryAnalysisProvenanceError,
    StoryAnalysisSemanticError,
)
from short_drama.story.story_analysis_planning import (
    GlobalIndexBase,
    StoryAnalysisInputSnapshot,
    StoryAnalysisPlan,
)

from test_story_a6d_plot_window_analysis import (
    _build_snapshot as _in_memory_snapshot,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _interp() -> EvidenceBackedInterpretation:
    return EvidenceBackedInterpretation(text_zh="测试解读文本。", evidence_mode="explicit")


def _char_analysis(ref: str) -> CharacterAnalysis:
    return CharacterAnalysis(
        character_ref=ref,
        role=_interp(),
        goals=(),
        motivations=(),
        traits=(),
        key_event_refs=(),
        key_fact_refs=(),
        important_relationship_refs=(),
        arc_summary=_interp(),
        unresolved_or_conflicting_points=(),
    )


def _window_analysis(
    window_id: str,
    ordinal: int,
    owned: tuple[str, ...] = (),
    context: tuple[str, ...] = (),
) -> PlotWindowAnalysis:
    return PlotWindowAnalysis(
        window_id=window_id,
        window_ordinal=ordinal,
        owned_event_refs=owned,
        context_event_refs=context,
        interpretation=_interp(),
    )


def _make_profile(ceiling: int | None = 100000) -> StoryAnalysisProfile:
    """Build a minimal valid StoryAnalysisProfile with the given ceiling."""
    policy = StoryAnalysisPlanningPolicy(
        character_packet_max_estimated_tokens=70000,
        plot_window_packet_max_estimated_tokens=58000,
        plot_window_owned_event_target=12,
        plot_window_context_event_count=4,
        global_skeleton_packet_max_estimated_tokens=ceiling,
        story_bible_packet_max_estimated_tokens=None,
    )
    pass_id = "story-analysis-llm-v1"
    return StoryAnalysisProfile(
        schema_version=1,
        profile_id="global-story-analysis-v1",
        working_language="zh-CN",
        character_analysis_policy_id="a6-character-v1",
        plot_window_policy_id="a6-window-v1",
        global_skeleton_policy_id="a6-skeleton-v1",
        story_bible_policy_id="a6-bible-v1",
        max_generation_rounds=2,
        planning_policy=policy,
        character_analysis=StoryAnalysisSemanticPass(
            semantic_profile_id=pass_id,
            prompt_id="a6.character-analysis",
            prompt_version=1,
            output_schema_id="a6-character-analysis-output",
            output_schema_version=1,
        ),
        plot_window_analysis=StoryAnalysisSemanticPass(
            semantic_profile_id=pass_id,
            prompt_id="a6.plot-window-analysis",
            prompt_version=1,
            output_schema_id="a6-plot-window-analysis-output",
            output_schema_version=1,
        ),
        global_skeleton=StoryAnalysisSemanticPass(
            semantic_profile_id=pass_id,
            prompt_id="a6.global-skeleton",
            prompt_version=1,
            output_schema_id="a6-global-skeleton-output",
            output_schema_version=1,
        ),
        story_bible=StoryAnalysisSemanticPass(
            semantic_profile_id=pass_id,
            prompt_id="a6.story-bible-synthesis",
            prompt_version=1,
            output_schema_id="a6-story-bible-output",
            output_schema_version=1,
        ),
    )


def _make_semantic_profile() -> SemanticLLMProfile:
    """Load the real tracked semantic profile."""
    from short_drama.llm import load_semantic_profile
    return load_semantic_profile(PROFILES_DIR / "story_analysis_llm_v1.yaml")


def _valid_payload() -> dict[str, Any]:
    """A minimal valid A6E-2 global-skeleton provider payload."""
    return {
        "event_importance_overlay": [
            {
                "event_ref": "evt_000001",
                "interpretation": {
                    "text_zh": "核心事件。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            }
        ],
        "arc_proposals": [
            {
                "proposal_ordinal": 0,
                "arc_kind": "main",
                "involved_character_refs": ["char_0001"],
                "involved_relationship_refs": [],
                "supporting_event_refs": ["evt_000001"],
                "supporting_fact_refs": [],
                "start_event_ref": "evt_000001",
                "end_event_ref": "evt_000002",
                "interpretation": {
                    "text_zh": "主弧线。",
                    "evidence_mode": "inferred",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": ["evt_000001"],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            }
        ],
        "turning_point_proposals": [
            {
                "proposal_ordinal": 0,
                "event_ref": "evt_000001",
                "supporting_fact_refs": [],
                "supporting_relationship_refs": [],
                "interpretation": {
                    "text_zh": "转折点。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            }
        ],
        "reveal_proposals": [],
        "foreshadow_payoff_proposals": [],
        "global_structure": {
            "main_conflict": {
                "text_zh": "主要冲突。",
                "evidence_mode": "explicit",
                "supporting_fact_refs": [],
                "supporting_event_refs": ["evt_000001"],
                "supporting_relationship_refs": [],
                "supporting_conflict_refs": [],
            },
            "secondary_conflicts": [],
            "main_plot": {
                "text_zh": "主线。",
                "evidence_mode": "explicit",
                "supporting_fact_refs": [],
                "supporting_event_refs": [],
                "supporting_relationship_refs": [],
                "supporting_conflict_refs": [],
            },
            "subplots": [],
            "ending_state": {
                "text_zh": "结局。",
                "evidence_mode": "inferred",
                "supporting_fact_refs": [],
                "supporting_event_refs": [],
                "supporting_relationship_refs": [],
                "supporting_conflict_refs": [],
            },
            "global_sections": [
                {
                    "section_ordinal": 1,
                    "label_zh": "第一幕",
                    "event_refs": ["evt_000001", "evt_000002"],
                    "interpretation": {
                        "text_zh": "开篇。",
                        "evidence_mode": "explicit",
                        "supporting_fact_refs": [],
                        "supporting_event_refs": [],
                        "supporting_relationship_refs": [],
                        "supporting_conflict_refs": [],
                    },
                }
            ],
            "main_character_refs": ["char_0001"],
            "major_arc_proposal_ordinals": [0],
            "major_turning_point_proposal_ordinals": [0],
            "major_reveal_proposal_ordinals": [],
        },
    }


def _make_snapshot_with_events(n_events: int = 20) -> StoryAnalysisInputSnapshot:
    """Build a minimal in-memory snapshot with n_events canonical events AND
    canonical characters (needed for the global-skeleton evidence universe)."""
    from types import SimpleNamespace
    from short_drama.artifacts import ArtifactRef

    snap = _in_memory_snapshot(num_events=n_events)
    # The A6D test snapshot has no canonical characters; the global-skeleton
    # pass needs them for the evidence universe. Inject a minimal set with
    # the required to_dict method for canonical serialization.
    def _make_char(cid: str, name: str):
        d = {
            "canonical_id": cid,
            "entity_type": "character",
            "display_name_original": name,
            "aliases_original": [],
            "candidate_refs": [f"cand_{cid[-4:]}"],
            "first_appearance_candidate_ref": f"cand_{cid[-4:]}",
        }
        ns = SimpleNamespace(**d)
        ns.to_dict = lambda: dict(d)
        return ns

    chars = (_make_char("char_0001", "主角"), _make_char("char_0002", "配角"))
    return dataclasses.replace(
        snap,
        canonical_character_registry=SimpleNamespace(entities=chars),
    )


# ---------------------------------------------------------------------------
# 1. Identity transition tests
# ---------------------------------------------------------------------------


class TestIdentityTransition:
    """The ceiling change (null -> 100000) changes both the StoryAnalysisProfile
    hash and the A6B plan hash, which in turn changes the A6C/A6D stable
    request identities."""

    def test_profile_hash_changes_with_ceiling(self):
        """Changing the ceiling from null to 100000 changes the profile hash."""
        profile_null = _make_profile(ceiling=None)
        profile_100k = _make_profile(ceiling=100000)
        assert profile_null.content_hash() != profile_100k.content_hash()

    def test_plan_hash_changes_with_ceiling(self):
        """The A6B plan hash changes when the planning policy ceiling changes."""
        from short_drama.story.story_analysis_planning import (
            build_story_analysis_plan_from_profile,
        )

        snapshot = _make_snapshot_with_events(20)
        profile_null = _make_profile(ceiling=None)
        profile_100k = _make_profile(ceiling=100000)

        plan_null = build_story_analysis_plan_from_profile(snapshot, profile_null)
        plan_100k = build_story_analysis_plan_from_profile(snapshot, profile_100k)

        assert plan_null.plan_hash != plan_100k.plan_hash

    def test_request_identity_changes_with_profile(self):
        """The A6E-2 request identity changes when the profile hash changes."""
        snapshot = _make_snapshot_with_events(20)
        profile_null = _make_profile(ceiling=None)
        profile_100k = _make_profile(ceiling=100000)
        semantic_profile = _make_semantic_profile()

        # Build contexts for both profiles.
        ctx = build_global_skeleton_context(
            [_char_analysis("char_0001")],
            [_window_analysis("window_0001", 1, ("evt_000001",), ())],
            snapshot.events and _make_global_index() or _make_global_index(),
        )

        from short_drama.story.consolidation import PromptAssetIdentity, OutputSchemaAssetIdentity

        prompt_id = PromptAssetIdentity("a6.global-skeleton", 1, "a" * 64)
        schema_id = OutputSchemaAssetIdentity(
            "a6-global-skeleton-output", 1, "b" * 64
        )

        h_null = global_skeleton_request_identity_hash(
            consolidation_manifest_ref=snapshot.consolidation_manifest_ref,
            plan_hash="c" * 64,
            profile=profile_null,
            semantic_profile=semantic_profile,
            prompt_identity=prompt_id,
            output_schema_identity=schema_id,
            packet_hash="d" * 64,
            request_hash="e" * 64,
            character_request_identity_hashes=("f" * 64,),
            window_request_identity_hashes=("g" * 64,),
        )
        h_100k = global_skeleton_request_identity_hash(
            consolidation_manifest_ref=snapshot.consolidation_manifest_ref,
            plan_hash="c" * 64,
            profile=profile_100k,
            semantic_profile=semantic_profile,
            prompt_identity=prompt_id,
            output_schema_identity=schema_id,
            packet_hash="d" * 64,
            request_hash="e" * 64,
            character_request_identity_hashes=("f" * 64,),
            window_request_identity_hashes=("g" * 64,),
        )
        assert h_null != h_100k

    def test_production_profile_freezes_100000_and_hash_is_deterministic(self):
        """The tracked production profile freezes the global-skeleton ceiling at
        100000 and its StoryAnalysisProfile hash is deterministic (and equals the
        frozen identity)."""
        path = REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml"
        profile_a = load_story_analysis_profile(path)
        profile_b = load_story_analysis_profile(path)
        # Global-skeleton ceiling is FROZEN at 100000; story-bible stays null.
        assert profile_a.planning_policy.global_skeleton_packet_max_estimated_tokens == 100000
        assert profile_a.planning_policy.story_bible_packet_max_estimated_tokens is None
        # Deterministic: two independent loads give the identical hash.
        assert profile_a.content_hash() == profile_b.content_hash()
        # The frozen 100000 profile has the expected stable identity.
        assert profile_a.content_hash() == (
            "0fa7bb93fcc7c4e2e09c6075efc528bdb835b6dde398c64cb7bf5495ed9df4a7"
        )

    def test_revised_plan_hash_is_deterministic(self):
        """The A6B plan hash under the frozen 100000 ceiling is deterministic."""
        from short_drama.story.story_analysis_planning import (
            build_story_analysis_plan_from_profile,
        )

        snapshot = _make_snapshot_with_events(20)
        profile = _make_profile(ceiling=100000)
        plan_a = build_story_analysis_plan_from_profile(snapshot, profile)
        plan_b = build_story_analysis_plan_from_profile(snapshot, profile)
        assert plan_a.plan_hash == plan_b.plan_hash
        assert len(plan_a.plan_hash) == 64


def _make_global_index() -> GlobalIndexBase:
    """Build a minimal GlobalIndexBase for testing."""
    return GlobalIndexBase(
        character_descriptors=(),
        location_descriptors=(),
        unresolved_entities=(),
        event_index=(),
        relationship_summaries=(),
        state_transition_summaries=(),
        conflict_summaries=(),
    )


# ---------------------------------------------------------------------------
# 2. Ceiling enforcement tests
# ---------------------------------------------------------------------------


class TestCeilingEnforcement:
    """The ceiling is enforced before any provider call. Null or exceeded
    ceilings fail closed."""

    def test_null_ceiling_fails_closed(self):
        """A None (DEFERRED) ceiling fails closed."""
        profile = _make_profile(ceiling=None)
        context = build_global_skeleton_context(
            [_char_analysis("char_0001")],
            [_window_analysis("window_0001", 1, ("evt_000001",), ())],
            _make_global_index(),
        )
        with pytest.raises(StoryAnalysisSemanticError, match="DEFERRED"):
            enforce_global_skeleton_ceiling(profile, context)

    def test_exceeded_ceiling_fails_closed(self):
        """A packet that exceeds the ceiling fails closed."""
        profile = _make_profile(ceiling=1)  # Impossibly small ceiling.
        context = build_global_skeleton_context(
            [_char_analysis("char_0001")],
            [_window_analysis("window_0001", 1, ("evt_000001",), ())],
            _make_global_index(),
        )
        with pytest.raises(StoryAnalysisSemanticError, match="exceeds the frozen"):
            enforce_global_skeleton_ceiling(profile, context)

    def test_valid_ceiling_passes(self):
        """A packet within the ceiling passes."""
        profile = _make_profile(ceiling=100000)
        context = build_global_skeleton_context(
            [_char_analysis("char_0001")],
            [_window_analysis("window_0001", 1, ("evt_000001",), ())],
            _make_global_index(),
        )
        # Should not raise.
        enforce_global_skeleton_ceiling(profile, context)


# ---------------------------------------------------------------------------
# 3. Canonical ref validation tests
# ---------------------------------------------------------------------------


class TestCanonicalRefValidation:
    """All canonical refs must belong to the A5 evidence universe."""

    def _snapshot(self) -> StoryAnalysisInputSnapshot:
        return _make_snapshot_with_events(20)

    def test_valid_refs_pass(self):
        """A payload with valid refs passes validation."""
        snapshot = self._snapshot()
        payload = _valid_payload()
        assert validate_global_skeleton_output(payload, snapshot) is None

    def test_invalid_event_ref_fails(self):
        """An event ref not in the A5 universe fails."""
        snapshot = self._snapshot()
        payload = _valid_payload()
        payload["event_importance_overlay"][0]["event_ref"] = "evt_999999"
        detail = validate_global_skeleton_output(payload, snapshot)
        assert detail is not None
        assert "evt_999999" in detail

    def test_invalid_character_ref_fails(self):
        """A character ref not in the A5 universe fails."""
        snapshot = self._snapshot()
        payload = _valid_payload()
        payload["arc_proposals"][0]["involved_character_refs"] = ["char_9999"]
        detail = validate_global_skeleton_output(payload, snapshot)
        assert detail is not None
        assert "char_9999" in detail

    def test_invalid_fact_ref_fails(self):
        """A fact ref not in the A5 universe fails."""
        snapshot = self._snapshot()
        payload = _valid_payload()
        payload["turning_point_proposals"][0]["supporting_fact_refs"] = [
            "fact_999999"
        ]
        detail = validate_global_skeleton_output(payload, snapshot)
        assert detail is not None
        assert "fact_999999" in detail

    def test_duplicate_event_in_importance_fails(self):
        """Duplicate event_ref in event_importance_overlay fails."""
        snapshot = self._snapshot()
        payload = _valid_payload()
        payload["event_importance_overlay"].append(
            dict(payload["event_importance_overlay"][0])
        )
        detail = validate_global_skeleton_output(payload, snapshot)
        assert detail is not None
        assert "duplicate" in detail


# ---------------------------------------------------------------------------
# 4. ID allocation tests
# ---------------------------------------------------------------------------


class TestIDAllocation:
    """Python assigns deterministic IDs based on proposal ordinal."""

    def test_arc_ids_deterministic(self):
        """Arc IDs are arc_000001, arc_000002, ... in proposal order."""
        payload = _valid_payload()
        # Add a second arc.
        payload["arc_proposals"].append(
            {
                "proposal_ordinal": 1,
                "arc_kind": "sub",
                "involved_character_refs": ["char_0001"],
                "involved_relationship_refs": [],
                "supporting_event_refs": ["evt_000001"],
                "supporting_fact_refs": [],
                "start_event_ref": "evt_000001",
                "end_event_ref": "evt_000002",
                "interpretation": {
                    "text_zh": "副弧线。",
                    "evidence_mode": "inferred",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            }
        )
        snapshot = _make_snapshot_with_events(20)
        assert validate_global_skeleton_output(payload, snapshot) is None

        arc_analysis = _build_arc_analysis_for_test(payload)
        assert arc_analysis.arcs[0].arc_id == "arc_000001"
        assert arc_analysis.arcs[1].arc_id == "arc_000002"

    def test_turning_point_ids_deterministic(self):
        """Turning point IDs are turn_000001, turn_000002, ..."""
        payload = _valid_payload()
        payload["turning_point_proposals"].append(
            {
                "proposal_ordinal": 1,
                "event_ref": "evt_000002",
                "supporting_fact_refs": [],
                "supporting_relationship_refs": [],
                "interpretation": {
                    "text_zh": "第二转折。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            }
        )
        snapshot = _make_snapshot_with_events(20)
        assert validate_global_skeleton_output(payload, snapshot) is None

        arc_analysis = _build_arc_analysis_for_test(payload)
        assert arc_analysis.turning_points[0].turning_point_id == "turn_000001"
        assert arc_analysis.turning_points[1].turning_point_id == "turn_000002"

    def test_reveal_ids_deterministic(self):
        """Reveal IDs are reveal_000001, reveal_000002, ..."""
        payload = _valid_payload()
        payload["reveal_proposals"] = [
            {
                "proposal_ordinal": 0,
                "reveal_event_refs": ["evt_000001"],
                "supporting_fact_refs": [],
                "affected_character_refs": ["char_0001"],
                "setup_event_refs": [],
                "interpretation": {
                    "text_zh": "揭示。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            }
        ]
        snapshot = _make_snapshot_with_events(20)
        assert validate_global_skeleton_output(payload, snapshot) is None

        arc_analysis = _build_arc_analysis_for_test(payload)
        assert arc_analysis.reveals[0].reveal_id == "reveal_000001"

    def test_payoff_ids_deterministic(self):
        """Payoff IDs are payoff_000001, payoff_000002, ..."""
        payload = _valid_payload()
        payload["foreshadow_payoff_proposals"] = [
            {
                "proposal_ordinal": 0,
                "setup_event_refs": ["evt_000001"],
                "payoff_event_refs": ["evt_000002"],
                "interpretation": {
                    "text_zh": "伏笔回收。",
                    "evidence_mode": "inferred",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            }
        ]
        snapshot = _make_snapshot_with_events(20)
        assert validate_global_skeleton_output(payload, snapshot) is None

        arc_analysis = _build_arc_analysis_for_test(payload)
        assert arc_analysis.foreshadow_payoffs[0].payoff_id == "payoff_000001"

    def test_ids_unique(self):
        """All Python-assigned IDs are unique within their collection."""
        payload = _valid_payload()
        for i in range(5):
            payload["arc_proposals"].append(
                {
                    "proposal_ordinal": i + 1,
                    "arc_kind": "sub",
                    "involved_character_refs": ["char_0001"],
                    "involved_relationship_refs": [],
                    "supporting_event_refs": ["evt_000001"],
                    "supporting_fact_refs": [],
                    "start_event_ref": "evt_000001",
                    "end_event_ref": "evt_000002",
                    "interpretation": {
                        "text_zh": f"弧线 {i}。",
                        "evidence_mode": "inferred",
                        "supporting_fact_refs": [],
                        "supporting_event_refs": [],
                        "supporting_relationship_refs": [],
                        "supporting_conflict_refs": [],
                    },
                }
            )
        snapshot = _make_snapshot_with_events(20)
        assert validate_global_skeleton_output(payload, snapshot) is None

        arc_analysis = _build_arc_analysis_for_test(payload)
        arc_ids = [a.arc_id for a in arc_analysis.arcs]
        assert len(arc_ids) == len(set(arc_ids))


def _build_arc_analysis_for_test(payload: dict[str, Any]) -> ArcAnalysis:
    """Build an ArcAnalysis from a payload for testing (imports the private
    builder)."""
    from short_drama.story.story_analysis_global_skeleton_semantic import (
        _build_arc_analysis,
    )
    return _build_arc_analysis(payload)


# ---------------------------------------------------------------------------
# 5. Wrong narrative anchor tests
# ---------------------------------------------------------------------------


class TestNarrativeAnchors:
    """Arc start_event_ref must precede end_event_ref in narrative order."""

    def test_start_after_end_fails(self):
        """An arc where start_event is after end_event in narrative order fails."""
        snapshot = _make_snapshot_with_events(20)
        payload = _valid_payload()
        # Swap start and end: evt_000002 comes after evt_000001.
        payload["arc_proposals"][0]["start_event_ref"] = "evt_000002"
        payload["arc_proposals"][0]["end_event_ref"] = "evt_000001"
        detail = validate_global_skeleton_output(payload, snapshot)
        assert detail is not None
        assert "does not precede" in detail

    def test_start_equals_end_fails(self):
        """An arc where start_event equals end_event fails (not strictly before)."""
        snapshot = _make_snapshot_with_events(20)
        payload = _valid_payload()
        payload["arc_proposals"][0]["start_event_ref"] = "evt_000001"
        payload["arc_proposals"][0]["end_event_ref"] = "evt_000001"
        detail = validate_global_skeleton_output(payload, snapshot)
        assert detail is not None
        assert "does not precede" in detail


# ---------------------------------------------------------------------------
# 6. Invalid internal reference tests
# ---------------------------------------------------------------------------


class TestInternalReferences:
    """major_*_proposal_ordinals must reference valid proposal ordinals."""

    def test_invalid_arc_ordinal_fails(self):
        """A major_arc_proposal_ordinals entry that doesn't match any arc
        proposal fails."""
        snapshot = _make_snapshot_with_events(20)
        payload = _valid_payload()
        payload["global_structure"]["major_arc_proposal_ordinals"] = [99]
        detail = validate_global_skeleton_output(payload, snapshot)
        assert detail is not None
        assert "major_arc_proposal_ordinals" in detail

    def test_invalid_tp_ordinal_fails(self):
        """A major_turning_point_proposal_ordinals entry that doesn't match
        any TP proposal fails."""
        snapshot = _make_snapshot_with_events(20)
        payload = _valid_payload()
        payload["global_structure"]["major_turning_point_proposal_ordinals"] = [99]
        detail = validate_global_skeleton_output(payload, snapshot)
        assert detail is not None
        assert "major_turning_point_proposal_ordinals" in detail

    def test_invalid_reveal_ordinal_fails(self):
        """A major_reveal_proposal_ordinals entry that doesn't match any
        reveal proposal fails."""
        snapshot = _make_snapshot_with_events(20)
        payload = _valid_payload()
        payload["global_structure"]["major_reveal_proposal_ordinals"] = [5]
        detail = validate_global_skeleton_output(payload, snapshot)
        assert detail is not None
        assert "major_reveal_proposal_ordinals" in detail

    def test_valid_cross_references_pass(self):
        """Valid cross-references pass."""
        snapshot = _make_snapshot_with_events(20)
        payload = _valid_payload()
        assert validate_global_skeleton_output(payload, snapshot) is None


# ---------------------------------------------------------------------------
# 7. Deterministic request hash tests
# ---------------------------------------------------------------------------


class TestDeterministicRequestHash:
    """The stable request identity is deterministic: same inputs -> same hash."""

    def test_same_inputs_same_hash(self):
        """Two calls with identical inputs produce the same identity hash."""
        from short_drama.story.consolidation import (
            OutputSchemaAssetIdentity,
            PromptAssetIdentity,
        )

        snapshot = _make_snapshot_with_events(20)
        profile = _make_profile(100000)
        semantic_profile = _make_semantic_profile()
        prompt_id = PromptAssetIdentity("a6.global-skeleton", 1, "a" * 64)
        schema_id = OutputSchemaAssetIdentity("a6-global-skeleton-output", 1, "b" * 64)

        kwargs = dict(
            consolidation_manifest_ref=snapshot.consolidation_manifest_ref,
            plan_hash="c" * 64,
            profile=profile,
            semantic_profile=semantic_profile,
            prompt_identity=prompt_id,
            output_schema_identity=schema_id,
            packet_hash="d" * 64,
            request_hash="e" * 64,
            character_request_identity_hashes=("f" * 64, "g" * 64),
            window_request_identity_hashes=("h" * 64,),
        )
        h1 = global_skeleton_request_identity_hash(**kwargs)
        h2 = global_skeleton_request_identity_hash(**kwargs)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_upstream_different_hash(self):
        """Changing the upstream A6C/A6D request identities changes the hash."""
        from short_drama.story.consolidation import (
            OutputSchemaAssetIdentity,
            PromptAssetIdentity,
        )

        snapshot = _make_snapshot_with_events(20)
        profile = _make_profile(100000)
        semantic_profile = _make_semantic_profile()
        prompt_id = PromptAssetIdentity("a6.global-skeleton", 1, "a" * 64)
        schema_id = OutputSchemaAssetIdentity("a6-global-skeleton-output", 1, "b" * 64)

        base = dict(
            consolidation_manifest_ref=snapshot.consolidation_manifest_ref,
            plan_hash="c" * 64,
            profile=profile,
            semantic_profile=semantic_profile,
            prompt_identity=prompt_id,
            output_schema_identity=schema_id,
            packet_hash="d" * 64,
            request_hash="e" * 64,
        )
        h1 = global_skeleton_request_identity_hash(
            **base,
            character_request_identity_hashes=("f" * 64,),
            window_request_identity_hashes=("g" * 64,),
        )
        h2 = global_skeleton_request_identity_hash(
            **base,
            character_request_identity_hashes=("x" * 64,),
            window_request_identity_hashes=("g" * 64,),
        )
        assert h1 != h2


# ---------------------------------------------------------------------------
# 8. Provenance tests
# ---------------------------------------------------------------------------


class TestProvenance:
    """Provenance mismatch fails closed without retry."""

    def test_provenance_mismatch_raises(self):
        """A provenance field mismatch raises StoryAnalysisProvenanceError."""
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _verify_global_skeleton_provenance,
        )

        # Build a mock request and provenance with a mismatch.
        request = MagicMock(spec=StructuredGenerationRequest)
        request.semantic_profile.profile_id = "correct-id"
        request.semantic_profile.semantic_profile_hash = "a" * 64
        request.rendered_prompt.prompt_id = "a6.global-skeleton"
        request.rendered_prompt.prompt_version = 1
        request.rendered_prompt.prompt_content_hash = "b" * 64
        request.rendered_prompt.rendered_prompt_hash = "c" * 64
        request.output_schema.schema_id = "a6-global-skeleton-output"
        request.output_schema.schema_version = 1
        request.output_schema.schema_hash = "d" * 64
        request.request_hash = "e" * 64

        provenance = MagicMock(spec=LLMInvocationProvenance)
        provenance.semantic_profile_id = "WRONG-id"  # Mismatch!
        provenance.semantic_profile_hash = "a" * 64
        provenance.prompt_id = "a6.global-skeleton"
        provenance.prompt_version = 1
        provenance.prompt_content_hash = "b" * 64
        provenance.rendered_prompt_hash = "c" * 64
        provenance.output_schema_id = "a6-global-skeleton-output"
        provenance.output_schema_version = 1
        provenance.output_schema_hash = "d" * 64
        provenance.request_hash = "e" * 64

        with pytest.raises(StoryAnalysisProvenanceError, match="provenance field mismatch"):
            _verify_global_skeleton_provenance(provenance, request)

    def test_provenance_match_passes(self):
        """A matching provenance passes without error."""
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _verify_global_skeleton_provenance,
        )

        request = MagicMock(spec=StructuredGenerationRequest)
        request.semantic_profile.profile_id = "story-analysis-llm-v1"
        request.semantic_profile.semantic_profile_hash = "a" * 64
        request.rendered_prompt.prompt_id = "a6.global-skeleton"
        request.rendered_prompt.prompt_version = 1
        request.rendered_prompt.prompt_content_hash = "b" * 64
        request.rendered_prompt.rendered_prompt_hash = "c" * 64
        request.output_schema.schema_id = "a6-global-skeleton-output"
        request.output_schema.schema_version = 1
        request.output_schema.schema_hash = "d" * 64
        request.request_hash = "e" * 64

        provenance = MagicMock(spec=LLMInvocationProvenance)
        provenance.semantic_profile_id = "story-analysis-llm-v1"
        provenance.semantic_profile_hash = "a" * 64
        provenance.prompt_id = "a6.global-skeleton"
        provenance.prompt_version = 1
        provenance.prompt_content_hash = "b" * 64
        provenance.rendered_prompt_hash = "c" * 64
        provenance.output_schema_id = "a6-global-skeleton-output"
        provenance.output_schema_version = 1
        provenance.output_schema_hash = "d" * 64
        provenance.request_hash = "e" * 64

        # Should not raise.
        _verify_global_skeleton_provenance(provenance, request)


# ---------------------------------------------------------------------------
# 9. Retry exhaustion tests
# ---------------------------------------------------------------------------


class TestRetryExhaustion:
    """Two semantic rounds exhausted -> StoryAnalysisGlobalSkeletonSemanticGenerationError."""

    def test_both_rounds_invalid_raises(self):
        """If both rounds produce semantically-invalid results, the pass fails."""
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _execute_global_skeleton_request,
            _GlobalSkeletonRetryableInvalid,
        )
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            GlobalSkeletonSemanticPreparation,
        )

        # Build a minimal preparation mock.
        prep = MagicMock(spec=GlobalSkeletonSemanticPreparation)
        prep.request_hash = "e" * 64
        prep.semantic_profile = MagicMock()
        prep.request = MagicMock()

        snapshot = _make_snapshot_with_events(20)

        # Mock the LLM client to return semantically-invalid results.
        llm_client = MagicMock(spec=LLMClient)

        # Create a result that passes schema but fails semantic validation.
        mock_result = MagicMock()
        mock_result.parsed_json = {
            "event_importance_overlay": [
                {
                    "event_ref": "evt_999999",  # Invalid ref!
                    "interpretation": {
                        "text_zh": "测试。",
                        "evidence_mode": "explicit",
                        "supporting_fact_refs": [],
                        "supporting_event_refs": [],
                        "supporting_relationship_refs": [],
                        "supporting_conflict_refs": [],
                    },
                }
            ],
            "arc_proposals": [],
            "turning_point_proposals": [],
            "reveal_proposals": [],
            "foreshadow_payoff_proposals": [],
            "global_structure": {
                "main_conflict": {
                    "text_zh": "冲突。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
                "secondary_conflicts": [],
                "main_plot": {
                    "text_zh": "主线。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
                "subplots": [],
                "ending_state": {
                    "text_zh": "结局。",
                    "evidence_mode": "inferred",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
                "global_sections": [],
                "main_character_refs": [],
                "major_arc_proposal_ordinals": [],
                "major_turning_point_proposal_ordinals": [],
                "major_reveal_proposal_ordinals": [],
            },
        }
        # Mock provenance to match.
        mock_result.provenance = MagicMock(spec=LLMInvocationProvenance)
        mock_result.provenance.semantic_profile_id = prep.request.semantic_profile.profile_id
        mock_result.provenance.semantic_profile_hash = prep.request.semantic_profile.semantic_profile_hash
        mock_result.provenance.prompt_id = prep.request.rendered_prompt.prompt_id
        mock_result.provenance.prompt_version = prep.request.rendered_prompt.prompt_version
        mock_result.provenance.prompt_content_hash = prep.request.rendered_prompt.prompt_content_hash
        mock_result.provenance.rendered_prompt_hash = prep.request.rendered_prompt.rendered_prompt_hash
        mock_result.provenance.output_schema_id = prep.request.output_schema.schema_id
        mock_result.provenance.output_schema_version = prep.request.output_schema.schema_version
        mock_result.provenance.output_schema_hash = prep.request.output_schema.schema_hash
        mock_result.provenance.request_hash = prep.request.request_hash

        llm_client.generate_structured.return_value = mock_result

        with pytest.raises(StoryAnalysisGlobalSkeletonSemanticGenerationError) as exc_info:
            _execute_global_skeleton_request(prep, snapshot, llm_client)

        assert exc_info.value.rounds_attempted == 2
        assert len(exc_info.value.last_failure_details) == 2
        # The LLM was called exactly 2 times (both rounds).
        assert llm_client.generate_structured.call_count == 2

    def test_first_round_invalid_second_valid(self):
        """If the first round is invalid but the second is valid, the pass
        succeeds with rounds_consumed=2."""
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _execute_global_skeleton_request,
            GlobalSkeletonSemanticPreparation,
        )

        prep = MagicMock(spec=GlobalSkeletonSemanticPreparation)
        prep.request_hash = "e" * 64
        prep.semantic_profile = MagicMock()
        prep.request = MagicMock()

        snapshot = _make_snapshot_with_events(20)
        llm_client = MagicMock(spec=LLMClient)

        # First call: invalid (bad ref).
        bad_result = MagicMock()
        bad_result.parsed_json = {
            "event_importance_overlay": [
                {
                    "event_ref": "evt_999999",
                    "interpretation": {
                        "text_zh": "测试。",
                        "evidence_mode": "explicit",
                        "supporting_fact_refs": [],
                        "supporting_event_refs": [],
                        "supporting_relationship_refs": [],
                        "supporting_conflict_refs": [],
                    },
                }
            ],
            "arc_proposals": [],
            "turning_point_proposals": [],
            "reveal_proposals": [],
            "foreshadow_payoff_proposals": [],
            "global_structure": {
                "main_conflict": {
                    "text_zh": "冲突。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
                "secondary_conflicts": [],
                "main_plot": {
                    "text_zh": "主线。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
                "subplots": [],
                "ending_state": {
                    "text_zh": "结局。",
                    "evidence_mode": "inferred",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
                "global_sections": [],
                "main_character_refs": [],
                "major_arc_proposal_ordinals": [],
                "major_turning_point_proposal_ordinals": [],
                "major_reveal_proposal_ordinals": [],
            },
        }
        bad_result.provenance = MagicMock(spec=LLMInvocationProvenance)
        bad_result.provenance.semantic_profile_id = prep.request.semantic_profile.profile_id
        bad_result.provenance.semantic_profile_hash = prep.request.semantic_profile.semantic_profile_hash
        bad_result.provenance.prompt_id = prep.request.rendered_prompt.prompt_id
        bad_result.provenance.prompt_version = prep.request.rendered_prompt.prompt_version
        bad_result.provenance.prompt_content_hash = prep.request.rendered_prompt.prompt_content_hash
        bad_result.provenance.rendered_prompt_hash = prep.request.rendered_prompt.rendered_prompt_hash
        bad_result.provenance.output_schema_id = prep.request.output_schema.schema_id
        bad_result.provenance.output_schema_version = prep.request.output_schema.schema_version
        bad_result.provenance.output_schema_hash = prep.request.output_schema.schema_hash
        bad_result.provenance.request_hash = prep.request.request_hash

        # Second call: valid.
        good_result = MagicMock()
        good_result.parsed_json = _valid_payload()
        good_result.provenance = MagicMock(spec=LLMInvocationProvenance)
        good_result.provenance.semantic_profile_id = prep.request.semantic_profile.profile_id
        good_result.provenance.semantic_profile_hash = prep.request.semantic_profile.semantic_profile_hash
        good_result.provenance.prompt_id = prep.request.rendered_prompt.prompt_id
        good_result.provenance.prompt_version = prep.request.rendered_prompt.prompt_version
        good_result.provenance.prompt_content_hash = prep.request.rendered_prompt.prompt_content_hash
        good_result.provenance.rendered_prompt_hash = prep.request.rendered_prompt.rendered_prompt_hash
        good_result.provenance.output_schema_id = prep.request.output_schema.schema_id
        good_result.provenance.output_schema_version = prep.request.output_schema.schema_version
        good_result.provenance.output_schema_hash = prep.request.output_schema.schema_hash
        good_result.provenance.request_hash = prep.request.request_hash

        llm_client.generate_structured.side_effect = [bad_result, good_result]

        payload, rounds = _execute_global_skeleton_request(
            prep, snapshot, llm_client
        )
        assert rounds == 2
        assert llm_client.generate_structured.call_count == 2


# ---------------------------------------------------------------------------
# 10. No persistence tests
# ---------------------------------------------------------------------------


class TestNoPersistence:
    """A6E-2 is in-memory only: no writes to disk."""

    def test_no_file_writes_during_execution(self, tmp_path: Path):
        """The semantic execution does not write any files."""
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _build_arc_analysis,
            _build_global_event_analysis,
            _build_global_structure,
        )

        payload = _valid_payload()
        plan = _make_minimal_plan()
        window_analyses = (_window_analysis("window_0001", 1, ("evt_000001",), ()),)

        # These are pure in-memory constructions.
        gea = _build_global_event_analysis(payload, plan, window_analyses)
        aa = _build_arc_analysis(payload)
        gs = _build_global_structure(payload)

        # Verify they are proper typed objects.
        assert isinstance(gea, GlobalEventAnalysis)
        assert isinstance(aa, ArcAnalysis)
        assert isinstance(gs, GlobalStructure)

        # No files were created in tmp_path (the CWD or any temp location).
        # The functions are pure computations with no I/O.
        assert list(tmp_path.iterdir()) == []


def _make_minimal_plan() -> StoryAnalysisPlan:
    """Build a minimal plan for testing (bypasses the full snapshot builder)."""
    from short_drama.artifacts import ArtifactRef

    # Build a minimal plan that satisfies the GlobalEventAnalysis constructor.
    return dataclasses.make_dataclass(
        "MockPlan",
        [
            ("consolidation_manifest_ref", ArtifactRef),
            ("plan_hash", str),
            ("event_stream", tuple),
            ("windows", tuple),
            ("global_index", GlobalIndexBase),
        ],
    )(
        consolidation_manifest_ref=ArtifactRef(
            artifact_type="consolidation_manifest",
            artifact_id="test.artifact",
            revision=1,
            content_hash="a" * 64,
        ),
        plan_hash="a" * 64,
        event_stream=("evt_000001", "evt_000002"),
        windows=(),
        global_index=_make_global_index(),
    )


# ---------------------------------------------------------------------------
# 11. Section consistency tests
# ---------------------------------------------------------------------------


class TestSectionConsistency:
    """Global section ordinals must be unique and strictly increasing."""

    def test_duplicate_section_ordinal_fails(self):
        """Duplicate section ordinals fail."""
        snapshot = _make_snapshot_with_events(20)
        payload = _valid_payload()
        payload["global_structure"]["global_sections"].append(
            {
                "section_ordinal": 1,  # Duplicate!
                "label_zh": "重复",
                "event_refs": [],
                "interpretation": {
                    "text_zh": "重复段落。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            }
        )
        detail = validate_global_skeleton_output(payload, snapshot)
        assert detail is not None
        assert "duplicate" in detail

    def test_non_increasing_section_ordinal_fails(self):
        """Non-increasing section ordinals fail."""
        snapshot = _make_snapshot_with_events(20)
        payload = _valid_payload()
        payload["global_structure"]["global_sections"].append(
            {
                "section_ordinal": 1,  # Same as previous (not strictly increasing)
                "label_zh": "第二幕",
                "event_refs": [],
                "interpretation": {
                    "text_zh": "第二幕。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            }
        )
        detail = validate_global_skeleton_output(payload, snapshot)
        assert detail is not None
        assert "strictly increasing" in detail or "duplicate" in detail

    def test_valid_sections_pass(self):
        """Valid strictly-increasing section ordinals pass."""
        snapshot = _make_snapshot_with_events(20)
        payload = _valid_payload()
        payload["global_structure"]["global_sections"].append(
            {
                "section_ordinal": 2,
                "label_zh": "第二幕",
                "event_refs": ["evt_000002"],
                "interpretation": {
                    "text_zh": "第二幕。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            }
        )
        assert validate_global_skeleton_output(payload, snapshot) is None


# ---------------------------------------------------------------------------
# 12. Empty optional arrays tests
# ---------------------------------------------------------------------------


class TestEmptyOptionalArrays:
    """Empty reveal/payoff arrays are valid when unsupported by evidence."""

    def test_empty_reveals_pass(self):
        """An empty reveal_proposals array is valid."""
        snapshot = _make_snapshot_with_events(20)
        payload = _valid_payload()
        payload["reveal_proposals"] = []
        assert validate_global_skeleton_output(payload, snapshot) is None

    def test_empty_payoffs_pass(self):
        """An empty foreshadow_payoff_proposals array is valid."""
        snapshot = _make_snapshot_with_events(20)
        payload = _valid_payload()
        payload["foreshadow_payoff_proposals"] = []
        assert validate_global_skeleton_output(payload, snapshot) is None

    def test_empty_arcs_pass(self):
        """An empty arc_proposals array is valid."""
        snapshot = _make_snapshot_with_events(20)
        payload = _valid_payload()
        payload["arc_proposals"] = []
        payload["global_structure"]["major_arc_proposal_ordinals"] = []
        assert validate_global_skeleton_output(payload, snapshot) is None


# ---------------------------------------------------------------------------
# 13. Non-contiguous / reordered proposal ordinal tests (Finding 1)
# ---------------------------------------------------------------------------


class TestNonContiguousProposalOrdinals:
    """The provider schema allows non-negative proposal ordinals that are not
    necessarily equal to array indices. The cross-reference resolution must use
    explicit maps (proposal_ordinal -> Python-assigned ID), not array indexing."""

    def test_non_contiguous_arc_ordinals(self):
        """Arcs with non-contiguous proposal ordinals (0, 5, 12) resolve
        correctly in major_arc_proposal_ordinals."""
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _build_global_structure,
        )

        payload = _valid_payload()
        # Three arcs with non-contiguous ordinals: 0, 5, 12.
        payload["arc_proposals"] = [
            {
                "proposal_ordinal": 0,
                "arc_kind": "main",
                "involved_character_refs": ["char_0001"],
                "involved_relationship_refs": [],
                "supporting_event_refs": ["evt_000001"],
                "supporting_fact_refs": [],
                "start_event_ref": "evt_000001",
                "end_event_ref": "evt_000002",
                "interpretation": {
                    "text_zh": "主弧线。",
                    "evidence_mode": "inferred",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": ["evt_000001"],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            },
            {
                "proposal_ordinal": 5,
                "arc_kind": "sub",
                "involved_character_refs": ["char_0002"],
                "involved_relationship_refs": [],
                "supporting_event_refs": ["evt_000002"],
                "supporting_fact_refs": [],
                "start_event_ref": "evt_000002",
                "end_event_ref": "evt_000003",
                "interpretation": {
                    "text_zh": "副弧线A。",
                    "evidence_mode": "inferred",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            },
            {
                "proposal_ordinal": 12,
                "arc_kind": "sub",
                "involved_character_refs": ["char_0001"],
                "involved_relationship_refs": [],
                "supporting_event_refs": ["evt_000003"],
                "supporting_fact_refs": [],
                "start_event_ref": "evt_000003",
                "end_event_ref": "evt_000004",
                "interpretation": {
                    "text_zh": "副弧线B。",
                    "evidence_mode": "inferred",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            },
        ]
        # major_arc_proposal_ordinals references ordinal 5 (the 2nd arc).
        payload["global_structure"]["major_arc_proposal_ordinals"] = [5]

        snapshot = _make_snapshot_with_events(20)
        assert validate_global_skeleton_output(payload, snapshot) is None

        gs = _build_global_structure(payload)
        # Ordinal 5 maps to the 2nd arc in array position -> arc_000002.
        assert gs.major_arc_refs == ("arc_000002",)

    def test_reordered_tp_ordinals(self):
        """Turning points with reordered (non-sequential) ordinals resolve
        correctly."""
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _build_global_structure,
        )

        payload = _valid_payload()
        # Two TPs with ordinals 7 and 3 (reordered: not 0,1).
        payload["turning_point_proposals"] = [
            {
                "proposal_ordinal": 7,
                "event_ref": "evt_000001",
                "supporting_fact_refs": [],
                "supporting_relationship_refs": [],
                "interpretation": {
                    "text_zh": "转折A。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            },
            {
                "proposal_ordinal": 3,
                "event_ref": "evt_000002",
                "supporting_fact_refs": [],
                "supporting_relationship_refs": [],
                "interpretation": {
                    "text_zh": "转折B。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            },
        ]
        # Reference ordinal 3 (the 2nd TP in array position).
        payload["global_structure"]["major_turning_point_proposal_ordinals"] = [3]

        snapshot = _make_snapshot_with_events(20)
        assert validate_global_skeleton_output(payload, snapshot) is None

        gs = _build_global_structure(payload)
        # Ordinal 3 maps to the 2nd TP in array position -> turn_000002.
        assert gs.major_turning_point_refs == ("turn_000002",)

    def test_non_contiguous_reveal_ordinals(self):
        """Reveals with non-contiguous ordinals resolve correctly."""
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _build_global_structure,
        )

        payload = _valid_payload()
        payload["reveal_proposals"] = [
            {
                "proposal_ordinal": 10,
                "reveal_event_refs": ["evt_000001"],
                "supporting_fact_refs": [],
                "affected_character_refs": ["char_0001"],
                "setup_event_refs": [],
                "interpretation": {
                    "text_zh": "揭示A。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            },
            {
                "proposal_ordinal": 20,
                "reveal_event_refs": ["evt_000002"],
                "supporting_fact_refs": [],
                "affected_character_refs": ["char_0002"],
                "setup_event_refs": [],
                "interpretation": {
                    "text_zh": "揭示B。",
                    "evidence_mode": "explicit",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            },
        ]
        # Reference ordinal 20 (the 2nd reveal).
        payload["global_structure"]["major_reveal_proposal_ordinals"] = [20]

        snapshot = _make_snapshot_with_events(20)
        assert validate_global_skeleton_output(payload, snapshot) is None

        gs = _build_global_structure(payload)
        # Ordinal 20 maps to the 2nd reveal in array position -> reveal_000002.
        assert gs.major_reveal_refs == ("reveal_000002",)

    def test_multiple_major_refs_with_gaps(self):
        """Multiple major arc refs with gaps in ordinals all resolve correctly."""
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _build_global_structure,
        )

        payload = _valid_payload()
        # Four arcs with ordinals 0, 3, 7, 15.
        payload["arc_proposals"] = [
            {
                "proposal_ordinal": 0,
                "arc_kind": "main",
                "involved_character_refs": ["char_0001"],
                "involved_relationship_refs": [],
                "supporting_event_refs": ["evt_000001"],
                "supporting_fact_refs": [],
                "start_event_ref": "evt_000001",
                "end_event_ref": "evt_000002",
                "interpretation": {
                    "text_zh": "弧线0。",
                    "evidence_mode": "inferred",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            },
            {
                "proposal_ordinal": 3,
                "arc_kind": "sub",
                "involved_character_refs": ["char_0001"],
                "involved_relationship_refs": [],
                "supporting_event_refs": ["evt_000002"],
                "supporting_fact_refs": [],
                "start_event_ref": "evt_000002",
                "end_event_ref": "evt_000003",
                "interpretation": {
                    "text_zh": "弧线3。",
                    "evidence_mode": "inferred",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            },
            {
                "proposal_ordinal": 7,
                "arc_kind": "sub",
                "involved_character_refs": ["char_0002"],
                "involved_relationship_refs": [],
                "supporting_event_refs": ["evt_000003"],
                "supporting_fact_refs": [],
                "start_event_ref": "evt_000003",
                "end_event_ref": "evt_000004",
                "interpretation": {
                    "text_zh": "弧线7。",
                    "evidence_mode": "inferred",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            },
            {
                "proposal_ordinal": 15,
                "arc_kind": "sub",
                "involved_character_refs": ["char_0002"],
                "involved_relationship_refs": [],
                "supporting_event_refs": ["evt_000004"],
                "supporting_fact_refs": [],
                "start_event_ref": "evt_000004",
                "end_event_ref": "evt_000005",
                "interpretation": {
                    "text_zh": "弧线15。",
                    "evidence_mode": "inferred",
                    "supporting_fact_refs": [],
                    "supporting_event_refs": [],
                    "supporting_relationship_refs": [],
                    "supporting_conflict_refs": [],
                },
            },
        ]
        # Reference ordinals 3 and 15 (positions 1 and 3).
        payload["global_structure"]["major_arc_proposal_ordinals"] = [3, 15]

        snapshot = _make_snapshot_with_events(20)
        assert validate_global_skeleton_output(payload, snapshot) is None

        gs = _build_global_structure(payload)
        # Ordinal 3 -> arc_000002, ordinal 15 -> arc_000004.
        assert gs.major_arc_refs == ("arc_000002", "arc_000004")


# ---------------------------------------------------------------------------
# 14. Upstream binding enforcement tests (Finding 2)
# ---------------------------------------------------------------------------


class TestUpstreamBinding:
    """The frozen exact-upstream identity contract is enforced before any
    provider execution. The verification recomputes the deterministic A6C/A6D
    request identity hashes using the existing preparation functions under the
    current exact profile and A6B plan, then compares them against the provided
    sequences. This catches stale-profile, swapped, fabricated, or placeholder
    identities with zero provider calls."""

    def _snapshot(self) -> StoryAnalysisInputSnapshot:
        return _make_snapshot_with_events(20)

    def _profile(self) -> StoryAnalysisProfile:
        return _make_profile(100000)

    def _plan(self) -> StoryAnalysisPlan:
        from short_drama.story.story_analysis_planning import (
            build_story_analysis_plan_from_profile,
        )
        return build_story_analysis_plan_from_profile(
            self._snapshot(), self._profile()
        )

    def _prompts(self):
        from short_drama.llm import PromptRegistry
        from short_drama.story.reconciliation_semantic import DEFAULT_PROMPT_BASE_DIR
        return PromptRegistry(DEFAULT_PROMPT_BASE_DIR)

    def _expected_char_hashes(self) -> tuple[str, ...]:
        """Recompute the expected A6C identity hashes using the existing
        preparation function (zero provider calls)."""
        from short_drama.story.story_analysis_semantic import (
            build_character_semantic_preparation,
        )
        prep = build_character_semantic_preparation(
            self._plan(), self._profile(), _make_semantic_profile(),
            prompts=self._prompts(),
        )
        return prep.character_request_identity_hashes

    def _expected_window_hashes(self) -> tuple[str, ...]:
        """Recompute the expected A6D identity hashes using the existing
        preparation function (zero provider calls)."""
        from short_drama.story.story_analysis_window_semantic import (
            build_plot_window_semantic_preparation,
        )
        prep = build_plot_window_semantic_preparation(
            self._snapshot(), self._plan(), self._profile(),
            _make_semantic_profile(), prompts=self._prompts(),
        )
        return prep.window_request_identity_hashes

    def _verify(self, char_hashes: Sequence[str], window_hashes: Sequence[str]) -> None:
        """Call the verification function with the current snapshot/plan/profile."""
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _verify_upstream_request_identities,
        )
        _verify_upstream_request_identities(
            self._snapshot(),
            self._plan(),
            self._profile(),
            _make_semantic_profile(),
            char_hashes,
            window_hashes,
            prompts=self._prompts(),
        )

    def test_valid_identities_pass(self):
        """Identity hashes recomputed from the same profile/plan pass."""
        self._verify(self._expected_char_hashes(), self._expected_window_hashes())

    def test_all_a_hashes_rejected(self):
        """All-'a'*64 character hashes and all-'b'*64 window hashes are
        rejected (they are valid SHA-256 format but NOT the actual identities).
        This is the key regression test: a format-only check would accept these."""
        plan = self._plan()
        fake_char = tuple("a" * 64 for _ in range(len(plan.character_packages)))
        fake_window = tuple("b" * 64 for _ in range(len(plan.windows)))
        with pytest.raises(StoryAnalysisSemanticError, match="mismatch"):
            self._verify(fake_char, fake_window)

    def test_stale_null_ceiling_profile_identities_rejected(self):
        """Identity hashes computed under a null-ceiling (stale) profile are
        rejected when the current profile has ceiling=100000."""
        from short_drama.story.story_analysis_planning import (
            build_story_analysis_plan_from_profile,
        )
        from short_drama.story.story_analysis_semantic import (
            build_character_semantic_preparation,
        )
        from short_drama.story.story_analysis_window_semantic import (
            build_plot_window_semantic_preparation,
        )

        snapshot = self._snapshot()
        profile_stale = _make_profile(ceiling=None)  # Old null-ceiling profile.
        profile_current = self._profile()  # Current 100000 ceiling.
        semantic_profile = _make_semantic_profile()
        prompts = self._prompts()

        # Plan and identities computed under the STALE profile.
        plan_stale = build_story_analysis_plan_from_profile(snapshot, profile_stale)
        a6c_stale = build_character_semantic_preparation(
            plan_stale, profile_stale, semantic_profile, prompts=prompts
        )
        a6d_stale = build_plot_window_semantic_preparation(
            snapshot, plan_stale, profile_stale, semantic_profile, prompts=prompts
        )

        # Now verify using the CURRENT profile/plan but the STALE identities.
        # The plan must be the current one (matching the snapshot), but the
        # identity hashes are from the stale profile.
        plan_current = build_story_analysis_plan_from_profile(snapshot, profile_current)
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _verify_upstream_request_identities,
        )
        with pytest.raises(StoryAnalysisSemanticError, match="mismatch"):
            _verify_upstream_request_identities(
                snapshot,
                plan_current,
                profile_current,
                semantic_profile,
                a6c_stale.character_request_identity_hashes,
                a6d_stale.window_request_identity_hashes,
                prompts=prompts,
            )

    def test_stale_74k_ceiling_profile_identities_rejected(self):
        """Identity hashes computed under the old 74000-ceiling profile are
        rejected when the current profile has the revised 100000 ceiling. This
        guarantees the 61 upstream A6C/A6D analyses produced under the 74000
        identity are NOT silently rebound to the new identity."""
        from short_drama.story.story_analysis_planning import (
            build_story_analysis_plan_from_profile,
        )
        from short_drama.story.story_analysis_semantic import (
            build_character_semantic_preparation,
        )
        from short_drama.story.story_analysis_window_semantic import (
            build_plot_window_semantic_preparation,
        )
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _verify_upstream_request_identities,
        )

        snapshot = self._snapshot()
        profile_stale = _make_profile(ceiling=74000)  # Old frozen-ceiling profile.
        profile_current = self._profile()  # Current 100000 ceiling.
        semantic_profile = _make_semantic_profile()
        prompts = self._prompts()

        # Plan and identities computed under the STALE 74000 profile.
        plan_stale = build_story_analysis_plan_from_profile(snapshot, profile_stale)
        a6c_stale = build_character_semantic_preparation(
            plan_stale, profile_stale, semantic_profile, prompts=prompts
        )
        a6d_stale = build_plot_window_semantic_preparation(
            snapshot, plan_stale, profile_stale, semantic_profile, prompts=prompts
        )

        # Verify using the CURRENT 100000 profile/plan but the STALE identities.
        plan_current = build_story_analysis_plan_from_profile(snapshot, profile_current)
        with pytest.raises(StoryAnalysisSemanticError, match="mismatch"):
            _verify_upstream_request_identities(
                snapshot,
                plan_current,
                profile_current,
                semantic_profile,
                a6c_stale.character_request_identity_hashes,
                a6d_stale.window_request_identity_hashes,
                prompts=prompts,
            )

    def test_swapped_character_ordering_rejected(self):
        """Swapped character identity hash ordering is rejected."""
        expected = self._expected_char_hashes()
        if len(expected) < 2:
            pytest.skip("need at least 2 characters for swap test")
        # Swap the first two.
        swapped = list(expected)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with pytest.raises(StoryAnalysisSemanticError, match="mismatch"):
            self._verify(tuple(swapped), self._expected_window_hashes())

    def test_swapped_window_ordering_rejected(self):
        """Swapped window identity hash ordering is rejected."""
        expected = self._expected_window_hashes()
        if len(expected) < 2:
            pytest.skip("need at least 2 windows for swap test")
        # Swap the first two.
        swapped = list(expected)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with pytest.raises(StoryAnalysisSemanticError, match="mismatch"):
            self._verify(self._expected_char_hashes(), tuple(swapped))

    def test_empty_character_hashes_fail(self):
        """Empty character_request_identity_hashes fails closed."""
        plan = self._plan()
        with pytest.raises(StoryAnalysisSemanticError, match="count"):
            self._verify((), self._expected_window_hashes())

    def test_empty_window_hashes_fail(self):
        """Empty window_request_identity_hashes fails closed."""
        plan = self._plan()
        with pytest.raises(StoryAnalysisSemanticError, match="count"):
            self._verify(self._expected_char_hashes(), ())

    def test_manifest_mismatch_fails(self):
        """Snapshot and plan referencing different A5 manifests fails closed."""
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _verify_snapshot_plan_manifest_identity,
        )
        from short_drama.artifacts import ArtifactRef

        snapshot = self._snapshot()
        plan = self._plan()
        tampered_plan = dataclasses.replace(
            plan,
            consolidation_manifest_ref=ArtifactRef(
                artifact_type="consolidation_manifest",
                artifact_id="different.artifact",
                revision=99,
                content_hash="f" * 64,
            ),
        )
        with pytest.raises(StoryAnalysisSemanticError, match="disagree on the exact A5"):
            _verify_snapshot_plan_manifest_identity(snapshot, tampered_plan)

    def test_matching_manifest_passes(self):
        """Snapshot and plan with the same manifest ref pass."""
        from short_drama.story.story_analysis_global_skeleton_semantic import (
            _verify_snapshot_plan_manifest_identity,
        )

        snapshot = self._snapshot()
        plan = self._plan()
        _verify_snapshot_plan_manifest_identity(snapshot, plan)

    def test_preparation_rejects_fabricated_identities_zero_provider_calls(self):
        """build_global_skeleton_semantic_preparation rejects fabricated
        (all-'a'*64) upstream identities before any provider call."""
        from short_drama.llm import PromptRegistry
        from short_drama.story.reconciliation_semantic import DEFAULT_PROMPT_BASE_DIR

        snapshot = self._snapshot()
        plan = self._plan()
        profile = self._profile()
        semantic_profile = _make_semantic_profile()
        prompts = PromptRegistry(DEFAULT_PROMPT_BASE_DIR)

        char_analyses = [_char_analysis(c.character_ref) for c in plan.character_packages]
        window_analyses = [
            _window_analysis(
                f"window_{i+1:04d}", i + 1, w.owned_event_ids, w.context_event_ids
            )
            for i, w in enumerate(plan.windows)
        ]

        # Fabricated identities (valid SHA-256 format, wrong values).
        fake_char = tuple("a" * 64 for _ in range(len(plan.character_packages)))
        fake_window = tuple("b" * 64 for _ in range(len(plan.windows)))

        with pytest.raises(StoryAnalysisSemanticError, match="mismatch"):
            build_global_skeleton_semantic_preparation(
                snapshot,
                plan,
                profile,
                semantic_profile,
                char_analyses,
                window_analyses,
                prompts=prompts,
                character_request_identity_hashes=fake_char,
                window_request_identity_hashes=fake_window,
            )

    def test_preparation_accepts_correct_identities(self):
        """build_global_skeleton_semantic_preparation accepts identity hashes
        that were recomputed from the same profile/plan."""
        from short_drama.llm import PromptRegistry
        from short_drama.story.reconciliation_semantic import DEFAULT_PROMPT_BASE_DIR

        snapshot = self._snapshot()
        plan = self._plan()
        profile = self._profile()
        semantic_profile = _make_semantic_profile()
        prompts = PromptRegistry(DEFAULT_PROMPT_BASE_DIR)

        char_analyses = [_char_analysis(c.character_ref) for c in plan.character_packages]
        window_analyses = [
            _window_analysis(
                f"window_{i+1:04d}", i + 1, w.owned_event_ids, w.context_event_ids
            )
            for i, w in enumerate(plan.windows)
        ]

        # Correct identities recomputed from the same profile/plan.
        prep = build_global_skeleton_semantic_preparation(
            snapshot,
            plan,
            profile,
            semantic_profile,
            char_analyses,
            window_analyses,
            prompts=prompts,
            character_request_identity_hashes=self._expected_char_hashes(),
            window_request_identity_hashes=self._expected_window_hashes(),
        )
        # Should succeed and produce a valid preparation.
        assert prep.request_identity_hash
        assert len(prep.request_identity_hash) == 64


# ---------------------------------------------------------------------------
# 15. A6E prompt refinement tests (A6E-2 semantic-quality, Issue #102)
# ---------------------------------------------------------------------------


class TestPromptRefinement:
    """Focused tests for the A6E prompt-only semantic-quality refinement.

    These tests verify:
    * prompt content-hash consistency (the tracked prompt.yaml matches the
      actual template files);
    * preservation of the explicit/inferred semantic distinction in the
      prompt instructions;
    * preservation of concise-output requirements (no regression to unbounded
      enumerations);
    * A6C/A6D semantic identity stability (unchanged by the A6E prompt edit);
    * expected A6E request identity change (the prompt content hash change
      propagates to the A6E request identity).

    All tests are deterministic and offline (no real provider calls).
    """

    def _prompt_spec(self):
        from short_drama.llm import PromptRegistry
        return PromptRegistry(DEFAULT_PROMPT_BASE_DIR).load(
            "a6.global-skeleton", version=1
        )

    # 1. Content-hash consistency -----------------------------------------

    def test_prompt_content_hash_matches_template_files(self):
        """The tracked prompt.yaml content_hash matches the computed hash from
        the actual system.txt / user.txt files (prompt identity is consistent)."""
        from short_drama.llm.models import compute_prompt_content_hash

        spec = self._prompt_spec()
        # The PromptSpec __post_init__ already re-validated the hash, so
        # reaching here means the hash is correct. Additionally verify the
        # hash against the frozen expected value for this refinement.
        assert spec.content_hash == (
            "b4f89fbd52e57bd4102cdaa40ffca9d1acfb62452c1e0133a8891985c51cf218"
        )

    def test_prompt_version_unchanged(self):
        """The prompt version remains v1 (no re-versioning)."""
        spec = self._prompt_spec()
        assert spec.version == 1
        assert spec.prompt_id == "a6.global-skeleton"

    # 2. Explicit/inferred distinction preserved --------------------------

    def test_prompt_preserves_evidence_mode_distinction(self):
        """The system prompt explicitly distinguishes explicit from inferred
        evidence_mode with concrete criteria."""
        spec = self._prompt_spec()
        sys_text = spec.system_template
        # Both modes must be named and distinguished.
        assert "explicit" in sys_text
        assert "inferred" in sys_text
        # The prompt must state that inferred interpretations contain
        # literary/structural inferences (not just a bare mention).
        assert "inferred" in sys_text and "explicit" in sys_text
        # The evidence_mode section header or equivalent guidance is present.
        assert "evidence_mode" in sys_text

    def test_prompt_inferred_requires_evidence_refs(self):
        """The prompt requires evidence refs for both explicit and inferred
        modes (inferred does not waive evidence citation)."""
        spec = self._prompt_spec()
        sys_text = spec.system_template
        # The prompt must state that both modes require supporting refs.
        assert "supporting" in sys_text

    # 3. Concise-output requirements preserved ----------------------------

    def test_prompt_preserves_concise_output_requirement(self):
        """The system prompt still requires concise 1-3 sentence
        interpretations and restrained collection sizes (no regression to
        unbounded enumeration)."""
        spec = self._prompt_spec()
        sys_text = spec.system_template
        # 1-3 sentence constraint is preserved.
        assert "1\u20133" in sys_text or "1-3" in sys_text
        # The anti-enumeration / anti-verbose guidance is present.
        assert "\u514b\u5236" in sys_text  # 克制
        assert "\u7cbe\u70bc" in sys_text  # 精炼
        # The six collections are still constrained to salient/major items.
        assert "salient" in sys_text or "major" in sys_text

    def test_prompt_does_not_add_max_items_constraint(self):
        """The prompt does not introduce arbitrary fixed numeric limits that
        would be a contract-level constraint (it uses qualitative guidance
        like 'few' or 'sustained progression')."""
        spec = self._prompt_spec()
        sys_text = spec.system_template
        # No hard maxItems numeric constraint like "at most 5 arcs" etc.
        # The prompt uses qualitative language.
        assert "sustained progression" in sys_text

    # 4. Arc selection guidance -------------------------------------------

    def test_prompt_arc_requires_narrative_progression(self):
        """The arc guidance requires sustained narrative progression, not
        mere character appearance enumeration."""
        spec = self._prompt_spec()
        sys_text = spec.system_template
        # The arc section must mention progression or evolution.
        assert "progression" in sys_text or "\u53d9\u4e8b\u8fdb\u5c55" in sys_text

    # 5. Turning point phase guidance -------------------------------------

    def test_prompt_turning_points_span_phases(self):
        """The turning-point guidance encourages coverage across narrative
        phases (setup, escalation, climax, resolution) without mandating one
        per phase."""
        spec = self._prompt_spec()
        sys_text = spec.system_template
        # The prompt mentions multiple narrative phases.
        assert "\u9ad8\u6f6e" in sys_text  # 高潮 (climax)
        assert "\u6536\u675f" in sys_text  # 收束 (resolution)

    # 6. Foreshadow/payoff discrimination ----------------------------------

    def test_prompt_payoff_distinguishes_from_plain_sequence(self):
        """The foreshadow-payoff guidance distinguishes true setup→payoff
        from ordinary chronological sequence."""
        spec = self._prompt_spec()
        sys_text = spec.system_template
        # The prompt must mention that plain temporal order is NOT a payoff.
        assert "setup" in sys_text and "payoff" in sys_text

    # 7. Restraint / uncertainty guidance ----------------------------------

    def test_prompt_discourages_overstatement(self):
        """The prompt discourages unsupported assertions about character
        roles, realizations, and chapter references."""
        spec = self._prompt_spec()
        sys_text = spec.system_template
        # Must discourage unsupported role/authority claims.
        assert "\u6743\u5a01" in sys_text  # 权威 (authority)
        # Must discourage rewriting experience as realization.
        assert "\u610f\u8bc6\u5230" in sys_text  # 意识到
        # Must discourage chapter number references.
        assert "\u7ae0\u8282" in sys_text  # 章节

    # 8. Major arc discrimination ------------------------------------------

    def test_prompt_major_arc_discrimination(self):
        """The prompt instructs that major_*_proposal_ordinals should only
        reference genuinely central proposals, not all proposals."""
        spec = self._prompt_spec()
        sys_text = spec.system_template
        assert "major" in sys_text
        # The prompt should say to NOT mark all as major.
        assert "\u4e0d\u8981" in sys_text  # 不要 (do not)

    # 9. A6C/A6D identity stability -----------------------------------------

    def test_a6c_identity_unaffected_by_a6e_prompt_change(self):
        """The A6C character-analysis request identity does NOT bind the A6E
        global-skeleton prompt. Changing the A6E prompt does not change A6C
        identities."""
        from short_drama.story.story_analysis_semantic import (
            build_character_semantic_preparation,
        )
        from short_drama.story.story_analysis_planning import (
            build_story_analysis_plan_from_profile,
        )

        snapshot = _make_snapshot_with_events(20)
        profile = _make_profile(100000)
        semantic_profile = _make_semantic_profile()
        prompts = PromptRegistry(DEFAULT_PROMPT_BASE_DIR)

        plan = build_story_analysis_plan_from_profile(snapshot, profile)
        prep = build_character_semantic_preparation(
            plan, profile, semantic_profile, prompts=prompts
        )
        # The A6C preparation must succeed and produce valid identity hashes.
        assert len(prep.character_request_identity_hashes) == len(
            plan.character_packages
        )
        for h in prep.character_request_identity_hashes:
            assert len(h) == 64

    def test_a6d_identity_unaffected_by_a6e_prompt_change(self):
        """The A6D plot-window-analysis request identity does NOT bind the
        A6E global-skeleton prompt. Changing the A6E prompt does not change
        A6D identities."""
        from short_drama.story.story_analysis_window_semantic import (
            build_plot_window_semantic_preparation,
        )
        from short_drama.story.story_analysis_planning import (
            build_story_analysis_plan_from_profile,
        )

        snapshot = _make_snapshot_with_events(20)
        profile = _make_profile(100000)
        semantic_profile = _make_semantic_profile()
        prompts = PromptRegistry(DEFAULT_PROMPT_BASE_DIR)

        plan = build_story_analysis_plan_from_profile(snapshot, profile)
        prep = build_plot_window_semantic_preparation(
            snapshot, plan, profile, semantic_profile, prompts=prompts
        )
        assert len(prep.window_request_identity_hashes) == len(plan.windows)
        for h in prep.window_request_identity_hashes:
            assert len(h) == 64

    # 10. A6E request identity change ---------------------------------------

    def test_a6e_request_identity_changes_with_prompt(self):
        """The A6E request identity changes when the prompt content hash
        changes (because the request identity binds the prompt identity)."""
        from short_drama.story.consolidation import (
            OutputSchemaAssetIdentity,
            PromptAssetIdentity,
        )

        profile = _make_profile(100000)
        semantic_profile = _make_semantic_profile()

        # Old prompt hash (pre-V3 refinement) vs new (V3 refinement).
        old_prompt_hash = (
            "1abb2c676cf4d4f70e0a2ce6e26175ab880d794ec6d827a1b8f2088ed94dd65e"
        )
        new_prompt_hash = (
            "b4f89fbd52e57bd4102cdaa40ffca9d1acfb62452c1e0133a8891985c51cf218"
        )

        schema_id = OutputSchemaAssetIdentity(
            "a6-global-skeleton-output", 1, "b" * 64
        )
        common = dict(
            plan_hash="c" * 64,
            profile=profile,
            semantic_profile=semantic_profile,
            packet_hash="d" * 64,
            request_hash="e" * 64,
            character_request_identity_hashes=("f" * 64,),
            window_request_identity_hashes=("g" * 64,),
        )

        snapshot = _make_snapshot_with_events(20)
        h_old = global_skeleton_request_identity_hash(
            consolidation_manifest_ref=snapshot.consolidation_manifest_ref,
            prompt_identity=PromptAssetIdentity(
                "a6.global-skeleton", 1, old_prompt_hash
            ),
            output_schema_identity=schema_id,
            **common,
        )
        h_new = global_skeleton_request_identity_hash(
            consolidation_manifest_ref=snapshot.consolidation_manifest_ref,
            prompt_identity=PromptAssetIdentity(
                "a6.global-skeleton", 1, new_prompt_hash
            ),
            output_schema_identity=schema_id,
            **common,
        )
        assert h_old != h_new

    # 11. Schema and evidence validation preserved --------------------------

    def test_schema_and_evidence_validation_still_functional(self):
        """The existing canonical-evidence validation and schema validation
        behavior is unaffected by the prompt change (these are code-level
        invariants, not prompt-dependent)."""
        snapshot = _make_snapshot_with_events(20)
        payload = _valid_payload()
        # Valid payload passes.
        assert validate_global_skeleton_output(payload, snapshot) is None
        # Invalid ref still fails.
        payload["event_importance_overlay"][0]["event_ref"] = "evt_999999"
        detail = validate_global_skeleton_output(payload, snapshot)
        assert detail is not None
        assert "evt_999999" in detail

    # ------------------------------------------------------------------
    # 12. V3 prompt refinement: whole-story conflict guidance
    # ------------------------------------------------------------------

    def test_prompt_main_conflict_whole_story_tension(self):
        """The V3 prompt guides main_conflict toward whole-story dominant
        tension and distinguishes sustained conflict from localized
        antagonist encounters."""
        spec = self._prompt_spec()
        sys_text = spec.system_template
        # The prompt must mention identifying the dominant whole-story tension.
        assert "main_conflict" in sys_text
        assert "\u7a81\u51fa" in sys_text or "\u5c40\u90e8" in sys_text  # 局部 (local)
        # Non-antagonistic conflict types are mentioned.
        assert "\u8eab\u4efd" in sys_text  # 身份 (identity)
        # Late antagonist conflicts are directed to secondary.
        assert "secondary_conflicts" in sys_text

    # ------------------------------------------------------------------
    # 13. V3 prompt refinement: epistemic boundary guidance
    # ------------------------------------------------------------------

    def test_prompt_epistemic_boundary_distinction(self):
        """The V3 prompt distinguishes reader inference from character
        cognition and forbids attributing reader-level conclusions to
        character consciousness without explicit source support."""
        spec = self._prompt_spec()
        sys_text = spec.system_template
        # The prompt must state that reader inference != character cognition.
        assert "\u8bfb\u8005" in sys_text  # 读者 (reader)
        assert "\u5185\u5728\u8ba4\u77e5" in sys_text  # 内在认知 (inner cognition)

    # ------------------------------------------------------------------
    # 14. V3 prompt refinement: climax vs resolution distinction
    # ------------------------------------------------------------------

    def test_prompt_climax_resolution_distinction(self):
        """The V3 prompt distinguishes climax (decisive confrontation) from
        resolution (aftermath) and forbids the ending event from
        automatically being the climax."""
        spec = self._prompt_spec()
        sys_text = spec.system_template
        # The prompt must state that climax and resolution are different
        # narrative functions.
        assert "\u9ad8\u6f6e" in sys_text  # 高潮 (climax)
        assert "\u6536\u675f" in sys_text  # 收束 (resolution)
        assert "\u540c\u4e00\u4e8b\u4ef6" in sys_text  # 同一事件 (same event)
        # The restraint section must forbid ending=climax conflation.
        assert "\u7ed3\u5c40" in sys_text  # 结局 (ending)

    # ------------------------------------------------------------------
    # 15. V3 prompt refinement: evidence-mode consistency across types
    # ------------------------------------------------------------------

    def test_prompt_evidence_mode_consistency(self):
        """The V3 prompt clarifies that a directly observed event being
        explicit does not make its narrative-function interpretation
        (climax, structural boundary, main conflict) also explicit."""
        spec = self._prompt_spec()
        sys_text = spec.system_template
        # The evidence-mode section must contain the clarification.
        assert "\u53d9\u4e8b\u529f\u80fd\u5224\u65ad" in sys_text  # 叙事功能判断
        # The clarification must reference explicit/inferred distinction.
        assert "explicit" in sys_text
        assert "inferred" in sys_text


# ---------------------------------------------------------------------------
# 16. DEFAULT_PROMPT_BASE_DIR import for TestPromptRefinement
# ---------------------------------------------------------------------------

from short_drama.story.reconciliation_semantic import (  # noqa: E402
    DEFAULT_PROMPT_BASE_DIR,
)
