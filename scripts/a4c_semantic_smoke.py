"""A4C controlled real-backend semantic resolution smoke.

Verifies the full A4C semantic resolution path against a real local
OpenAI-compatible server (llama.cpp / Qwen):

    synthetic in-memory A4B planning material (3 semantic pairs / 6 candidates)
      -> deterministic block packing (1 block)
      -> pair-context rendering (pair-local evidence selectors)
      -> PromptRegistry rendering (a4.entity-reconciliation v3)
      -> OutputSchema build (reconciliation-decision-selector-payload.schema.json)
      -> LLMClient.generate_structured(...)
      -> typed selector payload validation
      -> deterministic selector validation + exact EvidenceRef resolution
      -> provenance verification
      -> ReconciliationDecision construction

Expected results:
    Pair A (Alice/Alicia) → same_entity
    Pair B (Bob/Carol)    → different_entity
    Pair C (The doctor/Dr. Chen) → uncertain

This smoke does NOT:
  * read real novel source
  * persist artifacts
  * write CURRENT
  * manage server lifecycle
  * query /models
  * start/stop/restart the server

Usage:
    python scripts/a4c_semantic_smoke.py \\
        --runtime-config profiles/llm_local.yaml \\
        --reconciliation-profile profiles/entity_reconciliation_v2.yaml \\
        --llm-profile profiles/entity_reconciliation_llm_v1.yaml

Exit code 0 on success, 2 on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from short_drama.artifacts import ArtifactRef
from short_drama.io import load_json, load_yaml
from short_drama.llm import (
    LLMError,
    LLMInvocationProvenance,
    OpenAICompatibleLLMClient,
    OutputSchema,
    PromptRegistry,
    SemanticLLMProfile,
    load_runtime_config,
    load_semantic_profile,
)
from short_drama.story import (
    EntityReconciliationProfile,
    EvidenceRef,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
    ReconciliationDecision,
    ReconciliationInputSnapshot,
    ReconciliationPairPlan,
    ReconciliationPlanningError,
    ReconciliationPlanningResult,
    ReconciliationProvenanceError,
    ReconciliationSemanticError,
    ReconciliationSemanticGenerationError,
    resolve_semantic_ambiguity,
)
from short_drama.story.reconciliation import (
    CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
    CandidateEntityIndex,
    CandidateEntityIndexEntry,
)
from short_drama.story.reconciliation_semantic import (
    DEFAULT_OUTPUT_SCHEMA_PATH,
    DEFAULT_PROMPT_BASE_DIR,
)


# ---------------------------------------------------------------------------
# Synthetic A4B planning fixture (one block, 3 semantic pairs, 6 candidates)
# ---------------------------------------------------------------------------

CHUNK_ID = "CH001_C001"

# Pair A: Alice / Alicia → same_entity
# Explicit evidence: "Alice, also known as Alicia, entered the room."
PAIR_A_LEFT = f"{CHUNK_ID}:cand_char_001"
PAIR_A_RIGHT = f"{CHUNK_ID}:cand_char_002"
EVIDENCE_A = EvidenceRef(
    paragraph_id="CH001_P001",
    role="primary",
    strength="explicit",
    excerpt="Alice, also known as Alicia, entered the room.",
)

# Pair B: Bob / Carol → different_entity
# Explicit evidence: "Bob and Carol are two different people standing beside each other."
PAIR_B_LEFT = f"{CHUNK_ID}:cand_char_003"
PAIR_B_RIGHT = f"{CHUNK_ID}:cand_char_004"
EVIDENCE_B = EvidenceRef(
    paragraph_id="CH001_P002",
    role="primary",
    strength="explicit",
    excerpt="Bob and Carol are two different people standing beside each other.",
)

# Pair C: The doctor / Dr. Chen → uncertain
# Only independent evidence, no identity link or contradiction.
PAIR_C_LEFT = f"{CHUNK_ID}:cand_char_005"
PAIR_C_RIGHT = f"{CHUNK_ID}:cand_char_006"
EVIDENCE_C_LEFT = EvidenceRef(
    paragraph_id="CH001_P003",
    role="primary",
    strength="explicit",
    excerpt="The doctor walked into the hospital corridor.",
)
EVIDENCE_C_RIGHT = EvidenceRef(
    paragraph_id="CH001_P004",
    role="primary",
    strength="explicit",
    excerpt="Dr. Chen examined the patient in room 302.",
)


def make_artifact_ref(artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_type="candidate_extraction",
        artifact_id=artifact_id,
        revision=1,
        content_hash="a" * 64,
    )


def build_smoke_planning_result() -> ReconciliationPlanningResult:
    """Build the synthetic A4B planning result for the smoke test.

    One block: 3 semantic pairs, 6 unique candidates.
    """
    entries = (
        CandidateEntityIndexEntry(
            candidate_ref=PAIR_A_LEFT,
            candidate_kind="character",
            candidate_extraction_ref=make_artifact_ref("ext_a4c_smoke_001"),
            source_order_key="000001:000000001:01:000000001:" + PAIR_A_LEFT,
            display_name_original="Alice",
            aliases_original=("Alicia",),
            descriptors_zh=("女主角，也称作 Alicia。",),
            evidence_refs=(EVIDENCE_A,),
            possible_candidate_refs=(),
        ),
        CandidateEntityIndexEntry(
            candidate_ref=PAIR_A_RIGHT,
            candidate_kind="character",
            candidate_extraction_ref=make_artifact_ref("ext_a4c_smoke_001"),
            source_order_key="000001:000000002:01:000000002:" + PAIR_A_RIGHT,
            display_name_original="Alicia",
            aliases_original=("Alice",),
            descriptors_zh=("女主角，也称作 Alice。",),
            evidence_refs=(EVIDENCE_A,),
            possible_candidate_refs=(),
        ),
        CandidateEntityIndexEntry(
            candidate_ref=PAIR_B_LEFT,
            candidate_kind="character",
            candidate_extraction_ref=make_artifact_ref("ext_a4c_smoke_001"),
            source_order_key="000001:000000003:01:000000003:" + PAIR_B_LEFT,
            display_name_original="Bob",
            aliases_original=(),
            descriptors_zh=("站在 Carol 旁边的男性角色。",),
            evidence_refs=(EVIDENCE_B,),
            possible_candidate_refs=(),
        ),
        CandidateEntityIndexEntry(
            candidate_ref=PAIR_B_RIGHT,
            candidate_kind="character",
            candidate_extraction_ref=make_artifact_ref("ext_a4c_smoke_001"),
            source_order_key="000001:000000004:01:000000004:" + PAIR_B_RIGHT,
            display_name_original="Carol",
            aliases_original=(),
            descriptors_zh=("站在 Bob 旁边的女性角色。",),
            evidence_refs=(EVIDENCE_B,),
            possible_candidate_refs=(),
        ),
        CandidateEntityIndexEntry(
            candidate_ref=PAIR_C_LEFT,
            candidate_kind="character",
            candidate_extraction_ref=make_artifact_ref("ext_a4c_smoke_001"),
            source_order_key="000001:000000005:01:000000005:" + PAIR_C_LEFT,
            display_name_original="The doctor",
            aliases_original=(),
            descriptors_zh=("走进医院走廊的医生。",),
            evidence_refs=(EVIDENCE_C_LEFT,),
            possible_candidate_refs=(),
        ),
        CandidateEntityIndexEntry(
            candidate_ref=PAIR_C_RIGHT,
            candidate_kind="character",
            candidate_extraction_ref=make_artifact_ref("ext_a4c_smoke_001"),
            source_order_key="000001:000000006:01:000000006:" + PAIR_C_RIGHT,
            display_name_original="Dr. Chen",
            aliases_original=(),
            descriptors_zh=("在 302 房间检查病人的医生。",),
            evidence_refs=(EVIDENCE_C_RIGHT,),
            possible_candidate_refs=(),
        ),
    )

    pair_plans = (
        ReconciliationPairPlan(
            left_candidate_ref=PAIR_A_LEFT,
            right_candidate_ref=PAIR_A_RIGHT,
            state=PAIR_STATE_NEEDS_SEMANTIC_DECISION,
            signals=("identity_token_overlap",),
            shared_identity_keys=("alice", "alicia"),
            shared_tokens=("alice", "alicia"),
        ),
        ReconciliationPairPlan(
            left_candidate_ref=PAIR_B_LEFT,
            right_candidate_ref=PAIR_B_RIGHT,
            state=PAIR_STATE_NEEDS_SEMANTIC_DECISION,
            signals=("adjacent_chunk",),
            shared_identity_keys=(),
            shared_tokens=(),
        ),
        ReconciliationPairPlan(
            left_candidate_ref=PAIR_C_LEFT,
            right_candidate_ref=PAIR_C_RIGHT,
            state=PAIR_STATE_NEEDS_SEMANTIC_DECISION,
            signals=("identity_token_overlap",),
            shared_identity_keys=("doctor",),
            shared_tokens=("doctor",),
        ),
    )

    # Deterministic plan_hash (stable for the smoke fixture)
    from short_drama.artifacts import content_hash

    plan_hash = content_hash(
        {
            "normalization_policy_id": "a4-name-normalization-v1",
            "blocking_policy_id": "a4-blocking-v1",
            "canonicalization_policy_id": "a4-canonicalization-v1",
            "candidate_index": {"schema_version": 1, "entries": [e.to_dict() for e in entries]},
            "pair_plans": [p.to_dict() for p in pair_plans],
        }
    )

    return ReconciliationPlanningResult(
        candidate_index=CandidateEntityIndex(
            schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
            entries=entries,
        ),
        pair_plans=pair_plans,
        decisions=(),
        normalization_policy_id="a4-name-normalization-v1",
        blocking_policy_id="a4-blocking-v1",
        canonicalization_policy_id="a4-canonicalization-v1",
        plan_hash=plan_hash,
    )


# ---------------------------------------------------------------------------
# Smoke execution
# ---------------------------------------------------------------------------


def run_smoke(
    runtime_config_path: str,
    reconciliation_profile_path: str,
    llm_profile_path: str,
) -> int:
    """Execute the A4C semantic smoke. Returns 0 on success, 2 on failure."""

    # 1. Load configs
    runtime_config = load_runtime_config(runtime_config_path)
    semantic_profile = load_semantic_profile(llm_profile_path)

    reconciliation_profile = EntityReconciliationProfile.from_dict(
        load_yaml(reconciliation_profile_path)
    )

    # 2. Build LLM client
    client = OpenAICompatibleLLMClient(runtime_config)

    # 3. Build synthetic planning result
    planning_result = build_smoke_planning_result()

    # 4. Execute A4C semantic resolution
    print(f"Runtime: {runtime_config.base_url} model={runtime_config.request_model}")
    print(f"Semantic profile: {semantic_profile.profile_id}")
    print(f"Reconciliation profile: {reconciliation_profile.profile_id}")
    print(f"Semantic pairs: {sum(1 for p in planning_result.pair_plans if p.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION)}")
    print("Executing A4C semantic resolution...")

    try:
        result = resolve_semantic_ambiguity(
            planning_result,
            reconciliation_profile,
            semantic_profile,
            client,
            prompt_registry=PromptRegistry(DEFAULT_PROMPT_BASE_DIR),
            output_schema_path=DEFAULT_OUTPUT_SCHEMA_PATH,
        )
    except LLMError as exc:
        print(f"SMOKE FAIL: LLM transport error: {exc}")
        return 2
    except ReconciliationProvenanceError as exc:
        print(f"SMOKE FAIL: provenance error: {exc}")
        return 2
    except ReconciliationSemanticGenerationError as exc:
        print(f"SMOKE FAIL: semantic generation error: {exc}")
        return 2
    except ReconciliationSemanticError as exc:
        print(f"SMOKE FAIL: semantic error: {exc}")
        return 2

    # 5. Validate results
    print(f"\n--- Results ---")
    print(f"Blocks: {len(result.blocks)}")
    if result.blocks:
        print(f"  Block 1: {result.blocks[0].block_id}")
        print(f"  Semantic rounds: {result.block_results[0].semantic_rounds}")
        print(f"  Request hash: {result.block_results[0].request_hash[:40]}...")

    # Check pair decisions
    decisions_by_pair = {}
    for d in result.semantic_decisions:
        decisions_by_pair[(d.left_candidate_ref, d.right_candidate_ref)] = d.decision

    pair_a_key = (PAIR_A_LEFT, PAIR_A_RIGHT)
    pair_b_key = (PAIR_B_LEFT, PAIR_B_RIGHT)
    pair_c_key = (PAIR_C_LEFT, PAIR_C_RIGHT)

    pair_a_result = decisions_by_pair.get(pair_a_key, "<missing>")
    pair_b_result = decisions_by_pair.get(pair_b_key, "<missing>")
    pair_c_result = decisions_by_pair.get(pair_c_key, "<missing>")

    print(f"\nPair A (Alice/Alicia):     {pair_a_result}")
    print(f"Pair B (Bob/Carol):        {pair_b_result}")
    print(f"Pair C (The doctor/Dr.Chen): {pair_c_result}")

    # Verify expected results
    failures = []
    if pair_a_result != "same_entity":
        failures.append(f"Pair A expected same_entity, got {pair_a_result}")
    if pair_b_result != "different_entity":
        failures.append(f"Pair B expected different_entity, got {pair_b_result}")
    if pair_c_result != "uncertain":
        failures.append(f"Pair C expected uncertain, got {pair_c_result}")

    if len(result.blocks) != 1:
        failures.append(f"Expected 1 block, got {len(result.blocks)}")

    if len(result.semantic_decisions) != 3:
        failures.append(f"Expected 3 semantic decisions, got {len(result.semantic_decisions)}")

    if failures:
        print(f"\nSMOKE FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 2

    # Provenance check
    prov = result.block_results[0].generation_provenance
    print(f"\nProvenance:")
    print(f"  semantic_profile_id: {prov.semantic_profile_id}")
    print(f"  semantic_profile_hash: {prov.semantic_profile_hash[:40]}...")
    print(f"  prompt_id: {prov.prompt_id}")
    print(f"  prompt_version: {prov.prompt_version}")
    print(f"  prompt_content_hash: {prov.prompt_content_hash[:40]}...")
    print(f"  rendered_prompt_hash: {prov.rendered_prompt_hash[:40]}...")
    print(f"  output_schema_id: {prov.output_schema_id}")
    print(f"  output_schema_version: {prov.output_schema_version}")
    print(f"  output_schema_hash: {prov.output_schema_hash[:40]}...")
    print(f"  request_hash: {prov.request_hash[:40]}...")
    print(f"  provider_family: {prov.provider_family}")
    print(f"  model: {prov.model}")
    if prov.attempts if hasattr(prov, 'attempts') else None:
        print(f"  attempts: {prov.attempts}")

    print(f"\nSMOKE PASS")
    return 0


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A4C controlled real-backend semantic resolution smoke"
    )
    parser.add_argument(
        "--runtime-config",
        default="profiles/llm_local.yaml",
        help="Path to runtime config YAML",
    )
    parser.add_argument(
        "--reconciliation-profile",
        default="profiles/entity_reconciliation_v2.yaml",
        help="Path to entity reconciliation profile YAML",
    )
    parser.add_argument(
        "--llm-profile",
        default="profiles/entity_reconciliation_llm_v1.yaml",
        help="Path to semantic LLM profile YAML",
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    runtime_config_path = repo_root / args.runtime_config
    reconciliation_profile_path = repo_root / args.reconciliation_profile
    llm_profile_path = repo_root / args.llm_profile

    if not runtime_config_path.is_file():
        print(f"ERROR: runtime config not found: {runtime_config_path}")
        sys.exit(2)
    if not reconciliation_profile_path.is_file():
        print(f"ERROR: reconciliation profile not found: {reconciliation_profile_path}")
        sys.exit(2)
    if not llm_profile_path.is_file():
        print(f"ERROR: LLM profile not found: {llm_profile_path}")
        sys.exit(2)

    exit_code = run_smoke(
        str(runtime_config_path),
        str(reconciliation_profile_path),
        str(llm_profile_path),
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
