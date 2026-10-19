"""B1.2 real-provider compatibility assets; deterministic, zero-provider tests.

The v1 assets remain frozen. The v2 schema avoids a llama.cpp grammar
issue with allOf(ref(sourceRef), kind const) emitting duplicate kind keys.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator

from short_drama.adaptation import (
    AdaptationStrategyError, B12SemanticBudget, B12SemanticBudgetError,
    build_b12_semantic_preparation, load_b12_semantic_assets,
)
from short_drama.artifacts import CanonicalSerializationError, content_hash, strict_json_loads
from short_drama.io import load_json
from short_drama.llm import OutputSchema, load_semantic_profile
from short_drama.paths import PROFILES_DIR, SCHEMAS_DIR

from test_adaptation_strategy import _semantic_preparation


V1_SCHEMA = "b12-adaptation-strategy-proposal.schema.json"
V2_SCHEMA = "b12-adaptation-strategy-proposal-v2.schema.json"
V2_SCHEMA_HASH = "2148703d5cc06bf44b137baa86447f1af4206bb7133dda14114df1745591ed2b"
V2_PROFILE_FILE_SHA256 = "b0e1db9901eb2d419039de29eabe231d32bbfeb4b25c6892da9a492b6d5dc586"
TYPED_REFS = (
    ("characterStrategy", "character"),
    ("relationshipStrategy", "relationship"),
    ("revealStrategy", "reveal"),
)


def _v2_schema() -> dict:
    return load_json(SCHEMAS_DIR / V2_SCHEMA)


def test_v2_schema_is_exactly_three_flattened_refs_and_new_id():
    v1 = load_json(SCHEMAS_DIR / V1_SCHEMA)
    actual = _v2_schema()
    expected = deepcopy(v1)
    source_ref = v1["$defs"]["sourceRef"]
    for name, kind in TYPED_REFS:
        previous = {"allOf": [
            {"$ref": "#/$defs/sourceRef"},
            {"properties": {"kind": {"const": kind}}},
        ]}
        assert v1["$defs"][name]["properties"]["source_ref"] == previous
        flat = deepcopy(source_ref)
        flat["properties"]["kind"] = {"const": kind}
        expected["$defs"][name]["properties"]["source_ref"] = flat
    expected["$id"] = (
        "https://novel-to-short-drama.local/schemas/"
        "b12-adaptation-strategy-proposal-v2.schema.json"
    )
    assert actual == expected  # No other schema drift.
    assert '"allOf"' not in json.dumps(actual)
    Draft202012Validator.check_schema(actual)
    assert content_hash(actual) == V2_SCHEMA_HASH
    snapshot = OutputSchema.create(
        schema_id="b12-adaptation-strategy-proposal", schema_version=2, schema=actual,
    )
    assert snapshot.schema_hash == V2_SCHEMA_HASH


@pytest.mark.parametrize("definition,kind", TYPED_REFS)
def test_v2_typed_refs_allow_only_exact_kind_and_known_fields(definition, kind):
    sub_schema = _v2_schema()["$defs"][definition]["properties"]["source_ref"]
    validator = Draft202012Validator(sub_schema)
    good = {"kind": kind, "canonical_id": "demo_01"}
    assert not list(validator.iter_errors(good))
    for invalid in (
        {**good, "kind": "event"},
        {**good, "kind": "other"},
        {**good, "canonical_id": ""},
        {**good, "extra": "unrecognized"},
        {"kind": kind},
    ):
        assert list(validator.iter_errors(invalid)), invalid


def test_v2_profile_is_exactly_the_live_accepted_version():
    path = PROFILES_DIR / "adaptation_strategy_llm_v2.yaml"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == V2_PROFILE_FILE_SHA256
    v2 = load_semantic_profile(path)
    assert v2.profile_id == "b12-adaptation-strategy-llm-v2"
    assert v2.schema_version == 2
    assert v2.max_output_tokens == 57344
    assert v2.structured_output_mode == "json_schema"
    assert v2.temperature == 0.0
    assert v2.reasoning.enabled is False
    v1 = load_semantic_profile(PROFILES_DIR / "adaptation_strategy_llm_v1.yaml")
    assert v1.max_output_tokens == 8192
    assert v1.profile_id != v2.profile_id


def test_v2_semantic_identity_is_distinct_from_frozen_v1(tmp_path):
    _tree, _prepared, v1 = _semantic_preparation(tmp_path)
    v2_assets = load_b12_semantic_assets(version=2)
    v2 = build_b12_semantic_preparation(
        v1.preparation, semantic_asset_version=2,
        semantic_profile=v2_assets.semantic_profile,
        output_schema=v2_assets.output_schema,
        runtime_config=v1.runtime_config,
    )
    assert v2.request_identity != v1.request_identity
    assert v2.request.output_schema.schema_version == 2
    assert v2.measurement.reserved_output_tokens == 57344
    assert v2.output_schema.schema_hash == V2_SCHEMA_HASH
    assert v2.preparation.packet.source_snapshot == v1.preparation.packet.source_snapshot


def test_closed_selection_loads_only_the_matching_v1_or_v2_pair():
    v1 = load_b12_semantic_assets(version=1)
    v2 = load_b12_semantic_assets(version=2)
    assert (v1.version, v1.semantic_profile.profile_id, v1.output_schema.schema_version) == (
        1, "b12-adaptation-strategy-llm-v1", 1,
    )
    assert (v2.version, v2.semantic_profile.profile_id, v2.output_schema.schema_version) == (
        2, "b12-adaptation-strategy-llm-v2", 2,
    )
    assert v1.semantic_profile.semantic_profile_hash != v2.semantic_profile.semantic_profile_hash
    assert v1.output_schema.schema_hash != v2.output_schema.schema_hash


@pytest.mark.parametrize("version", (0, 3, True, "2", None))
def test_closed_selection_rejects_unsupported_versions(version):
    with pytest.raises(AdaptationStrategyError, match="unsupported B1.2 semantic asset version"):
        load_b12_semantic_assets(version=version)


def test_preparation_rejects_missing_cross_version_or_altered_assets(tmp_path):
    _tree, _prepared, v1 = _semantic_preparation(tmp_path)
    v1_assets = load_b12_semantic_assets(version=1)
    v2_assets = load_b12_semantic_assets(version=2)
    with pytest.raises(AdaptationStrategyError, match="supplied together"):
        build_b12_semantic_preparation(
            v1.preparation, runtime_config=v1.runtime_config,
            semantic_profile=v2_assets.semantic_profile,
        )
    with pytest.raises(AdaptationStrategyError, match="pinned v2 profile/schema pair"):
        build_b12_semantic_preparation(
            v1.preparation, semantic_asset_version=2, runtime_config=v1.runtime_config,
            semantic_profile=v1_assets.semantic_profile, output_schema=v2_assets.output_schema,
        )
    with pytest.raises(AdaptationStrategyError, match="pinned v1 profile/schema pair"):
        build_b12_semantic_preparation(
            v1.preparation, runtime_config=v1.runtime_config,
            semantic_profile=replace(v1_assets.semantic_profile, max_output_tokens=8193),
            output_schema=v1_assets.output_schema,
        )


def test_v2_budget_reserves_ceiling_but_requires_actual_runtime_capacity(tmp_path):
    _tree, _prepared, v1 = _semantic_preparation(tmp_path)
    v2 = build_b12_semantic_preparation(
        v1.preparation, semantic_asset_version=2, runtime_config=v1.runtime_config,
    )
    assert v2.measurement.reserved_output_tokens == 57_344
    required = (
        v2.measurement.provider_request_estimated_tokens
        + v2.measurement.reserved_output_tokens
    )
    with pytest.raises(B12SemanticBudgetError, match="reserved output"):
        v2.enforce_budget(B12SemanticBudget(
            v2.runtime_config.request_model, required - 1,
            v2.measurement.provider_request_estimated_tokens,
        ))
    v2.enforce_budget(B12SemanticBudget(
        v2.runtime_config.request_model, required,
        v2.measurement.provider_request_estimated_tokens,
    ))


def test_duplicate_kind_remains_strictly_rejected_locally():
    duplicate = '{"source_ref":{"kind":"character","kind":"character","canonical_id":"demo_01"}}'
    with pytest.raises(CanonicalSerializationError, match="duplicate"):
        strict_json_loads(duplicate)
