from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from short_drama.foundation import (
    ApprovalRecord,
    ApprovalRef,
    CurrentPointer,
    FindingDisposition,
    LineageRef,
    PointerKind,
    ReviewItemDisposition,
    SupersessionRecord,
    ValidationFinding,
    ValidationReport,
)
from conftest import put_artifact

SCHEMAS = Path(__file__).parents[1] / "schemas"


def load_schema(name):
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def assert_valid(name, value):
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(value)


FOUNDATION_SCHEMAS = (
    "lineage-ref.schema.json",
    "current-pointer.schema.json",
    "supersession-record.schema.json",
    "validation-report.schema.json",
    "approval-ref.schema.json",
    "approval-record.schema.json",
    "pointer-head.schema.json",
)


def test_foundation_schemas_are_valid():
    for name in FOUNDATION_SCHEMAS:
        Draft202012Validator.check_schema(load_schema(name))


def test_embedded_artifact_ref_contract_matches_issue1_shape():
    artifact_ref_schema = load_schema("artifact-ref.schema.json")
    canonical = {
        "type": artifact_ref_schema["type"],
        "additionalProperties": artifact_ref_schema["additionalProperties"],
        "required": artifact_ref_schema["required"],
        "properties": artifact_ref_schema["properties"],
    }
    assert load_schema("lineage-ref.schema.json")["$defs"]["artifactRef"] == canonical
    for name in (
        "current-pointer.schema.json",
        "supersession-record.schema.json",
        "validation-report.schema.json",
        "approval-ref.schema.json",
        "approval-record.schema.json",
        "pointer-head.schema.json",
    ):
        assert load_schema(name)["$defs"]["artifactRef"] == canonical


def test_schema_accepts_domain_payloads(artifact_store):
    a = put_artifact(artifact_store, artifact_id="a")
    b = put_artifact(artifact_store, artifact_id="b")
    lineage = LineageRef("target", a)
    assert_valid("lineage-ref.schema.json", lineage.to_dict())

    pointer1 = CurrentPointer("current-a", PointerKind.CURRENT, a)
    assert_valid("current-pointer.schema.json", pointer1.to_dict())

    # Synthetic refs are sufficient for structural schema coverage.
    pointer1_ref = put_artifact(artifact_store, artifact_type="current_pointer", artifact_id="current-a", revision=1)
    pointer2_ref = put_artifact(artifact_store, artifact_type="current_pointer", artifact_id="current-a", revision=2)
    supersession = SupersessionRecord("current-a", PointerKind.CURRENT, pointer1_ref, pointer2_ref, a, b)
    assert_valid("supersession-record.schema.json", supersession.to_dict())

    finding = ValidationFinding("f1", "demo", "REVIEW_REQUIRED", "b1", "b1", "review", (a,))
    report = ValidationReport((lineage,), (finding,))
    assert_valid("validation-report.schema.json", report.to_dict())

    approval = ApprovalRecord(
        target_ref=a,
        validation_report_ref=put_artifact(artifact_store, artifact_type="validation_report", artifact_id="report"),
        target_role="target",
        decision="APPROVE",
        reviewer_ref="human:song",
        decision_timestamp="2026-09-13T20:00:00-06:00",
        review_item_dispositions=(FindingDisposition("f1", ReviewItemDisposition.ACCEPT_AS_DESIGNED),),
    )
    assert_valid("approval-record.schema.json", approval.to_dict())
    approval_ref = ApprovalRef(a, put_artifact(artifact_store, artifact_type="approval_record", artifact_id="approval"))
    assert_valid("approval-ref.schema.json", approval_ref.to_dict())
    assert_valid("pointer-head.schema.json", {"current_pointer_ref": pointer1_ref.to_dict()})
