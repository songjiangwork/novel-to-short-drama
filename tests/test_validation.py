from __future__ import annotations

import pytest

from short_drama.artifacts import canonical_json_bytes
from short_drama.foundation import (
    LineageRef,
    ValidationFinding,
    ValidationModelError,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    jsonschema_findings,
    load_validation_report,
    persist_validation_report,
)
from conftest import put_artifact


def finding(fid, severity, ref, *, owner="b1", route="b1.2"):
    return ValidationFinding(
        finding_id=fid,
        code=f"code.{fid}",
        severity=severity,
        owner_stage=owner,
        repair_route=route,
        message=f"message {fid}",
        artifact_refs=(ref,),
    )


def test_validation_results(artifact_store):
    ref = put_artifact(artifact_store)
    lineage = (LineageRef("target", ref),)
    assert ValidationReport(lineage, ()).summary.result is ValidationResult.PASS
    assert ValidationReport(lineage, (finding("w", "WARNING", ref),)).summary.result is ValidationResult.PASS
    assert ValidationReport(lineage, (finding("r", "REVIEW_REQUIRED", ref),)).summary.result is ValidationResult.PASS_WITH_REVIEW_ITEMS
    assert ValidationReport(lineage, (finding("b", "BLOCKING", ref),)).summary.result is ValidationResult.FAIL


def test_invalid_severity_rejected(artifact_store):
    ref = put_artifact(artifact_store)
    with pytest.raises(ValidationModelError):
        finding("x", "INFO", ref)


def test_report_ordering_is_deterministic(artifact_store):
    a = put_artifact(artifact_store, artifact_id="a")
    b = put_artifact(artifact_store, artifact_id="b")
    f1 = finding("z", "WARNING", b, owner="b2", route="b2.1")
    f2 = finding("a", "BLOCKING", a, owner="a1", route="a1")
    r1 = ValidationReport((LineageRef("z", b), LineageRef("a", a)), (f1, f2))
    r2 = ValidationReport((LineageRef("a", a), LineageRef("z", b)), (f2, f1))
    assert r1 == r2
    assert canonical_json_bytes(r1.to_dict()) == canonical_json_bytes(r2.to_dict())


def test_report_inconsistent_summary_rejected(artifact_store):
    ref = put_artifact(artifact_store)
    report = ValidationReport((LineageRef("target", ref),), (finding("b", "BLOCKING", ref),))
    data = report.to_dict()
    data["summary"]["result"] = "PASS"
    with pytest.raises(ValidationModelError):
        ValidationReport.from_dict(data)


def test_duplicate_finding_id_rejected(artifact_store):
    ref = put_artifact(artifact_store)
    f1 = finding("same", "WARNING", ref)
    f2 = finding("same", "BLOCKING", ref)
    with pytest.raises(ValidationModelError):
        ValidationReport((LineageRef("target", ref),), (f1, f2))


def test_validation_roundtrip_owner_route_and_provenance(artifact_store):
    a = put_artifact(artifact_store, artifact_id="a")
    b = put_artifact(artifact_store, artifact_id="b")
    f = ValidationFinding("f", "cross.ref", "WARNING", "b4", "b3", "cross issue", (b, a), ("shots", 2))
    report = ValidationReport((LineageRef("target", a), LineageRef("source", b)), (f,))
    restored = ValidationReport.from_dict(report.to_dict())
    assert restored == report
    assert restored.findings[0].owner_stage == "b4"
    assert restored.findings[0].repair_route == "b3"
    assert restored.findings[0].artifact_refs == tuple(sorted((a, b), key=lambda x:(x.artifact_type,x.artifact_id,x.revision,x.content_hash)))


def test_validation_persistence_roundtrip(artifact_store):
    ref = put_artifact(artifact_store)
    report = ValidationReport((LineageRef("target", ref),), ())
    report_ref = persist_validation_report(artifact_store, report, artifact_id="report", revision=1)
    assert load_validation_report(artifact_store, report_ref) == report


def test_jsonschema_findings_deterministic(artifact_store):
    ref = put_artifact(artifact_store)
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["name", "age"],
        "properties": {"age": {"type": "integer", "minimum": 0}},
    }
    a = jsonschema_findings({}, schema, code_prefix="schema", owner_stage="a1", repair_route="a1", artifact_refs=(ref,))
    b = jsonschema_findings({}, schema, code_prefix="schema", owner_stage="a1", repair_route="a1", artifact_refs=(ref,))
    assert a == b
    assert len(a) == 2
    assert all(item.severity is ValidationSeverity.BLOCKING for item in a)
    assert len({item.finding_id for item in a}) == 2


def test_finding_sort_handles_mixed_string_and_integer_paths(artifact_store):
    ref = put_artifact(artifact_store)
    a = ValidationFinding("a", "same", "WARNING", "x", "x", "m", (ref,), (0,))
    b = ValidationFinding("b", "same", "WARNING", "x", "x", "m", (ref,), ("key",))
    report = ValidationReport((LineageRef("target", ref),), (b, a))
    assert {finding.finding_id for finding in report.findings} == {"a", "b"}


def test_invalid_json_schema_rejected():
    with pytest.raises(ValidationModelError):
        jsonschema_findings({}, {"type": 123}, code_prefix="schema", owner_stage="a1", repair_route="a1")
