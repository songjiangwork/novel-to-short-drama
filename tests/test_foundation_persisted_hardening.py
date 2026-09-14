from __future__ import annotations

import pytest

from short_drama.artifacts import ImmutableArtifactEnvelope, canonical_json_bytes
from short_drama.foundation import (
    APPROVAL_RECORD_ARTIFACT_TYPE,
    CURRENT_POINTER_ARTIFACT_TYPE,
    SUPERSESSION_RECORD_ARTIFACT_TYPE,
    VALIDATION_REPORT_ARTIFACT_TYPE,
    ApprovalRecord,
    ApprovalResolutionError,
    CurrentPointer,
    FilePointerStore,
    FindingDisposition,
    LineageRef,
    PointerIntegrityError,
    PointerKind,
    ReviewItemDisposition,
    SupersessionError,
    SupersessionRecord,
    ValidationFinding,
    ValidationModelError,
    ValidationReport,
    load_approval_record,
    load_validation_report,
    persist_approval_record,
    persist_validation_report,
)

from conftest import put_artifact


def put_envelope(store, *, artifact_type, artifact_id, revision, schema_version, payload):
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        revision=revision,
        schema_version=schema_version,
        payload=payload,
    )
    store.put(envelope)
    return envelope.ref


def make_approval(store, target, *, approval_id="approval"):
    report = ValidationReport((LineageRef("target", target),), ())
    report_ref = persist_validation_report(
        store, report, artifact_id=f"{approval_id}-report", revision=1
    )
    record = ApprovalRecord(
        target_ref=target,
        validation_report_ref=report_ref,
        target_role="target",
        decision="APPROVE",
        reviewer_ref="reviewer",
        decision_timestamp="2026-09-13T20:00:00-06:00",
    )
    approval_ref = persist_approval_record(
        store, record, artifact_id=approval_id, revision=1
    )
    return report_ref, record, approval_ref


def test_current_pointer_rejects_unsupported_schema_version(tmp_path, artifact_store):
    target = put_artifact(artifact_store)
    pointer = CurrentPointer("p", PointerKind.CURRENT, target)
    pointer_ref = put_envelope(
        artifact_store,
        artifact_type=CURRENT_POINTER_ARTIFACT_TYPE,
        artifact_id="p",
        revision=1,
        schema_version=2,
        payload=pointer.to_dict(),
    )
    root = tmp_path / "state"
    pointers = FilePointerStore(root, artifact_store)
    (root / "heads" / "p.json").write_bytes(
        canonical_json_bytes({"current_pointer_ref": pointer_ref.to_dict()})
    )
    with pytest.raises(PointerIntegrityError, match="schema_version"):
        pointers.resolve_current("p")


def test_validation_report_rejects_unsupported_schema_version(artifact_store):
    target = put_artifact(artifact_store)
    report = ValidationReport((LineageRef("target", target),), ())
    report_ref = put_envelope(
        artifact_store,
        artifact_type=VALIDATION_REPORT_ARTIFACT_TYPE,
        artifact_id="report-v2",
        revision=1,
        schema_version=2,
        payload=report.to_dict(),
    )
    with pytest.raises(ValidationModelError, match="schema_version"):
        load_validation_report(artifact_store, report_ref)


def test_approval_record_rejects_unsupported_schema_version_and_cannot_authorize_current(
    tmp_path, artifact_store
):
    target = put_artifact(artifact_store)
    report = ValidationReport((LineageRef("target", target),), ())
    report_ref = persist_validation_report(
        artifact_store, report, artifact_id="report", revision=1
    )
    record = ApprovalRecord(
        target_ref=target,
        validation_report_ref=report_ref,
        target_role="target",
        decision="APPROVE",
        reviewer_ref="reviewer",
        decision_timestamp="2026-09-13T20:00:00-06:00",
    )
    approval_ref = put_envelope(
        artifact_store,
        artifact_type=APPROVAL_RECORD_ARTIFACT_TYPE,
        artifact_id="approval-v2",
        revision=1,
        schema_version=2,
        payload=record.to_dict(),
    )
    with pytest.raises(ApprovalResolutionError, match="schema_version"):
        load_approval_record(artifact_store, approval_ref)
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    with pytest.raises(PointerIntegrityError):
        pointers.compare_and_set(
            pointer_id="approved",
            pointer_kind=PointerKind.CURRENT_APPROVED,
            expected_pointer_ref=None,
            target_ref=target,
            authority_ref=approval_ref,
        )


def test_supersession_record_rejects_unsupported_schema_version(tmp_path, artifact_store):
    v1 = put_artifact(artifact_store, revision=1, payload={"v": 1})
    v2 = put_artifact(artifact_store, revision=2, payload={"v": 2})
    p1 = CurrentPointer("p", PointerKind.CURRENT, v1)
    p1_ref = put_envelope(
        artifact_store,
        artifact_type=CURRENT_POINTER_ARTIFACT_TYPE,
        artifact_id="p",
        revision=1,
        schema_version=1,
        payload=p1.to_dict(),
    )
    p2 = CurrentPointer("p", PointerKind.CURRENT, v2, previous_pointer_ref=p1_ref)
    p2_ref = put_envelope(
        artifact_store,
        artifact_type=CURRENT_POINTER_ARTIFACT_TYPE,
        artifact_id="p",
        revision=2,
        schema_version=1,
        payload=p2.to_dict(),
    )
    supersession = SupersessionRecord(
        "p", PointerKind.CURRENT, p1_ref, p2_ref, v1, v2
    )
    put_envelope(
        artifact_store,
        artifact_type=SUPERSESSION_RECORD_ARTIFACT_TYPE,
        artifact_id="p",
        revision=2,
        schema_version=2,
        payload=supersession.to_dict(),
    )
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    with pytest.raises(SupersessionError, match="schema_version"):
        pointers.resolve_supersession("p", p2_ref)


def test_persisted_pointer_chain_rejects_cross_kind_predecessor(tmp_path, artifact_store):
    target = put_artifact(artifact_store)
    _, _, approval_ref = make_approval(artifact_store, target)
    p1 = CurrentPointer("p", PointerKind.CURRENT, target)
    p1_ref = put_envelope(
        artifact_store,
        artifact_type=CURRENT_POINTER_ARTIFACT_TYPE,
        artifact_id="p",
        revision=1,
        schema_version=1,
        payload=p1.to_dict(),
    )
    p2 = CurrentPointer(
        "p",
        PointerKind.CURRENT_APPROVED,
        target,
        authority_ref=approval_ref,
        previous_pointer_ref=p1_ref,
    )
    p2_ref = put_envelope(
        artifact_store,
        artifact_type=CURRENT_POINTER_ARTIFACT_TYPE,
        artifact_id="p",
        revision=2,
        schema_version=1,
        payload=p2.to_dict(),
    )
    root = tmp_path / "state"
    pointers = FilePointerStore(root, artifact_store)
    (root / "heads" / "p.json").write_bytes(
        canonical_json_bytes({"current_pointer_ref": p2_ref.to_dict()})
    )
    with pytest.raises(PointerIntegrityError, match="pointer_kind"):
        pointers.resolve_current("p")
    with pytest.raises(PointerIntegrityError, match="pointer_kind"):
        pointers.active_history("p")


def test_persisted_validation_report_rejects_non_normal_semantic_order(artifact_store):
    a = put_artifact(artifact_store, artifact_id="a")
    b = put_artifact(artifact_store, artifact_id="b")
    f1 = ValidationFinding(
        "f1", "code.f1", "WARNING", "b2", "b2", "warning", (b, a)
    )
    f2 = ValidationFinding(
        "f2", "code.f2", "BLOCKING", "a1", "a1", "blocking", (a, b)
    )
    report = ValidationReport(
        (LineageRef("a", a), LineageRef("z", b)),
        (f1, f2),
    )
    payload = report.to_dict()
    payload["validated_refs"].reverse()
    payload["findings"].reverse()
    payload["findings"][0]["artifact_refs"].reverse()
    report_ref = put_envelope(
        artifact_store,
        artifact_type=VALIDATION_REPORT_ARTIFACT_TYPE,
        artifact_id="non-normal-report",
        revision=1,
        schema_version=1,
        payload=payload,
    )
    with pytest.raises(ValidationModelError, match="canonical semantic order"):
        load_validation_report(artifact_store, report_ref)


def test_persisted_approval_record_rejects_non_normal_disposition_order(artifact_store):
    target = put_artifact(artifact_store)
    f1 = ValidationFinding(
        "a", "code.a", "REVIEW_REQUIRED", "b1", "b1", "a", (target,)
    )
    f2 = ValidationFinding(
        "z", "code.z", "REVIEW_REQUIRED", "b1", "b1", "z", (target,)
    )
    report = ValidationReport((LineageRef("target", target),), (f1, f2))
    report_ref = persist_validation_report(
        artifact_store, report, artifact_id="review-report", revision=1
    )
    record = ApprovalRecord(
        target_ref=target,
        validation_report_ref=report_ref,
        target_role="target",
        decision="APPROVE",
        reviewer_ref="reviewer",
        decision_timestamp="2026-09-13T20:00:00-06:00",
        review_item_dispositions=(
            FindingDisposition("z", ReviewItemDisposition.ACCEPT_AS_DESIGNED),
            FindingDisposition("a", ReviewItemDisposition.ACCEPT_AS_DESIGNED),
        ),
    )
    payload = record.to_dict()
    payload["review_item_dispositions"].reverse()
    approval_ref = put_envelope(
        artifact_store,
        artifact_type=APPROVAL_RECORD_ARTIFACT_TYPE,
        artifact_id="non-normal-approval",
        revision=1,
        schema_version=1,
        payload=payload,
    )
    with pytest.raises(ApprovalResolutionError, match="canonical semantic order"):
        load_approval_record(artifact_store, approval_ref)
