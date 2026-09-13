from __future__ import annotations

import pytest

from short_drama.foundation import (
    ApprovalBlockedError,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalRef,
    ApprovalResolutionError,
    FilePointerStore,
    FindingDisposition,
    LineageRef,
    PointerConflictError,
    PointerIntegrityError,
    PointerKind,
    ReviewItemDisposition,
    ValidationFinding,
    ValidationReport,
    persist_approval_record,
    persist_validation_report,
    resolve_approval_ref,
)
from conftest import put_artifact


def make_report(store, target, findings=(), *, report_id="report", revision=1):
    report = ValidationReport((LineageRef("target", target),), tuple(findings))
    ref = persist_validation_report(store, report, artifact_id=report_id, revision=revision)
    return report, ref


def record(target, report_ref, decision="APPROVE", dispositions=()):
    return ApprovalRecord(
        target_ref=target,
        validation_report_ref=report_ref,
        target_role="target",
        decision=decision,
        reviewer_ref="human:song",
        decision_timestamp="2026-09-13T20:00:00-06:00",
        review_item_dispositions=tuple(dispositions),
    )


def test_approval_exact_target_report_and_ref_resolution(artifact_store):
    target = put_artifact(artifact_store)
    _, report_ref = make_report(artifact_store, target)
    approval_ref = persist_approval_record(artifact_store, record(target, report_ref), artifact_id="approval", revision=1)
    resolved = resolve_approval_ref(artifact_store, ApprovalRef(target, approval_ref))
    assert resolved.target_ref == target
    assert resolved.validation_report_ref == report_ref


def test_approve_fail_report_blocked(artifact_store):
    target = put_artifact(artifact_store)
    finding = ValidationFinding("b", "x", "BLOCKING", "b1", "b1", "bad", (target,))
    _, report_ref = make_report(artifact_store, target, (finding,))
    with pytest.raises(ApprovalBlockedError):
        persist_approval_record(artifact_store, record(target, report_ref), artifact_id="approval", revision=1)


def test_approve_undisposed_review_required_blocked(artifact_store):
    target = put_artifact(artifact_store)
    finding = ValidationFinding("r", "x", "REVIEW_REQUIRED", "b1", "b1", "review", (target,))
    _, report_ref = make_report(artifact_store, target, (finding,))
    with pytest.raises(ApprovalBlockedError):
        persist_approval_record(artifact_store, record(target, report_ref), artifact_id="approval", revision=1)


def test_approve_revision_requested_disposition_blocked(artifact_store):
    target = put_artifact(artifact_store)
    finding = ValidationFinding("r", "x", "REVIEW_REQUIRED", "b1", "b1", "review", (target,))
    _, report_ref = make_report(artifact_store, target, (finding,))
    disp = FindingDisposition("r", ReviewItemDisposition.REVISION_REQUESTED)
    with pytest.raises(ApprovalBlockedError):
        persist_approval_record(artifact_store, record(target, report_ref, dispositions=(disp,)), artifact_id="approval", revision=1)


def test_approve_all_review_items_accepted(artifact_store):
    target = put_artifact(artifact_store)
    finding = ValidationFinding("r", "x", "REVIEW_REQUIRED", "b1", "b1", "review", (target,))
    _, report_ref = make_report(artifact_store, target, (finding,))
    disp = FindingDisposition("r", ReviewItemDisposition.ACCEPT_AS_DESIGNED)
    approval_ref = persist_approval_record(artifact_store, record(target, report_ref, dispositions=(disp,)), artifact_id="approval", revision=1)
    assert resolve_approval_ref(artifact_store, ApprovalRef(target, approval_ref)).decision is ApprovalDecision.APPROVE


def test_revision_requested_and_abandon_persist_historically(artifact_store):
    target = put_artifact(artifact_store)
    _, report_ref = make_report(artifact_store, target)
    rr = persist_approval_record(artifact_store, record(target, report_ref, decision="REVISION_REQUESTED"), artifact_id="rr", revision=1)
    ab = persist_approval_record(artifact_store, record(target, report_ref, decision="ABANDON"), artifact_id="ab", revision=1)
    assert artifact_store.get_ref(rr)
    assert artifact_store.get_ref(ab)
    with pytest.raises(ApprovalResolutionError):
        resolve_approval_ref(artifact_store, ApprovalRef(target, rr))


def test_mismatched_approval_ref_fails(artifact_store):
    target = put_artifact(artifact_store, artifact_id="a")
    other = put_artifact(artifact_store, artifact_id="b")
    _, report_ref = make_report(artifact_store, target)
    approval = persist_approval_record(artifact_store, record(target, report_ref), artifact_id="approval", revision=1)
    with pytest.raises(ApprovalResolutionError):
        resolve_approval_ref(artifact_store, ApprovalRef(other, approval))


def test_current_approved_requires_valid_approve_record(tmp_path, artifact_store):
    target = put_artifact(artifact_store)
    _, report_ref = make_report(artifact_store, target)
    approved = persist_approval_record(artifact_store, record(target, report_ref), artifact_id="approval", revision=1)
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    pointer_ref = pointers.compare_and_set(
        pointer_id="approved-story",
        pointer_kind=PointerKind.CURRENT_APPROVED,
        expected_pointer_ref=None,
        target_ref=target,
        authority_ref=approved,
    )
    pointer = pointers.resolve_current("approved-story")
    assert pointer.target_ref == target
    assert pointer.authority_ref == approved
    assert pointers.resolve_current_pointer_ref("approved-story") == pointer_ref


def test_nonapprove_record_cannot_drive_current_approved(tmp_path, artifact_store):
    target = put_artifact(artifact_store)
    _, report_ref = make_report(artifact_store, target)
    rr = persist_approval_record(artifact_store, record(target, report_ref, decision="REVISION_REQUESTED"), artifact_id="rr", revision=1)
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    with pytest.raises(PointerIntegrityError):
        pointers.compare_and_set(
            pointer_id="approved-story",
            pointer_kind="CURRENT_APPROVED",
            expected_pointer_ref=None,
            target_ref=target,
            authority_ref=rr,
        )


def test_new_revision_requires_own_approval_and_supersession_preserves_history(tmp_path, artifact_store):
    v1 = put_artifact(artifact_store, revision=1, payload={"v": 1})
    v2 = put_artifact(artifact_store, revision=2, payload={"v": 2})
    _, r1 = make_report(artifact_store, v1, report_id="r1")
    _, r2 = make_report(artifact_store, v2, report_id="r2")
    a1 = persist_approval_record(artifact_store, record(v1, r1), artifact_id="a1", revision=1)
    a2 = persist_approval_record(artifact_store, record(v2, r2), artifact_id="a2", revision=1)
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    p1 = pointers.compare_and_set(pointer_id="approved-story", pointer_kind="CURRENT_APPROVED", expected_pointer_ref=None, target_ref=v1, authority_ref=a1)
    with pytest.raises(PointerIntegrityError):
        pointers.compare_and_set(pointer_id="other", pointer_kind="CURRENT_APPROVED", expected_pointer_ref=None, target_ref=v2, authority_ref=a1)
    p2 = pointers.compare_and_set(pointer_id="approved-story", pointer_kind="CURRENT_APPROVED", expected_pointer_ref=p1, target_ref=v2, authority_ref=a2)
    assert pointers.resolve_current("approved-story").target_ref == v2
    assert artifact_store.get_ref(v1)
    assert artifact_store.get_ref(a1)
    assert pointers.active_history("approved-story") == (p2, p1)


def test_stale_current_approved_update_fails(tmp_path, artifact_store):
    v1 = put_artifact(artifact_store, revision=1)
    v2 = put_artifact(artifact_store, revision=2)
    v3 = put_artifact(artifact_store, revision=3)
    _, r1 = make_report(artifact_store, v1, report_id="r1")
    _, r2 = make_report(artifact_store, v2, report_id="r2")
    _, r3 = make_report(artifact_store, v3, report_id="r3")
    a1 = persist_approval_record(artifact_store, record(v1, r1), artifact_id="a1", revision=1)
    a2 = persist_approval_record(artifact_store, record(v2, r2), artifact_id="a2", revision=1)
    a3 = persist_approval_record(artifact_store, record(v3, r3), artifact_id="a3", revision=1)
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    p1 = pointers.compare_and_set(pointer_id="approved", pointer_kind="CURRENT_APPROVED", expected_pointer_ref=None, target_ref=v1, authority_ref=a1)
    pointers.compare_and_set(pointer_id="approved", pointer_kind="CURRENT_APPROVED", expected_pointer_ref=p1, target_ref=v2, authority_ref=a2)
    with pytest.raises(PointerConflictError):
        pointers.compare_and_set(pointer_id="approved", pointer_kind="CURRENT_APPROVED", expected_pointer_ref=p1, target_ref=v3, authority_ref=a3)


def test_current_approved_resolver_returns_exact_approval_ref(tmp_path, artifact_store):
    target = put_artifact(artifact_store)
    _, report_ref = make_report(artifact_store, target)
    approval = persist_approval_record(artifact_store, record(target, report_ref), artifact_id="approval", revision=1)
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    pointers.compare_and_set(
        pointer_id="approved-story",
        pointer_kind="CURRENT_APPROVED",
        expected_pointer_ref=None,
        target_ref=target,
        authority_ref=approval,
    )
    assert pointers.resolve_current_approved("approved-story") == ApprovalRef(target, approval)
