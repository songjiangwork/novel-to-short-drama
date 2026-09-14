from __future__ import annotations

import pytest

from short_drama.artifacts import ImmutableArtifactEnvelope, canonical_json_bytes
from short_drama.foundation import (
    CURRENT_POINTER_ARTIFACT_TYPE,
    ApprovalRecord,
    CurrentPointer,
    FilePointerStore,
    LineageRef,
    PointerIntegrityError,
    PointerKind,
    ValidationReport,
    persist_approval_record,
    persist_validation_report,
)

from conftest import put_artifact


def _put_pointer(store, pointer: CurrentPointer, revision: int):
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type=CURRENT_POINTER_ARTIFACT_TYPE,
        artifact_id=pointer.pointer_id,
        revision=revision,
        schema_version=1,
        payload=pointer.to_dict(),
    )
    store.put(envelope)
    return envelope.ref


def test_current_resolution_rejects_transitive_cross_kind_predecessor(
    tmp_path, artifact_store
):
    target = put_artifact(artifact_store)
    report = ValidationReport((LineageRef("target", target),), ())
    report_ref = persist_validation_report(
        artifact_store,
        report,
        artifact_id="deep-report",
        revision=1,
    )
    approval = ApprovalRecord(
        target_ref=target,
        validation_report_ref=report_ref,
        target_role="target",
        decision="APPROVE",
        reviewer_ref="reviewer",
        decision_timestamp="2026-09-13T20:00:00-06:00",
    )
    approval_ref = persist_approval_record(
        artifact_store,
        approval,
        artifact_id="deep-approval",
        revision=1,
    )

    p1 = CurrentPointer("p", PointerKind.CURRENT, target)
    p1_ref = _put_pointer(artifact_store, p1, 1)
    p2 = CurrentPointer(
        "p",
        PointerKind.CURRENT_APPROVED,
        target,
        authority_ref=approval_ref,
        previous_pointer_ref=p1_ref,
    )
    p2_ref = _put_pointer(artifact_store, p2, 2)
    p3 = CurrentPointer(
        "p",
        PointerKind.CURRENT_APPROVED,
        target,
        authority_ref=approval_ref,
        previous_pointer_ref=p2_ref,
    )
    p3_ref = _put_pointer(artifact_store, p3, 3)

    root = tmp_path / "state"
    pointers = FilePointerStore(root, artifact_store)
    (root / "heads" / "p.json").write_bytes(
        canonical_json_bytes({"current_pointer_ref": p3_ref.to_dict()})
    )

    with pytest.raises(PointerIntegrityError, match="pointer_kind"):
        pointers.resolve_current("p")
    with pytest.raises(PointerIntegrityError, match="pointer_kind"):
        pointers.active_history("p")
