from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from short_drama.artifacts import ArtifactError, ArtifactRef, FileArtifactStore, ImmutableArtifactEnvelope

from ._common import (
    require_artifact_ref,
    require_exact_keys,
    require_optional_text,
    require_text,
)
from .errors import ApprovalBlockedError, ApprovalError, ApprovalResolutionError, ValidationModelError
from .validation import (
    ValidationResult,
    ValidationSeverity,
    load_validation_report,
)

APPROVAL_RECORD_ARTIFACT_TYPE = "approval_record"
APPROVAL_RECORD_SCHEMA_VERSION = 1


class ApprovalDecision(str, Enum):
    APPROVE = "APPROVE"
    REVISION_REQUESTED = "REVISION_REQUESTED"
    ABANDON = "ABANDON"


class ReviewItemDisposition(str, Enum):
    ACCEPT_AS_DESIGNED = "ACCEPT_AS_DESIGNED"
    REVISION_REQUESTED = "REVISION_REQUESTED"


def _coerce_enum(value: Any, enum_type: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ApprovalError(f"{field_name} must be one of: {allowed}") from exc


@dataclass(frozen=True, slots=True)
class FindingDisposition:
    finding_id: str
    value: ReviewItemDisposition

    def __post_init__(self) -> None:
        require_text(self.finding_id, "finding_id", ApprovalError)
        object.__setattr__(
            self,
            "value",
            _coerce_enum(self.value, ReviewItemDisposition, "disposition value"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"finding_id": self.finding_id, "value": self.value.value}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FindingDisposition":
        require_exact_keys(
            value,
            {"finding_id", "value"},
            "FindingDisposition",
            ApprovalError,
        )
        return cls(finding_id=value["finding_id"], value=value["value"])


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    target_ref: ArtifactRef
    validation_report_ref: ArtifactRef
    target_role: str
    decision: ApprovalDecision
    reviewer_ref: str
    decision_timestamp: str
    review_item_dispositions: tuple[FindingDisposition, ...] = ()
    reviewer_notes: str | None = None

    def __post_init__(self) -> None:
        require_artifact_ref(self.target_ref, "target_ref", ApprovalError)
        require_artifact_ref(self.validation_report_ref, "validation_report_ref", ApprovalError)
        require_text(self.target_role, "target_role", ApprovalError)
        object.__setattr__(self, "decision", _coerce_enum(self.decision, ApprovalDecision, "decision"))
        require_text(self.reviewer_ref, "reviewer_ref", ApprovalError)
        require_text(self.decision_timestamp, "decision_timestamp", ApprovalError)
        object.__setattr__(
            self,
            "reviewer_notes",
            require_optional_text(self.reviewer_notes, "reviewer_notes", ApprovalError),
        )

        dispositions = tuple(self.review_item_dispositions)
        if any(not isinstance(item, FindingDisposition) for item in dispositions):
            raise ApprovalError("review_item_dispositions must contain FindingDisposition values")
        dispositions = tuple(sorted(dispositions, key=lambda item: item.finding_id))
        ids = [item.finding_id for item in dispositions]
        if len(set(ids)) != len(ids):
            raise ApprovalError("review_item_dispositions must not contain duplicate finding_id values")
        object.__setattr__(self, "review_item_dispositions", dispositions)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_ref": self.target_ref.to_dict(),
            "validation_report_ref": self.validation_report_ref.to_dict(),
            "target_role": self.target_role,
            "decision": self.decision.value,
            "reviewer_ref": self.reviewer_ref,
            "decision_timestamp": self.decision_timestamp,
            "review_item_dispositions": [item.to_dict() for item in self.review_item_dispositions],
            "reviewer_notes": self.reviewer_notes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ApprovalRecord":
        require_exact_keys(
            value,
            {
                "target_ref",
                "validation_report_ref",
                "target_role",
                "decision",
                "reviewer_ref",
                "decision_timestamp",
                "review_item_dispositions",
                "reviewer_notes",
            },
            "ApprovalRecord",
            ApprovalError,
        )
        if not isinstance(value["review_item_dispositions"], list):
            raise ApprovalError("review_item_dispositions must be a list")
        try:
            target_ref = ArtifactRef.from_dict(value["target_ref"])
            validation_report_ref = ArtifactRef.from_dict(value["validation_report_ref"])
        except (TypeError, ValueError) as exc:
            raise ApprovalError(f"invalid ApprovalRecord ArtifactRef: {exc}") from exc
        return cls(
            target_ref=target_ref,
            validation_report_ref=validation_report_ref,
            target_role=value["target_role"],
            decision=value["decision"],
            reviewer_ref=value["reviewer_ref"],
            decision_timestamp=value["decision_timestamp"],
            review_item_dispositions=tuple(
                FindingDisposition.from_dict(item)
                for item in value["review_item_dispositions"]
            ),
            reviewer_notes=value["reviewer_notes"],
        )


@dataclass(frozen=True, slots=True)
class ApprovalRef:
    target_ref: ArtifactRef
    approval_record_ref: ArtifactRef

    def __post_init__(self) -> None:
        require_artifact_ref(self.target_ref, "target_ref", ApprovalError)
        require_artifact_ref(self.approval_record_ref, "approval_record_ref", ApprovalError)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_ref": self.target_ref.to_dict(),
            "approval_record_ref": self.approval_record_ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ApprovalRef":
        require_exact_keys(
            value,
            {"target_ref", "approval_record_ref"},
            "ApprovalRef",
            ApprovalError,
        )
        try:
            return cls(
                target_ref=ArtifactRef.from_dict(value["target_ref"]),
                approval_record_ref=ArtifactRef.from_dict(value["approval_record_ref"]),
            )
        except (TypeError, ValueError) as exc:
            raise ApprovalError(f"invalid ApprovalRef: {exc}") from exc


def verify_approval_record(store: FileArtifactStore, record: ApprovalRecord) -> None:
    if not isinstance(record, ApprovalRecord):
        raise ApprovalError("record must be an ApprovalRecord")
    try:
        store.get_ref(record.target_ref)
    except ArtifactError as exc:
        raise ApprovalResolutionError(f"approval target cannot be resolved: {record.target_ref!r}") from exc
    try:
        report = load_validation_report(store, record.validation_report_ref)
    except ValidationModelError as exc:
        raise ApprovalResolutionError(
            f"approval ValidationReport cannot be resolved: {record.validation_report_ref!r}"
        ) from exc

    if not any(
        lineage.role == record.target_role and lineage.artifact_ref == record.target_ref
        for lineage in report.validated_refs
    ):
        raise ApprovalResolutionError(
            "ValidationReport does not validate the exact approval target under target_role"
        )

    review_findings = {
        finding.finding_id: finding
        for finding in report.findings
        if finding.severity is ValidationSeverity.REVIEW_REQUIRED
    }
    disposition_map = {item.finding_id: item.value for item in record.review_item_dispositions}
    unknown = sorted(set(disposition_map) - set(review_findings))
    if unknown:
        raise ApprovalError(
            "review_item_dispositions reference findings that are not REVIEW_REQUIRED: "
            + ", ".join(unknown)
        )

    if record.decision is not ApprovalDecision.APPROVE:
        return

    if report.summary.result is ValidationResult.FAIL:
        raise ApprovalBlockedError("cannot APPROVE an artifact whose ValidationReport result is FAIL")

    missing = sorted(set(review_findings) - set(disposition_map))
    if missing:
        raise ApprovalBlockedError(
            "cannot APPROVE with undisposed REVIEW_REQUIRED findings: " + ", ".join(missing)
        )

    revision_requested = sorted(
        finding_id
        for finding_id, disposition in disposition_map.items()
        if disposition is ReviewItemDisposition.REVISION_REQUESTED
    )
    if revision_requested:
        raise ApprovalBlockedError(
            "cannot APPROVE when REVIEW_REQUIRED findings request revision: "
            + ", ".join(revision_requested)
        )

    not_accepted = sorted(
        finding_id
        for finding_id in review_findings
        if disposition_map.get(finding_id) is not ReviewItemDisposition.ACCEPT_AS_DESIGNED
    )
    if not_accepted:
        raise ApprovalBlockedError(
            "all REVIEW_REQUIRED findings must be ACCEPT_AS_DESIGNED before APPROVE: "
            + ", ".join(not_accepted)
        )


def approval_record_envelope(
    record: ApprovalRecord,
    *,
    artifact_id: str,
    revision: int,
    schema_version: int = APPROVAL_RECORD_SCHEMA_VERSION,
) -> ImmutableArtifactEnvelope:
    if not isinstance(record, ApprovalRecord):
        raise ApprovalError("record must be an ApprovalRecord")
    return ImmutableArtifactEnvelope.create(
        artifact_type=APPROVAL_RECORD_ARTIFACT_TYPE,
        artifact_id=artifact_id,
        revision=revision,
        schema_version=schema_version,
        payload=record.to_dict(),
    )


def persist_approval_record(
    store: FileArtifactStore,
    record: ApprovalRecord,
    *,
    artifact_id: str,
    revision: int,
) -> ArtifactRef:
    verify_approval_record(store, record)
    try:
        return store.put(approval_record_envelope(record, artifact_id=artifact_id, revision=revision))
    except ArtifactError as exc:
        raise ApprovalError(f"failed to persist ApprovalRecord: {exc}") from exc


def load_approval_record(store: FileArtifactStore, ref: ArtifactRef) -> ApprovalRecord:
    require_artifact_ref(ref, "ref", ApprovalError)
    if ref.artifact_type != APPROVAL_RECORD_ARTIFACT_TYPE:
        raise ApprovalResolutionError(
            f"ApprovalRecord ref must have artifact_type={APPROVAL_RECORD_ARTIFACT_TYPE!r}"
        )
    try:
        envelope = store.get_ref(ref)
    except ArtifactError as exc:
        raise ApprovalResolutionError(f"failed to resolve ApprovalRecord: {ref!r}") from exc
    if envelope.schema_version != APPROVAL_RECORD_SCHEMA_VERSION:
        raise ApprovalResolutionError(
            "unsupported ApprovalRecord schema_version: "
            f"{envelope.schema_version}; supported={APPROVAL_RECORD_SCHEMA_VERSION}"
        )
    payload = envelope.payload
    if not isinstance(payload, dict):
        raise ApprovalResolutionError("persisted ApprovalRecord payload must be an object")
    try:
        record = ApprovalRecord.from_dict(payload)
    except ApprovalError as exc:
        raise ApprovalResolutionError(f"invalid persisted ApprovalRecord: {exc}") from exc
    if record.to_dict() != payload:
        raise ApprovalResolutionError("persisted ApprovalRecord payload is not in canonical semantic order")
    return record


def resolve_approval_ref(store: FileArtifactStore, approval_ref: ApprovalRef) -> ApprovalRecord:
    if not isinstance(approval_ref, ApprovalRef):
        raise ApprovalResolutionError("approval_ref must be an ApprovalRef")
    record = load_approval_record(store, approval_ref.approval_record_ref)
    if record.decision is not ApprovalDecision.APPROVE:
        raise ApprovalResolutionError("ApprovalRef must resolve to an APPROVE decision")
    if record.target_ref != approval_ref.target_ref:
        raise ApprovalResolutionError("ApprovalRef target does not match ApprovalRecord target")
    verify_approval_record(store, record)
    return record
