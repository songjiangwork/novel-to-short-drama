from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from short_drama.artifacts import (
    ArtifactError,
    ArtifactRef,
    FileArtifactStore,
    ImmutableArtifactEnvelope,
    content_hash,
)

from ._common import (
    artifact_ref_sort_key,
    require_artifact_ref,
    require_exact_keys,
    require_text,
)
from .errors import ValidationModelError
from .lineage import LineageRef

VALIDATION_REPORT_ARTIFACT_TYPE = "validation_report"
VALIDATION_REPORT_SCHEMA_VERSION = 1


class ValidationSeverity(str, Enum):
    BLOCKING = "BLOCKING"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    WARNING = "WARNING"


class ValidationResult(str, Enum):
    FAIL = "FAIL"
    PASS_WITH_REVIEW_ITEMS = "PASS_WITH_REVIEW_ITEMS"
    PASS = "PASS"


_SEVERITY_RANK = {
    ValidationSeverity.BLOCKING: 0,
    ValidationSeverity.REVIEW_REQUIRED: 1,
    ValidationSeverity.WARNING: 2,
}


def _coerce_enum(value: Any, enum_type: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValidationModelError(f"{field_name} must be one of: {allowed}") from exc


def _normalize_path(value: Any) -> tuple[str | int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValidationModelError("path must be null or a list of string/integer components")
    normalized: list[str | int] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (str, int)):
            raise ValidationModelError("path components must be strings or integers")
        if isinstance(component, str):
            require_text(component, "path component", ValidationModelError)
        normalized.append(component)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    finding_id: str
    code: str
    severity: ValidationSeverity
    owner_stage: str
    repair_route: str
    message: str
    artifact_refs: tuple[ArtifactRef, ...] = ()
    path: tuple[str | int, ...] | None = None

    def __post_init__(self) -> None:
        require_text(self.finding_id, "finding_id", ValidationModelError)
        require_text(self.code, "code", ValidationModelError)
        require_text(self.owner_stage, "owner_stage", ValidationModelError)
        require_text(self.repair_route, "repair_route", ValidationModelError)
        require_text(self.message, "message", ValidationModelError)
        object.__setattr__(self, "severity", _coerce_enum(self.severity, ValidationSeverity, "severity"))

        refs = tuple(self.artifact_refs)
        for ref in refs:
            require_artifact_ref(ref, "artifact_refs item", ValidationModelError)
        refs = tuple(sorted(refs, key=artifact_ref_sort_key))
        if len(set(refs)) != len(refs):
            raise ValidationModelError("artifact_refs must not contain duplicates")
        object.__setattr__(self, "artifact_refs", refs)
        object.__setattr__(self, "path", _normalize_path(self.path))

    def sort_key(self) -> tuple[object, ...]:
        path_key = tuple(
            ("int", component) if isinstance(component, int) else ("str", component)
            for component in (self.path or ())
        )
        return (
            _SEVERITY_RANK[self.severity],
            self.owner_stage,
            self.repair_route,
            self.code,
            self.finding_id,
            self.message,
            tuple(artifact_ref_sort_key(ref) for ref in self.artifact_refs),
            path_key,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "finding_id": self.finding_id,
            "code": self.code,
            "severity": self.severity.value,
            "owner_stage": self.owner_stage,
            "repair_route": self.repair_route,
            "message": self.message,
            "artifact_refs": [ref.to_dict() for ref in self.artifact_refs],
            "path": None if self.path is None else list(self.path),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ValidationFinding":
        require_exact_keys(
            value,
            {
                "finding_id",
                "code",
                "severity",
                "owner_stage",
                "repair_route",
                "message",
                "artifact_refs",
                "path",
            },
            "ValidationFinding",
            ValidationModelError,
        )
        if not isinstance(value["artifact_refs"], list):
            raise ValidationModelError("artifact_refs must be a list")
        try:
            refs = tuple(ArtifactRef.from_dict(item) for item in value["artifact_refs"])
        except (TypeError, ValueError) as exc:
            raise ValidationModelError(f"invalid ValidationFinding artifact_refs: {exc}") from exc
        return cls(
            finding_id=value["finding_id"],
            code=value["code"],
            severity=value["severity"],
            owner_stage=value["owner_stage"],
            repair_route=value["repair_route"],
            message=value["message"],
            artifact_refs=refs,
            path=value["path"],
        )


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    blocking_count: int
    review_required_count: int
    warning_count: int
    result: ValidationResult

    def __post_init__(self) -> None:
        for field_name in ("blocking_count", "review_required_count", "warning_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationModelError(f"{field_name} must be an integer >= 0")
        object.__setattr__(self, "result", _coerce_enum(self.result, ValidationResult, "result"))

    def to_dict(self) -> dict[str, object]:
        return {
            "blocking_count": self.blocking_count,
            "review_required_count": self.review_required_count,
            "warning_count": self.warning_count,
            "result": self.result.value,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ValidationSummary":
        require_exact_keys(
            value,
            {"blocking_count", "review_required_count", "warning_count", "result"},
            "ValidationSummary",
            ValidationModelError,
        )
        return cls(
            blocking_count=value["blocking_count"],
            review_required_count=value["review_required_count"],
            warning_count=value["warning_count"],
            result=value["result"],
        )


def derive_validation_summary(findings: Iterable[ValidationFinding]) -> ValidationSummary:
    blocking = review = warning = 0
    for finding in findings:
        if not isinstance(finding, ValidationFinding):
            raise ValidationModelError("findings must contain ValidationFinding values")
        if finding.severity is ValidationSeverity.BLOCKING:
            blocking += 1
        elif finding.severity is ValidationSeverity.REVIEW_REQUIRED:
            review += 1
        else:
            warning += 1
    if blocking:
        result = ValidationResult.FAIL
    elif review:
        result = ValidationResult.PASS_WITH_REVIEW_ITEMS
    else:
        result = ValidationResult.PASS
    return ValidationSummary(blocking, review, warning, result)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    validated_refs: tuple[LineageRef, ...]
    findings: tuple[ValidationFinding, ...]

    def __post_init__(self) -> None:
        refs = tuple(self.validated_refs)
        if any(not isinstance(ref, LineageRef) for ref in refs):
            raise ValidationModelError("validated_refs must contain LineageRef values")
        refs = tuple(sorted(refs, key=LineageRef.sort_key))
        if len({(ref.role, ref.artifact_ref) for ref in refs}) != len(refs):
            raise ValidationModelError("validated_refs must not contain duplicate role/ref pairs")
        object.__setattr__(self, "validated_refs", refs)

        findings = tuple(self.findings)
        if any(not isinstance(finding, ValidationFinding) for finding in findings):
            raise ValidationModelError("findings must contain ValidationFinding values")
        findings = tuple(sorted(findings, key=ValidationFinding.sort_key))
        ids = [finding.finding_id for finding in findings]
        if len(set(ids)) != len(ids):
            raise ValidationModelError("finding_id values must be unique within a ValidationReport")
        object.__setattr__(self, "findings", findings)

    @property
    def summary(self) -> ValidationSummary:
        return derive_validation_summary(self.findings)

    def to_dict(self) -> dict[str, object]:
        return {
            "validated_refs": [ref.to_dict() for ref in self.validated_refs],
            "findings": [finding.to_dict() for finding in self.findings],
            "summary": self.summary.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ValidationReport":
        require_exact_keys(
            value,
            {"validated_refs", "findings", "summary"},
            "ValidationReport",
            ValidationModelError,
        )
        if not isinstance(value["validated_refs"], list) or not isinstance(value["findings"], list):
            raise ValidationModelError("validated_refs and findings must be lists")
        report = cls(
            validated_refs=tuple(LineageRef.from_dict(item) for item in value["validated_refs"]),
            findings=tuple(ValidationFinding.from_dict(item) for item in value["findings"]),
        )
        serialized_summary = ValidationSummary.from_dict(value["summary"])
        if serialized_summary != report.summary:
            raise ValidationModelError("serialized ValidationReport summary is inconsistent with findings")
        return report


def validation_report_envelope(
    report: ValidationReport,
    *,
    artifact_id: str,
    revision: int,
    schema_version: int = VALIDATION_REPORT_SCHEMA_VERSION,
) -> ImmutableArtifactEnvelope:
    if not isinstance(report, ValidationReport):
        raise ValidationModelError("report must be a ValidationReport")
    return ImmutableArtifactEnvelope.create(
        artifact_type=VALIDATION_REPORT_ARTIFACT_TYPE,
        artifact_id=artifact_id,
        revision=revision,
        schema_version=schema_version,
        payload=report.to_dict(),
    )


def persist_validation_report(
    store: FileArtifactStore,
    report: ValidationReport,
    *,
    artifact_id: str,
    revision: int,
) -> ArtifactRef:
    try:
        return store.put(validation_report_envelope(report, artifact_id=artifact_id, revision=revision))
    except ArtifactError as exc:
        raise ValidationModelError(f"failed to persist ValidationReport: {exc}") from exc


def load_validation_report(store: FileArtifactStore, ref: ArtifactRef) -> ValidationReport:
    require_artifact_ref(ref, "ref", ValidationModelError)
    if ref.artifact_type != VALIDATION_REPORT_ARTIFACT_TYPE:
        raise ValidationModelError(
            f"ValidationReport ref must have artifact_type={VALIDATION_REPORT_ARTIFACT_TYPE!r}"
        )
    try:
        envelope = store.get_ref(ref)
    except ArtifactError as exc:
        raise ValidationModelError(f"failed to resolve ValidationReport: {ref!r}") from exc
    if envelope.schema_version != VALIDATION_REPORT_SCHEMA_VERSION:
        raise ValidationModelError(
            "unsupported ValidationReport schema_version: "
            f"{envelope.schema_version}; supported={VALIDATION_REPORT_SCHEMA_VERSION}"
        )
    payload = envelope.payload
    if not isinstance(payload, dict):
        raise ValidationModelError("persisted ValidationReport payload must be an object")
    report = ValidationReport.from_dict(payload)
    if report.to_dict() != payload:
        raise ValidationModelError("persisted ValidationReport payload is not in canonical semantic order")
    return report


def jsonschema_findings(
    instance: object,
    schema: dict[str, Any],
    *,
    code_prefix: str,
    owner_stage: str,
    repair_route: str,
    severity: ValidationSeverity = ValidationSeverity.BLOCKING,
    artifact_refs: Iterable[ArtifactRef] = (),
) -> tuple[ValidationFinding, ...]:
    require_text(code_prefix, "code_prefix", ValidationModelError)
    require_text(owner_stage, "owner_stage", ValidationModelError)
    require_text(repair_route, "repair_route", ValidationModelError)
    severity = _coerce_enum(severity, ValidationSeverity, "severity")
    refs = tuple(sorted(tuple(artifact_refs), key=artifact_ref_sort_key))
    for ref in refs:
        require_artifact_ref(ref, "artifact_refs item", ValidationModelError)

    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
    except Exception as exc:
        raise ValidationModelError(f"invalid JSON Schema: {exc}") from exc

    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            tuple(str(part) for part in error.absolute_schema_path),
            str(error.validator),
            error.message,
        ),
    )
    findings: list[ValidationFinding] = []
    for error in errors:
        path = tuple(error.absolute_path)
        schema_path = tuple(str(part) for part in error.absolute_schema_path)
        validator_name = str(error.validator)
        code = f"{code_prefix}.{validator_name}"
        identity_material = {
            "code": code,
            "path": list(path),
            "schema_path": list(schema_path),
            "message": error.message,
        }
        finding_id = f"{code_prefix}-{content_hash(identity_material)[:20]}"
        findings.append(
            ValidationFinding(
                finding_id=finding_id,
                code=code,
                severity=severity,
                owner_stage=owner_stage,
                repair_route=repair_route,
                message=error.message,
                artifact_refs=refs,
                path=path,
            )
        )
    return tuple(sorted(findings, key=ValidationFinding.sort_key))
