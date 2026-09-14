from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from short_drama.artifacts import ArtifactRef

from .errors import ChunkCoverageError, ChunkPlanningError, ChunkProfileError
from .source import SourceChapter, SourceDocument, SourceParagraph

SOURCE_CHUNK_ARTIFACT_TYPE = "source_chunk"
SOURCE_CHUNK_SCHEMA_VERSION = 1
CHUNK_MANIFEST_ARTIFACT_TYPE = "chunk_manifest"
CHUNK_MANIFEST_SCHEMA_VERSION = 1
CHUNK_PROFILE_SCHEMA_VERSION = 1
CHUNK_PLANNER_VERSION = "a2_chunk_planner_v1"
TOKEN_COUNTER_ID = "utf8-bytes-div3-v1"

_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_CHUNK_ID_RE = re.compile(r"^CH[0-9]{3,}_C[0-9]{3,}$")


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ChunkPlanningError(f"{field_name} must be a non-empty string without NUL")
    return value


def _require_exact_keys(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ChunkPlanningError(f"{name} must contain exactly: {', '.join(sorted(keys))}")
    return value


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ChunkPlanningError(f"{field_name} must be an integer >= 0")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    value = _require_nonnegative_int(value, field_name)
    if value < 1:
        raise ChunkPlanningError(f"{field_name} must be an integer >= 1")
    return value


@dataclass(frozen=True, slots=True)
class ChunkPlanningProfile:
    schema_version: int
    profile_id: str
    token_counter: str
    ownership_token_budget: int
    context_overlap_token_budget: int
    context_token_budget: int

    def __post_init__(self) -> None:
        if self.schema_version != CHUNK_PROFILE_SCHEMA_VERSION:
            raise ChunkProfileError(
                f"ChunkPlanningProfile.schema_version must be {CHUNK_PROFILE_SCHEMA_VERSION}"
            )
        if not isinstance(self.profile_id, str) or _PROFILE_ID_RE.fullmatch(self.profile_id) is None:
            raise ChunkProfileError("profile_id must be a safe lowercase storage identifier")
        if self.token_counter != TOKEN_COUNTER_ID:
            raise ChunkProfileError(f"token_counter must be {TOKEN_COUNTER_ID!r}")
        if (
            isinstance(self.ownership_token_budget, bool)
            or not isinstance(self.ownership_token_budget, int)
            or self.ownership_token_budget < 1
        ):
            raise ChunkProfileError("ownership_token_budget must be an integer >= 1")
        if (
            isinstance(self.context_overlap_token_budget, bool)
            or not isinstance(self.context_overlap_token_budget, int)
            or self.context_overlap_token_budget < 0
        ):
            raise ChunkProfileError("context_overlap_token_budget must be an integer >= 0")
        if (
            isinstance(self.context_token_budget, bool)
            or not isinstance(self.context_token_budget, int)
            or self.context_token_budget < 1
        ):
            raise ChunkProfileError("context_token_budget must be an integer >= 1")
        required = self.ownership_token_budget + 2 * self.context_overlap_token_budget
        if self.context_token_budget < required:
            raise ChunkProfileError(
                "context_token_budget must be >= ownership_token_budget + "
                "2 * context_overlap_token_budget"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "token_counter": self.token_counter,
            "ownership_token_budget": self.ownership_token_budget,
            "context_overlap_token_budget": self.context_overlap_token_budget,
            "context_token_budget": self.context_token_budget,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChunkPlanningProfile":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "profile_id",
            "token_counter",
            "ownership_token_budget",
            "context_overlap_token_budget",
            "context_token_budget",
        }:
            raise ChunkProfileError("ChunkPlanningProfile contains unexpected or missing fields")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ParagraphSpan:
    start: str
    end: str

    def __post_init__(self) -> None:
        _require_text(self.start, "span.start")
        _require_text(self.end, "span.end")

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ParagraphSpan":
        _require_exact_keys(value, {"start", "end"}, "ParagraphSpan")
        return cls(start=value["start"], end=value["end"])


@dataclass(frozen=True, slots=True)
class SourceChunk:
    schema_version: int
    chunk_id: str
    project_id: str
    document_id: str
    chapter_id: str
    source_document_ref: ArtifactRef
    context_span: ParagraphSpan
    ownership_span: ParagraphSpan
    paragraph_ids: tuple[str, ...]
    token_count_method: str
    context_token_count: int
    ownership_token_count: int

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_CHUNK_SCHEMA_VERSION:
            raise ChunkPlanningError(
                f"SourceChunk.schema_version must be {SOURCE_CHUNK_SCHEMA_VERSION}"
            )
        if not isinstance(self.chunk_id, str) or _CHUNK_ID_RE.fullmatch(self.chunk_id) is None:
            raise ChunkPlanningError(f"invalid chunk_id: {self.chunk_id!r}")
        _require_text(self.project_id, "project_id")
        _require_text(self.document_id, "document_id")
        _require_text(self.chapter_id, "chapter_id")
        if not isinstance(self.source_document_ref, ArtifactRef):
            raise ChunkPlanningError("source_document_ref must be ArtifactRef")
        if not isinstance(self.context_span, ParagraphSpan):
            raise ChunkPlanningError("context_span must be ParagraphSpan")
        if not isinstance(self.ownership_span, ParagraphSpan):
            raise ChunkPlanningError("ownership_span must be ParagraphSpan")
        paragraph_ids = tuple(self.paragraph_ids)
        if not paragraph_ids or any(not isinstance(item, str) or not item for item in paragraph_ids):
            raise ChunkPlanningError("paragraph_ids must be a non-empty tuple of IDs")
        if len(paragraph_ids) != len(set(paragraph_ids)):
            raise ChunkPlanningError("paragraph_ids must not contain duplicates")
        if paragraph_ids[0] != self.context_span.start or paragraph_ids[-1] != self.context_span.end:
            raise ChunkPlanningError("context_span must match paragraph_ids first/last values")
        if self.ownership_span.start not in paragraph_ids or self.ownership_span.end not in paragraph_ids:
            raise ChunkPlanningError("ownership_span endpoints must be within context paragraph_ids")
        if paragraph_ids.index(self.ownership_span.start) > paragraph_ids.index(self.ownership_span.end):
            raise ChunkPlanningError("ownership_span must not be reversed")
        if not self.chunk_id.startswith(self.chapter_id + "_C"):
            raise ChunkPlanningError("chunk_id must belong to chapter_id")
        if any(not pid.startswith(self.chapter_id + "_P") for pid in paragraph_ids):
            raise ChunkPlanningError("all paragraph_ids must belong to chapter_id")
        if self.source_document_ref.artifact_type != "source_document":
            raise ChunkPlanningError("source_document_ref must target source_document")
        if self.token_count_method != TOKEN_COUNTER_ID:
            raise ChunkPlanningError(f"token_count_method must be {TOKEN_COUNTER_ID!r}")
        _require_positive_int(self.context_token_count, "context_token_count")
        _require_positive_int(self.ownership_token_count, "ownership_token_count")
        object.__setattr__(self, "paragraph_ids", paragraph_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "chunk_id": self.chunk_id,
            "project_id": self.project_id,
            "document_id": self.document_id,
            "chapter_id": self.chapter_id,
            "source_document_ref": self.source_document_ref.to_dict(),
            "context_span": self.context_span.to_dict(),
            "ownership_span": self.ownership_span.to_dict(),
            "paragraph_ids": list(self.paragraph_ids),
            "token_count_method": self.token_count_method,
            "context_token_count": self.context_token_count,
            "ownership_token_count": self.ownership_token_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceChunk":
        _require_exact_keys(
            value,
            {
                "schema_version",
                "chunk_id",
                "project_id",
                "document_id",
                "chapter_id",
                "source_document_ref",
                "context_span",
                "ownership_span",
                "paragraph_ids",
                "token_count_method",
                "context_token_count",
                "ownership_token_count",
            },
            "SourceChunk",
        )
        if not isinstance(value["paragraph_ids"], list):
            raise ChunkPlanningError("SourceChunk.paragraph_ids must be a list")
        try:
            source_ref = ArtifactRef.from_dict(value["source_document_ref"])
        except Exception as exc:
            raise ChunkPlanningError(f"invalid source_document_ref: {exc}") from exc
        return cls(
            schema_version=value["schema_version"],
            chunk_id=value["chunk_id"],
            project_id=value["project_id"],
            document_id=value["document_id"],
            chapter_id=value["chapter_id"],
            source_document_ref=source_ref,
            context_span=ParagraphSpan.from_dict(value["context_span"]),
            ownership_span=ParagraphSpan.from_dict(value["ownership_span"]),
            paragraph_ids=tuple(value["paragraph_ids"]),
            token_count_method=value["token_count_method"],
            context_token_count=value["context_token_count"],
            ownership_token_count=value["ownership_token_count"],
        )


@dataclass(frozen=True, slots=True)
class ChunkCoverage:
    paragraphs_total: int
    owned_once: int
    unowned: int
    multiply_owned: int

    def __post_init__(self) -> None:
        for name in ("paragraphs_total", "owned_once", "unowned", "multiply_owned"):
            _require_nonnegative_int(getattr(self, name), name)

    @property
    def complete(self) -> bool:
        return (
            self.owned_once == self.paragraphs_total
            and self.unowned == 0
            and self.multiply_owned == 0
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "paragraphs_total": self.paragraphs_total,
            "owned_once": self.owned_once,
            "unowned": self.unowned,
            "multiply_owned": self.multiply_owned,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChunkCoverage":
        _require_exact_keys(
            value,
            {"paragraphs_total", "owned_once", "unowned", "multiply_owned"},
            "ChunkCoverage",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ChunkManifest:
    schema_version: int
    project_id: str
    document_id: str
    source_document_ref: ArtifactRef
    planner_version: str
    profile: ChunkPlanningProfile
    chunk_refs: tuple[ArtifactRef, ...]
    chunk_count: int
    coverage: ChunkCoverage
    state: str

    def __post_init__(self) -> None:
        if self.schema_version != CHUNK_MANIFEST_SCHEMA_VERSION:
            raise ChunkPlanningError(
                f"ChunkManifest.schema_version must be {CHUNK_MANIFEST_SCHEMA_VERSION}"
            )
        _require_text(self.project_id, "project_id")
        _require_text(self.document_id, "document_id")
        if not isinstance(self.source_document_ref, ArtifactRef):
            raise ChunkPlanningError("source_document_ref must be ArtifactRef")
        if self.planner_version != CHUNK_PLANNER_VERSION:
            raise ChunkPlanningError(f"planner_version must be {CHUNK_PLANNER_VERSION!r}")
        if not isinstance(self.profile, ChunkPlanningProfile):
            raise ChunkPlanningError("profile must be ChunkPlanningProfile")
        refs = tuple(self.chunk_refs)
        if any(not isinstance(ref, ArtifactRef) for ref in refs):
            raise ChunkPlanningError("chunk_refs must contain ArtifactRef values")
        if len(refs) != len(set(refs)):
            raise ChunkPlanningError("chunk_refs must not contain duplicates")
        if isinstance(self.chunk_count, bool) or not isinstance(self.chunk_count, int):
            raise ChunkPlanningError("chunk_count must be an integer")
        if self.chunk_count < 1:
            raise ChunkPlanningError("chunk_count must be >= 1")
        if self.chunk_count != len(refs):
            raise ChunkPlanningError("chunk_count must equal len(chunk_refs)")
        if self.source_document_ref.artifact_type != "source_document":
            raise ChunkPlanningError("source_document_ref must target source_document")
        if any(ref.artifact_type != SOURCE_CHUNK_ARTIFACT_TYPE for ref in refs):
            raise ChunkPlanningError("chunk_refs must target source_chunk artifacts")
        if not isinstance(self.coverage, ChunkCoverage):
            raise ChunkPlanningError("coverage must be ChunkCoverage")
        if self.coverage.paragraphs_total < 1:
            raise ChunkCoverageError("ChunkManifest must cover at least one paragraph")
        if self.state != "CHUNKING_COMPLETE":
            raise ChunkPlanningError("ChunkManifest.state must be CHUNKING_COMPLETE")
        if not self.coverage.complete:
            raise ChunkCoverageError("ChunkManifest coverage is not complete")
        object.__setattr__(self, "chunk_refs", refs)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "document_id": self.document_id,
            "source_document_ref": self.source_document_ref.to_dict(),
            "planner_version": self.planner_version,
            "profile": self.profile.to_dict(),
            "chunk_refs": [ref.to_dict() for ref in self.chunk_refs],
            "chunk_count": self.chunk_count,
            "coverage": self.coverage.to_dict(),
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChunkManifest":
        _require_exact_keys(
            value,
            {
                "schema_version",
                "project_id",
                "document_id",
                "source_document_ref",
                "planner_version",
                "profile",
                "chunk_refs",
                "chunk_count",
                "coverage",
                "state",
            },
            "ChunkManifest",
        )
        if not isinstance(value["chunk_refs"], list):
            raise ChunkPlanningError("ChunkManifest.chunk_refs must be a list")
        try:
            source_ref = ArtifactRef.from_dict(value["source_document_ref"])
            chunk_refs = tuple(ArtifactRef.from_dict(item) for item in value["chunk_refs"])
        except Exception as exc:
            raise ChunkPlanningError(f"invalid ChunkManifest ArtifactRef: {exc}") from exc
        return cls(
            schema_version=value["schema_version"],
            project_id=value["project_id"],
            document_id=value["document_id"],
            source_document_ref=source_ref,
            planner_version=value["planner_version"],
            profile=ChunkPlanningProfile.from_dict(value["profile"]),
            chunk_refs=chunk_refs,
            chunk_count=value["chunk_count"],
            coverage=ChunkCoverage.from_dict(value["coverage"]),
            state=value["state"],
        )


def estimate_tokens(text: str) -> int:
    if not isinstance(text, str):
        raise ChunkPlanningError("token estimation requires text")
    return max(1, math.ceil(len(text.encode("utf-8")) / 3))


def estimate_paragraphs(paragraphs: Sequence[SourceParagraph]) -> int:
    if not paragraphs:
        return 0
    return estimate_tokens("\n\n".join(paragraph.text_original for paragraph in paragraphs))


def _build_chapter_chunks(
    chapter: SourceChapter,
    *,
    project_id: str,
    document_id: str,
    source_document_ref: ArtifactRef,
    profile: ChunkPlanningProfile,
) -> list[SourceChunk]:
    paragraphs = list(chapter.paragraphs)
    ownership_ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(paragraphs):
        if estimate_paragraphs([paragraphs[start]]) > profile.ownership_token_budget:
            raise ChunkPlanningError(
                f"paragraph {paragraphs[start].paragraph_id} exceeds ownership_token_budget"
            )
        end = start
        while end + 1 < len(paragraphs):
            candidate = paragraphs[start : end + 2]
            if estimate_paragraphs(candidate) > profile.ownership_token_budget:
                break
            end += 1
        ownership_ranges.append((start, end))
        start = end + 1

    chunks: list[SourceChunk] = []
    for chunk_index, (owner_start, owner_end) in enumerate(ownership_ranges, start=1):
        context_start = owner_start
        context_end = owner_end

        while context_start > 0:
            candidate_start = context_start - 1
            left = paragraphs[candidate_start:owner_start]
            total = paragraphs[candidate_start : context_end + 1]
            if estimate_paragraphs(left) > profile.context_overlap_token_budget:
                break
            if estimate_paragraphs(total) > profile.context_token_budget:
                break
            context_start = candidate_start

        while context_end + 1 < len(paragraphs):
            candidate_end = context_end + 1
            right = paragraphs[owner_end + 1 : candidate_end + 1]
            total = paragraphs[context_start : candidate_end + 1]
            if estimate_paragraphs(right) > profile.context_overlap_token_budget:
                break
            if estimate_paragraphs(total) > profile.context_token_budget:
                break
            context_end = candidate_end

        context = paragraphs[context_start : context_end + 1]
        ownership = paragraphs[owner_start : owner_end + 1]
        chunks.append(
            SourceChunk(
                schema_version=SOURCE_CHUNK_SCHEMA_VERSION,
                chunk_id=f"{chapter.chapter_id}_C{chunk_index:03d}",
                project_id=project_id,
                document_id=document_id,
                chapter_id=chapter.chapter_id,
                source_document_ref=source_document_ref,
                context_span=ParagraphSpan(context[0].paragraph_id, context[-1].paragraph_id),
                ownership_span=ParagraphSpan(
                    ownership[0].paragraph_id, ownership[-1].paragraph_id
                ),
                paragraph_ids=tuple(paragraph.paragraph_id for paragraph in context),
                token_count_method=TOKEN_COUNTER_ID,
                context_token_count=estimate_paragraphs(context),
                ownership_token_count=estimate_paragraphs(ownership),
            )
        )
    return chunks


def ownership_ids_for_chunk(
    chunk: SourceChunk,
    chapter: SourceChapter,
) -> tuple[str, ...]:
    ids = [paragraph.paragraph_id for paragraph in chapter.paragraphs]
    try:
        start = ids.index(chunk.ownership_span.start)
        end = ids.index(chunk.ownership_span.end)
    except ValueError as exc:
        raise ChunkCoverageError(
            f"chunk {chunk.chunk_id} ownership span is not within source chapter"
        ) from exc
    if start > end:
        raise ChunkCoverageError(f"chunk {chunk.chunk_id} ownership span is reversed")
    return tuple(ids[start : end + 1])


def validate_chunks_against_source(
    source: SourceDocument,
    source_ref: ArtifactRef,
    profile: ChunkPlanningProfile,
    chunks: Iterable[SourceChunk],
) -> ChunkCoverage:
    chunk_list = list(chunks)
    chapter_map = source.chapter_index()
    ownership_count = {paragraph.paragraph_id: 0 for paragraph in source.paragraphs}

    seen_chunk_ids: set[str] = set()
    for chunk in chunk_list:
        if chunk.chunk_id in seen_chunk_ids:
            raise ChunkCoverageError(f"duplicate chunk_id: {chunk.chunk_id}")
        seen_chunk_ids.add(chunk.chunk_id)
        if chunk.project_id != source.project_id or chunk.document_id != source.document_id:
            raise ChunkCoverageError(f"chunk {chunk.chunk_id} source identity mismatch")
        if chunk.source_document_ref != source_ref:
            raise ChunkCoverageError(f"chunk {chunk.chunk_id} SourceDocument ref mismatch")
        chapter = chapter_map.get(chunk.chapter_id)
        if chapter is None:
            raise ChunkCoverageError(f"chunk {chunk.chunk_id} references unknown chapter")
        chapter_ids = [paragraph.paragraph_id for paragraph in chapter.paragraphs]
        try:
            context_positions = [chapter_ids.index(pid) for pid in chunk.paragraph_ids]
        except ValueError as exc:
            raise ChunkCoverageError(
                f"chunk {chunk.chunk_id} contains unknown paragraph ID"
            ) from exc
        if context_positions != list(range(context_positions[0], context_positions[-1] + 1)):
            raise ChunkCoverageError(f"chunk {chunk.chunk_id} context is not contiguous")
        ownership_ids = ownership_ids_for_chunk(chunk, chapter)
        context_set = set(chunk.paragraph_ids)
        if any(pid not in context_set for pid in ownership_ids):
            raise ChunkCoverageError(
                f"chunk {chunk.chunk_id} ownership is not contained in context"
            )

        paragraph_map = {paragraph.paragraph_id: paragraph for paragraph in chapter.paragraphs}
        context_paragraphs = [paragraph_map[pid] for pid in chunk.paragraph_ids]
        ownership_paragraphs = [paragraph_map[pid] for pid in ownership_ids]
        if estimate_paragraphs(context_paragraphs) != chunk.context_token_count:
            raise ChunkCoverageError(f"chunk {chunk.chunk_id} context token count mismatch")
        if estimate_paragraphs(ownership_paragraphs) != chunk.ownership_token_count:
            raise ChunkCoverageError(f"chunk {chunk.chunk_id} ownership token count mismatch")
        if chunk.context_token_count > profile.context_token_budget:
            raise ChunkCoverageError(f"chunk {chunk.chunk_id} exceeds context token budget")
        if chunk.ownership_token_count > profile.ownership_token_budget:
            raise ChunkCoverageError(f"chunk {chunk.chunk_id} exceeds ownership token budget")
        for pid in ownership_ids:
            ownership_count[pid] += 1

    owned_once = sum(count == 1 for count in ownership_count.values())
    unowned = sum(count == 0 for count in ownership_count.values())
    multiply_owned = sum(count > 1 for count in ownership_count.values())
    coverage = ChunkCoverage(
        paragraphs_total=len(ownership_count),
        owned_once=owned_once,
        unowned=unowned,
        multiply_owned=multiply_owned,
    )
    if not coverage.complete:
        raise ChunkCoverageError(
            "chunk coverage is incomplete: "
            f"owned_once={coverage.owned_once}, unowned={coverage.unowned}, "
            f"multiply_owned={coverage.multiply_owned}"
        )
    return coverage


def plan_chunks(
    source: SourceDocument,
    source_document_ref: ArtifactRef,
    profile: ChunkPlanningProfile,
) -> tuple[tuple[SourceChunk, ...], ChunkCoverage]:
    if not isinstance(source, SourceDocument):
        raise ChunkPlanningError("source must be SourceDocument")
    if not isinstance(source_document_ref, ArtifactRef):
        raise ChunkPlanningError("source_document_ref must be ArtifactRef")
    if source_document_ref.artifact_type != "source_document":
        raise ChunkPlanningError("source_document_ref must target source_document")
    if not isinstance(profile, ChunkPlanningProfile):
        raise ChunkPlanningError("profile must be ChunkPlanningProfile")

    chunks: list[SourceChunk] = []
    for chapter in source.chapters:
        chunks.extend(
            _build_chapter_chunks(
                chapter,
                project_id=source.project_id,
                document_id=source.document_id,
                source_document_ref=source_document_ref,
                profile=profile,
            )
        )
    coverage = validate_chunks_against_source(
        source, source_document_ref, profile, chunks
    )
    return tuple(chunks), coverage
