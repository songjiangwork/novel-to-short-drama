from __future__ import annotations

import codecs
import hashlib
import importlib.metadata
import io
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .errors import (
    SourceDecodeError,
    SourceLanguageError,
    SourcePdfError,
    SourceStructureError,
)

SOURCE_DOCUMENT_ARTIFACT_TYPE = "source_document"
SOURCE_DOCUMENT_SCHEMA_VERSION = 1
SOURCE_PARSER_ID = "short_drama_source_ingestion_v1"
SOURCE_PARSER_VERSION = "1"
LANGUAGE_DETECTOR_ID = "langid-1.1.6"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHAPTER_ID_RE = re.compile(r"^CH[0-9]{3,}$")
_PARAGRAPH_ID_RE = re.compile(r"^CH[0-9]{3,}_P[0-9]{4,}$")

_ENGLISH_CHAPTER_RE = re.compile(
    r"^(?P<kind>chapter|part)\s+"
    r"(?P<number>(?:\d+|[ivxlcdm]+|[a-z]+(?:[- ][a-z]+)*))"
    r"(?:\s*[:.\-–—]\s*|\s+)?(?P<title>.*)$",
    re.IGNORECASE,
)
_ENGLISH_SPECIAL_RE = re.compile(
    r"^(?P<kind>prologue|epilogue)(?:\s*[:.\-–—]\s*|\s+)?(?P<title>.*)$",
    re.IGNORECASE,
)
_CHINESE_HEADING_RE = re.compile(
    r"^(?P<kind>"
    r"第[零〇一二三四五六七八九十百千万两\d]+[章节卷部]"
    r"|序章|序言|楔子|尾声|后记"
    r")(?P<title>(?:\s*[:：.\-–—]?\s*.*)?)$"
)


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SourceStructureError(f"{field_name} must be a non-empty string without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SourceStructureError(f"{field_name} must contain valid UTF-8 text") from exc
    return value


def _require_exact_keys(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise SourceStructureError(f"{name} must contain exactly: {', '.join(sorted(keys))}")
    return value


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceStructureError(f"{field_name} must be an integer >= 0")
    return value


@dataclass(frozen=True, slots=True)
class SourceParagraph:
    paragraph_id: str
    text_original: str
    source_pages: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _require_text(self.paragraph_id, "paragraph_id")
        if _PARAGRAPH_ID_RE.fullmatch(self.paragraph_id) is None:
            raise SourceStructureError(f"invalid paragraph_id: {self.paragraph_id!r}")
        _require_text(self.text_original, "text_original")
        if self.source_pages is not None:
            pages = tuple(self.source_pages)
            if not pages:
                raise SourceStructureError("source_pages must be null or a non-empty page list")
            if any(isinstance(page, bool) or not isinstance(page, int) or page < 1 for page in pages):
                raise SourceStructureError("source_pages must contain positive integers")
            if tuple(sorted(set(pages))) != pages:
                raise SourceStructureError("source_pages must be strictly increasing and unique")
            object.__setattr__(self, "source_pages", pages)

    def to_dict(self) -> dict[str, object]:
        return {
            "paragraph_id": self.paragraph_id,
            "text_original": self.text_original,
            "source_pages": None if self.source_pages is None else list(self.source_pages),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceParagraph":
        _require_exact_keys(value, {"paragraph_id", "text_original", "source_pages"}, "SourceParagraph")
        pages = value["source_pages"]
        if pages is not None and not isinstance(pages, list):
            raise SourceStructureError("SourceParagraph.source_pages must be null or a list")
        return cls(
            paragraph_id=value["paragraph_id"],
            text_original=value["text_original"],
            source_pages=None if pages is None else tuple(pages),
        )


@dataclass(frozen=True, slots=True)
class SourceChapter:
    chapter_id: str
    title_original: str | None
    heading_kind: str
    paragraphs: tuple[SourceParagraph, ...]

    def __post_init__(self) -> None:
        _require_text(self.chapter_id, "chapter_id")
        if _CHAPTER_ID_RE.fullmatch(self.chapter_id) is None:
            raise SourceStructureError(f"invalid chapter_id: {self.chapter_id!r}")
        if self.title_original is not None:
            _require_text(self.title_original, "title_original")
        if self.heading_kind not in {"chapter", "part", "prologue", "epilogue", "synthetic"}:
            raise SourceStructureError(f"unsupported heading_kind: {self.heading_kind!r}")
        paragraphs = tuple(self.paragraphs)
        if not paragraphs:
            raise SourceStructureError(f"chapter {self.chapter_id} must contain at least one paragraph")
        expected_prefix = f"{self.chapter_id}_P"
        ids = [paragraph.paragraph_id for paragraph in paragraphs]
        if len(ids) != len(set(ids)):
            raise SourceStructureError(f"chapter {self.chapter_id} contains duplicate paragraph IDs")
        if any(not pid.startswith(expected_prefix) for pid in ids):
            raise SourceStructureError(f"paragraph ID does not belong to {self.chapter_id}")
        expected_ids = [
            f"{self.chapter_id}_P{index:04d}"
            for index in range(1, len(paragraphs) + 1)
        ]
        if ids != expected_ids:
            raise SourceStructureError(
                f"chapter {self.chapter_id} paragraph IDs must be deterministic and sequential"
            )
        if self.heading_kind == "synthetic" and self.title_original is not None:
            raise SourceStructureError("synthetic chapter must not have title_original")
        if self.heading_kind != "synthetic" and self.title_original is None:
            raise SourceStructureError("detected chapter heading requires title_original")
        object.__setattr__(self, "paragraphs", paragraphs)

    def to_dict(self) -> dict[str, object]:
        return {
            "chapter_id": self.chapter_id,
            "title_original": self.title_original,
            "heading_kind": self.heading_kind,
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceChapter":
        _require_exact_keys(
            value,
            {"chapter_id", "title_original", "heading_kind", "paragraphs"},
            "SourceChapter",
        )
        if not isinstance(value["paragraphs"], list):
            raise SourceStructureError("SourceChapter.paragraphs must be a list")
        return cls(
            chapter_id=value["chapter_id"],
            title_original=value["title_original"],
            heading_kind=value["heading_kind"],
            paragraphs=tuple(SourceParagraph.from_dict(item) for item in value["paragraphs"]),
        )


@dataclass(frozen=True, slots=True)
class SourceInfo:
    type: str
    path: str
    raw_sha256: str
    byte_size: int
    declared_language: str
    detected_language: str
    language_detector: str

    def __post_init__(self) -> None:
        if self.type not in {"txt", "pdf"}:
            raise SourceStructureError("source.type must be 'txt' or 'pdf'")
        _require_text(self.path, "source.path")
        if not isinstance(self.raw_sha256, str) or _SHA256_RE.fullmatch(self.raw_sha256) is None:
            raise SourceStructureError("source.raw_sha256 must be lowercase SHA-256 hex")
        _require_nonnegative_int(self.byte_size, "source.byte_size")
        _require_text(self.declared_language, "source.declared_language")
        _require_text(self.detected_language, "source.detected_language")
        _require_text(self.language_detector, "source.language_detector")

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "path": self.path,
            "raw_sha256": self.raw_sha256,
            "byte_size": self.byte_size,
            "declared_language": self.declared_language,
            "detected_language": self.detected_language,
            "language_detector": self.language_detector,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceInfo":
        _require_exact_keys(
            value,
            {
                "type",
                "path",
                "raw_sha256",
                "byte_size",
                "declared_language",
                "detected_language",
                "language_detector",
            },
            "SourceInfo",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class NormalizationInfo:
    input_encoding: str
    newline: str
    parser_id: str
    parser_version: str

    def __post_init__(self) -> None:
        _require_text(self.input_encoding, "normalization.input_encoding")
        if self.newline != "LF":
            raise SourceStructureError("normalization.newline must be LF")
        _require_text(self.parser_id, "normalization.parser_id")
        _require_text(self.parser_version, "normalization.parser_version")

    def to_dict(self) -> dict[str, str]:
        return {
            "input_encoding": self.input_encoding,
            "newline": self.newline,
            "parser_id": self.parser_id,
            "parser_version": self.parser_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NormalizationInfo":
        _require_exact_keys(
            value,
            {"input_encoding", "newline", "parser_id", "parser_version"},
            "NormalizationInfo",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    schema_version: int
    project_id: str
    document_id: str
    source: SourceInfo
    normalization: NormalizationInfo
    chapters: tuple[SourceChapter, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_DOCUMENT_SCHEMA_VERSION:
            raise SourceStructureError(
                f"SourceDocument.schema_version must be {SOURCE_DOCUMENT_SCHEMA_VERSION}"
            )
        _require_text(self.project_id, "project_id")
        _require_text(self.document_id, "document_id")
        if not isinstance(self.source, SourceInfo):
            raise SourceStructureError("source must be SourceInfo")
        if not isinstance(self.normalization, NormalizationInfo):
            raise SourceStructureError("normalization must be NormalizationInfo")
        chapters = tuple(self.chapters)
        if not chapters:
            raise SourceStructureError("SourceDocument must contain at least one chapter")
        chapter_ids = [chapter.chapter_id for chapter in chapters]
        if len(chapter_ids) != len(set(chapter_ids)):
            raise SourceStructureError("SourceDocument chapter IDs must be unique")
        expected_chapter_ids = [
            f"CH{index:03d}" for index in range(1, len(chapters) + 1)
        ]
        if chapter_ids != expected_chapter_ids:
            raise SourceStructureError(
                "SourceDocument chapter IDs must be deterministic and sequential"
            )
        paragraph_ids = [
            paragraph.paragraph_id
            for chapter in chapters
            for paragraph in chapter.paragraphs
        ]
        if len(paragraph_ids) != len(set(paragraph_ids)):
            raise SourceStructureError("SourceDocument paragraph IDs must be globally unique")
        object.__setattr__(self, "chapters", chapters)

    @property
    def paragraphs(self) -> tuple[SourceParagraph, ...]:
        return tuple(
            paragraph
            for chapter in self.chapters
            for paragraph in chapter.paragraphs
        )

    def paragraph_index(self) -> dict[str, SourceParagraph]:
        return {paragraph.paragraph_id: paragraph for paragraph in self.paragraphs}

    def chapter_index(self) -> dict[str, SourceChapter]:
        return {chapter.chapter_id: chapter for chapter in self.chapters}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "document_id": self.document_id,
            "source": self.source.to_dict(),
            "normalization": self.normalization.to_dict(),
            "structure": {
                "chapters": [chapter.to_dict() for chapter in self.chapters],
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceDocument":
        _require_exact_keys(
            value,
            {"schema_version", "project_id", "document_id", "source", "normalization", "structure"},
            "SourceDocument",
        )
        structure = value["structure"]
        _require_exact_keys(structure, {"chapters"}, "SourceDocument.structure")
        if not isinstance(structure["chapters"], list):
            raise SourceStructureError("SourceDocument.structure.chapters must be a list")
        return cls(
            schema_version=value["schema_version"],
            project_id=value["project_id"],
            document_id=value["document_id"],
            source=SourceInfo.from_dict(value["source"]),
            normalization=NormalizationInfo.from_dict(value["normalization"]),
            chapters=tuple(SourceChapter.from_dict(item) for item in structure["chapters"]),
        )


def _heading_kind(text: str) -> str | None:
    if "\n" in text:
        return None
    stripped = text.strip()
    special = _ENGLISH_SPECIAL_RE.fullmatch(stripped)
    if special:
        return special.group("kind").lower()
    english = _ENGLISH_CHAPTER_RE.fullmatch(stripped)
    if english:
        return english.group("kind").lower()
    chinese = _CHINESE_HEADING_RE.fullmatch(stripped)
    if chinese:
        raw = chinese.group("kind")
        if raw in {"序章", "序言", "楔子"}:
            return "prologue"
        if raw in {"尾声", "后记"}:
            return "epilogue"
        if raw.endswith(("卷", "部")):
            return "part"
        return "chapter"
    return None


@dataclass(frozen=True, slots=True)
class _SourceLine:
    text: str
    source_page: int | None
    page_break: bool = False


def _lines_from_text(text: str, source_page: int | None = None) -> list[_SourceLine]:
    text = normalize_newlines(text)
    return [_SourceLine(line, source_page) for line in text.split("\n")]


def _paragraphize_segment(lines: list[_SourceLine]) -> list[tuple[str, tuple[int, ...] | None]]:
    while lines and not lines[0].text.strip():
        lines = lines[1:]
    while lines and not lines[-1].text.strip():
        lines = lines[:-1]
    if not lines:
        return []

    has_blank_separator = any(not line.text.strip() for line in lines)
    groups: list[list[_SourceLine]] = []
    if not has_blank_separator:
        groups = [[line] for line in lines if line.text.strip()]
    else:
        current: list[_SourceLine] = []
        for line in lines:
            if not line.text.strip():
                if current:
                    groups.append(current)
                    current = []
            else:
                current.append(line)
        if current:
            groups.append(current)

    paragraphs: list[tuple[str, tuple[int, ...] | None]] = []
    for group in groups:
        text = "\n".join(line.text for line in group).strip()
        if not text:
            continue
        pages = tuple(
            sorted(
                {
                    line.source_page
                    for line in group
                    if line.source_page is not None
                }
            )
        )
        paragraphs.append((text, pages or None))
    return paragraphs


def _paragraphize(lines: list[_SourceLine]) -> list[tuple[str, tuple[int, ...] | None]]:
    # PDF page boundaries are hard paragraphization boundaries. They are not
    # interpreted as semantic blank lines inside either page.
    segments: list[list[_SourceLine]] = []
    current: list[_SourceLine] = []
    for line in lines:
        if line.page_break:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(line)
    if current:
        segments.append(current)

    paragraphs: list[tuple[str, tuple[int, ...] | None]] = []
    for segment in segments:
        paragraphs.extend(_paragraphize_segment(segment))
    return paragraphs


def _parse_lines(lines: Iterable[_SourceLine]) -> tuple[SourceChapter, ...]:
    chapter_specs: list[
        tuple[str | None, str, list[tuple[str, tuple[int, ...] | None]]]
    ] = []
    current_title: str | None = None
    current_kind = "synthetic"
    current_lines: list[_SourceLine] = []

    def flush() -> None:
        nonlocal current_title, current_kind, current_lines
        paragraphs = _paragraphize(current_lines)
        if paragraphs:
            chapter_specs.append((current_title, current_kind, paragraphs))
        current_title = None
        current_kind = "synthetic"
        current_lines = []

    for line in lines:
        if line.page_break:
            current_lines.append(line)
            continue
        stripped = line.text.strip()
        kind = _heading_kind(stripped) if stripped else None
        if kind is not None:
            flush()
            current_title = stripped
            current_kind = kind
            continue
        current_lines.append(line)
    flush()

    if not chapter_specs:
        raise SourceStructureError("source contains no usable paragraphs")

    chapters: list[SourceChapter] = []
    for chapter_index, (title, kind, paragraph_specs) in enumerate(chapter_specs, start=1):
        chapter_id = f"CH{chapter_index:03d}"
        paragraphs = tuple(
            SourceParagraph(
                paragraph_id=f"{chapter_id}_P{paragraph_index:04d}",
                text_original=text,
                source_pages=pages,
            )
            for paragraph_index, (text, pages) in enumerate(paragraph_specs, start=1)
        )
        chapters.append(
            SourceChapter(
                chapter_id=chapter_id,
                title_original=title,
                heading_kind=kind,
                paragraphs=paragraphs,
            )
        )
    return tuple(chapters)


def raw_sha256(raw: bytes) -> str:
    if not isinstance(raw, bytes):
        raise SourceStructureError("raw source must be bytes")
    return hashlib.sha256(raw).hexdigest()


def normalize_newlines(text: str) -> str:
    if not isinstance(text, str):
        raise SourceDecodeError("decoded source must be text")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def decode_txt_bytes(raw: bytes, explicit_encoding: str | None = None) -> tuple[str, str]:
    if not isinstance(raw, bytes):
        raise SourceDecodeError("TXT source must be bytes")
    if explicit_encoding is not None:
        try:
            encoding = codecs.lookup(explicit_encoding).name
        except LookupError as exc:
            raise SourceDecodeError(f"unknown explicit encoding: {explicit_encoding!r}") from exc
    elif raw.startswith(codecs.BOM_UTF32_LE) or raw.startswith(codecs.BOM_UTF32_BE):
        encoding = "utf-32"
    elif raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        encoding = "utf-16"
    elif raw.startswith(codecs.BOM_UTF8):
        encoding = "utf-8-sig"
    else:
        encoding = "utf-8"
    try:
        return normalize_newlines(raw.decode(encoding, errors="strict")), encoding
    except (UnicodeDecodeError, LookupError) as exc:
        hint = " use --encoding explicitly for non-UTF source text" if explicit_encoding is None else ""
        raise SourceDecodeError(f"failed to decode TXT as {encoding}:{hint} {exc}") from exc


def extract_txt_structure(raw: bytes, explicit_encoding: str | None = None) -> tuple[
    tuple[SourceChapter, ...], str, str
]:
    text, encoding = decode_txt_bytes(raw, explicit_encoding)
    chapters = _parse_lines(_lines_from_text(text))
    return chapters, encoding, SOURCE_PARSER_VERSION


def _load_pdf_reader():
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SourcePdfError("pypdf is required for text-based PDF ingestion") from exc
    return PdfReader


def _package_version(name: str, fallback: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def extract_pdf_structure(raw: bytes) -> tuple[tuple[SourceChapter, ...], str, str]:
    reader_cls = _load_pdf_reader()
    try:
        reader = reader_cls(io.BytesIO(raw))
    except Exception as exc:
        raise SourcePdfError(f"failed to open PDF: {exc}") from exc

    lines: list[_SourceLine] = []
    usable_chars = 0
    try:
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = normalize_newlines(page_text)
            usable_chars += len(page_text.strip())
            if page_number > 1:
                lines.append(_SourceLine("", None, page_break=True))
            lines.extend(_lines_from_text(page_text, source_page=page_number))
    except Exception as exc:
        raise SourcePdfError(f"failed to extract PDF text: {exc}") from exc

    if usable_chars == 0:
        raise SourcePdfError(
            "PDF contains no extractable text; OCR/image-only PDF support is outside the v1.2 MVP"
        )
    chapters = _parse_lines(lines)
    pypdf_version = _package_version("pypdf", "unknown")
    return chapters, "pdf-text-extraction", f"{SOURCE_PARSER_VERSION};pypdf={pypdf_version}"


def detect_language(text: str) -> tuple[str, str]:
    if not isinstance(text, str) or not text.strip():
        raise SourceLanguageError("cannot detect language from empty text")
    try:
        import langid
    except ImportError as exc:
        raise SourceLanguageError("langid==1.1.6 is required for source-language detection") from exc
    try:
        language, _score = langid.classify(text)
    except Exception as exc:
        raise SourceLanguageError(f"source-language detection failed: {exc}") from exc
    if not isinstance(language, str) or not language:
        raise SourceLanguageError("source-language detector returned no language")
    return language, LANGUAGE_DETECTOR_ID


def base_language(language: str) -> str:
    _require_text(language, "language")
    return re.split(r"[-_]", language, maxsplit=1)[0].lower()


def build_source_document(
    *,
    project_id: str,
    document_id: str,
    source_type: str,
    source_path: str,
    raw: bytes,
    declared_language: str,
    explicit_encoding: str | None = None,
) -> SourceDocument:
    if source_type == "txt":
        chapters, input_encoding, parser_version = extract_txt_structure(raw, explicit_encoding)
    elif source_type == "pdf":
        if explicit_encoding is not None:
            raise SourceDecodeError("--encoding is only valid for TXT sources")
        chapters, input_encoding, parser_version = extract_pdf_structure(raw)
    else:
        raise SourceStructureError(f"unsupported source type: {source_type!r}")

    detector_text = "\n\n".join(
        paragraph.text_original
        for chapter in chapters
        for paragraph in chapter.paragraphs
    )
    detected_language, detector_id = detect_language(detector_text)

    return SourceDocument(
        schema_version=SOURCE_DOCUMENT_SCHEMA_VERSION,
        project_id=project_id,
        document_id=document_id,
        source=SourceInfo(
            type=source_type,
            path=source_path,
            raw_sha256=raw_sha256(raw),
            byte_size=len(raw),
            declared_language=declared_language,
            detected_language=detected_language,
            language_detector=detector_id,
        ),
        normalization=NormalizationInfo(
            input_encoding=input_encoding,
            newline="LF",
            parser_id=SOURCE_PARSER_ID,
            parser_version=parser_version,
        ),
        chapters=chapters,
    )
