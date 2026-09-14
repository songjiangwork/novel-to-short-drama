from __future__ import annotations

import codecs
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from short_drama.artifacts import ArtifactRef, FileArtifactStore, ImmutableArtifactEnvelope
from short_drama.foundation import FilePointerStore, PointerKind
from short_drama.story import (
    ChunkPlanningError,
    ChunkPlanningProfile,
    SourceDecodeError,
    SourcePdfError,
    StoryIntegrityError,
    build_source_document,
    decode_txt_bytes,
    estimate_tokens,
    ingest_source_project,
    plan_chunks,
    plan_chunks_project,
)
from short_drama.story.chunking import ChunkManifest
from short_drama.story.persistence import (
    load_source_document,
    source_document_artifact_id,
    source_pointer_id,
)
from short_drama.story.source import (
    NormalizationInfo,
    SourceChapter,
    SourceDocument,
    SourceInfo,
    SourceParagraph,
    extract_pdf_structure,
)

HASH_A = "a" * 64


def _patch_language(monkeypatch, language: str = "en") -> None:
    import short_drama.story.source as source_module
    monkeypatch.setattr(source_module, "detect_language", lambda _text: (language, "langid-1.1.6"))


def _source_ref(revision: int = 1) -> ArtifactRef:
    return ArtifactRef("source_document", "demo.src_001", revision, HASH_A)


def _document(paragraphs_by_chapter: tuple[tuple[str, ...], ...]) -> SourceDocument:
    chapters = []
    for chapter_number, texts in enumerate(paragraphs_by_chapter, start=1):
        chapter_id = f"CH{chapter_number:03d}"
        paragraphs = tuple(
            SourceParagraph(
                paragraph_id=f"{chapter_id}_P{paragraph_number:04d}",
                text_original=text,
                source_pages=None,
            )
            for paragraph_number, text in enumerate(texts, start=1)
        )
        chapters.append(SourceChapter(chapter_id, None, "synthetic", paragraphs))
    return SourceDocument(
        schema_version=1,
        project_id="demo",
        document_id="src_001",
        source=SourceInfo(
            "txt", "source/novel.txt", HASH_A, 123, "en-CA", "en", "langid-1.1.6"
        ),
        normalization=NormalizationInfo(
            "utf-8", "LF", "short_drama_source_ingestion_v1", "1"
        ),
        chapters=tuple(chapters),
    )


def _profile(**overrides) -> ChunkPlanningProfile:
    values = {
        "schema_version": 1,
        "profile_id": "test-profile",
        "token_counter": "utf8-bytes-div3-v1",
        "ownership_token_budget": 8,
        "context_overlap_token_budget": 4,
        "context_token_budget": 16,
    }
    values.update(overrides)
    return ChunkPlanningProfile(**values)


def _write_project(project_dir: Path, text: str) -> Path:
    source_dir = project_dir / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "novel.txt").write_text(text, encoding="utf-8")
    project = {
        "schema_version": 1,
        "project_id": "story-test",
        "title": "Story Test",
        "source": {"type": "txt", "path": "source/novel.txt", "language": "en-CA"},
        "production": {"output_language": "en-CA", "profile": "h3_v1"},
        "approval_policy": {
            "adaptation_requires_approval": True,
            "generation_preflight_requires_approval": True,
            "shot_qc_requires_approval": True,
        },
    }
    path = project_dir / "project.yaml"
    path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    return path


def _write_profile(path: Path, ownership: int = 12) -> Path:
    profile = {
        "schema_version": 1,
        "profile_id": "story-analysis-test",
        "token_counter": "utf8-bytes-div3-v1",
        "ownership_token_budget": ownership,
        "context_overlap_token_budget": 3,
        "context_token_budget": ownership + 6,
    }
    path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return path


def test_txt_decoding_bom_newlines_and_strict_failure():
    text, encoding = decode_txt_bytes(codecs.BOM_UTF8 + b"one\r\ntwo\rthree")
    assert encoding == "utf-8-sig"
    assert text == "one\ntwo\nthree"

    utf16 = codecs.BOM_UTF16_LE + "alpha\r\nbeta".encode("utf-16-le")
    text, encoding = decode_txt_bytes(utf16)
    assert encoding == "utf-16"
    assert text == "alpha\nbeta"

    with pytest.raises(SourceDecodeError):
        decode_txt_bytes(b"\xff\xfe\x00")

    text, encoding = decode_txt_bytes("café".encode("latin-1"), "latin-1")
    assert text == "café"
    assert encoding == "iso8859-1"


def test_source_structure_headings_preface_and_synthetic(monkeypatch):
    _patch_language(monkeypatch)
    raw = (
        "Preface material.\n\nChapter One\nFirst paragraph.\n\nSecond paragraph.\n\n"
        "Part II\nThird paragraph.\n\nEpilogue\nLast paragraph.\n"
    ).encode()
    document = build_source_document(
        project_id="demo",
        document_id="src_001",
        source_type="txt",
        source_path="source/novel.txt",
        raw=raw,
        declared_language="en-CA",
    )
    assert [chapter.chapter_id for chapter in document.chapters] == [
        "CH001", "CH002", "CH003", "CH004"
    ]
    assert [chapter.heading_kind for chapter in document.chapters] == [
        "synthetic", "chapter", "part", "epilogue"
    ]
    assert document.chapters[1].title_original == "Chapter One"
    assert [p.paragraph_id for p in document.chapters[1].paragraphs] == [
        "CH002_P0001", "CH002_P0002"
    ]
    assert all("Chapter One" not in p.text_original for p in document.paragraphs)

    no_heading = build_source_document(
        project_id="demo",
        document_id="src_001",
        source_type="txt",
        source_path="source/novel.txt",
        raw=b"line one\nline two\nline three",
        declared_language="en",
    )
    assert no_heading.chapters[0].heading_kind == "synthetic"
    assert [p.text_original for p in no_heading.paragraphs] == [
        "line one", "line two", "line three"
    ]


def test_chinese_heading_detection(monkeypatch):
    _patch_language(monkeypatch, "zh")
    document = build_source_document(
        project_id="demo",
        document_id="src_001",
        source_type="txt",
        source_path="source/novel.txt",
        raw="序章\n开场。\n\n第一章\n正文。\n\n第十二章\n结尾。".encode("utf-8"),
        declared_language="zh-CN",
    )
    assert [chapter.heading_kind for chapter in document.chapters] == [
        "prologue", "chapter", "chapter"
    ]
    assert [chapter.title_original for chapter in document.chapters] == [
        "序章", "第一章", "第十二章"
    ]


def test_pdf_text_extraction_tracks_page_provenance(monkeypatch):
    class FakePage:
        def __init__(self, text): self.text = text
        def extract_text(self): return self.text

    class FakeReader:
        def __init__(self, _stream):
            self.pages = [FakePage("Chapter 1\nPage one."), FakePage("Page two.")]

    import short_drama.story.source as source_module
    monkeypatch.setattr(source_module, "_load_pdf_reader", lambda: FakeReader)
    chapters, encoding, parser_version = extract_pdf_structure(b"%PDF-fake")
    assert encoding == "pdf-text-extraction"
    assert "pypdf=" in parser_version
    assert [p.source_pages for p in chapters[0].paragraphs] == [(1,), (2,)]

    class EmptyReader:
        def __init__(self, _stream): self.pages = [FakePage(""), FakePage(None)]

    monkeypatch.setattr(source_module, "_load_pdf_reader", lambda: EmptyReader)
    with pytest.raises(SourcePdfError, match="OCR"):
        extract_pdf_structure(b"%PDF-empty")


def test_token_estimate_profile_and_chunk_coverage():
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("你好") == 2
    with pytest.raises(Exception):
        _profile(context_token_budget=15)

    source = _document((("aaa", "bbb", "ccc", "ddd"), ("eee", "fff")))
    profile = _profile(
        ownership_token_budget=4,
        context_overlap_token_budget=2,
        context_token_budget=8,
    )
    chunks, coverage = plan_chunks(source, _source_ref(), profile)
    assert coverage.complete
    assert coverage.paragraphs_total == 6
    assert [chunk.chunk_id for chunk in chunks] == [
        "CH001_C001", "CH001_C002", "CH002_C001"
    ]
    assert all(chunk.context_token_count <= profile.context_token_budget for chunk in chunks)
    assert all(chunk.ownership_token_count <= profile.ownership_token_budget for chunk in chunks)
    assert all(
        all(pid.startswith(chunk.chapter_id + "_P") for pid in chunk.paragraph_ids)
        for chunk in chunks
    )


def test_oversized_paragraph_fails_closed():
    source = _document((("x" * 100,),))
    with pytest.raises(ChunkPlanningError, match="exceeds ownership_token_budget"):
        plan_chunks(
            source,
            _source_ref(),
            _profile(
                ownership_token_budget=5,
                context_overlap_token_budget=0,
                context_token_budget=5,
            ),
        )


def test_typed_source_loader_rejects_future_envelope_schema(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    document = _document((("one",),))
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type="source_document",
        artifact_id=source_document_artifact_id("demo", "src_001"),
        revision=1,
        schema_version=2,
        payload=document.to_dict(),
    )
    ref = store.put(envelope)
    with pytest.raises(StoryIntegrityError, match="unsupported SourceDocument schema_version"):
        load_source_document(store, ref)


def test_service_reuse_source_revision_change_and_replan(monkeypatch, tmp_path):
    _patch_language(monkeypatch)
    project_dir = tmp_path / "project"
    project_path = _write_project(
        project_dir,
        "Chapter 1\nAlpha paragraph.\n\nBeta paragraph.\n",
    )
    profile_path = _write_profile(tmp_path / "profile.yaml")
    runs_root = tmp_path / "runs"

    first_source = ingest_source_project(project_path, runs_root=runs_root)
    assert first_source["reused"] is False
    assert first_source["source_document_ref"]["revision"] == 1
    reused_source = ingest_source_project(project_path, runs_root=runs_root)
    assert reused_source["reused"] is True
    assert reused_source["source_document_ref"] == first_source["source_document_ref"]

    first_plan = plan_chunks_project(project_path, runs_root=runs_root, profile_path=profile_path)
    assert first_plan["reused"] is False
    assert first_plan["coverage"] == {
        "paragraphs_total": 2,
        "owned_once": 2,
        "unowned": 0,
        "multiply_owned": 0,
    }
    reused_plan = plan_chunks_project(project_path, runs_root=runs_root, profile_path=profile_path)
    assert reused_plan["reused"] is True
    assert reused_plan["chunk_manifest_ref"] == first_plan["chunk_manifest_ref"]

    source_file = project_dir / "source" / "novel.txt"
    source_file.write_text(
        "Chapter 1\nAlpha changed.\n\nBeta paragraph.\n\nGamma paragraph.\n",
        encoding="utf-8",
    )
    second_source = ingest_source_project(project_path, runs_root=runs_root)
    assert second_source["reused"] is False
    assert second_source["source_document_ref"]["revision"] == 2

    second_plan = plan_chunks_project(project_path, runs_root=runs_root, profile_path=profile_path)
    assert second_plan["reused"] is False
    assert second_plan["source_document_ref"] == second_source["source_document_ref"]
    assert second_plan["chunk_manifest_ref"]["revision"] > first_plan["chunk_manifest_ref"]["revision"]

    artifact_store = FileArtifactStore(runs_root / "story-test" / "story" / "artifacts")
    assert artifact_store.get_ref(ArtifactRef.from_dict(first_source["source_document_ref"]))
    assert artifact_store.get_ref(ArtifactRef.from_dict(first_plan["chunk_manifest_ref"]))


def test_orphan_artifact_does_not_become_current(monkeypatch, tmp_path):
    _patch_language(monkeypatch)
    project_path = _write_project(tmp_path / "project", "Chapter 1\nAlpha paragraph.\n")
    runs_root = tmp_path / "runs"
    result = ingest_source_project(project_path, runs_root=runs_root)
    current_ref = ArtifactRef.from_dict(result["source_document_ref"])

    store = FileArtifactStore(runs_root / "story-test" / "story" / "artifacts")
    pointers = FilePointerStore(runs_root / "story-test" / "story" / "pointers", store)
    current_doc = load_source_document(store, current_ref)
    orphan = ImmutableArtifactEnvelope.create(
        artifact_type="source_document",
        artifact_id=source_document_artifact_id("story-test", "src_001"),
        revision=99,
        schema_version=1,
        payload=current_doc.to_dict(),
    )
    orphan_ref = store.put(orphan)
    assert orphan_ref.revision == 99
    assert pointers.resolve_current(source_pointer_id("story-test", "src_001")).target_ref == current_ref


def test_plan_requires_exact_a1_validation(monkeypatch, tmp_path):
    _patch_language(monkeypatch)
    project_path = _write_project(tmp_path / "project", "Chapter 1\nAlpha paragraph.\n")
    profile_path = _write_profile(tmp_path / "profile.yaml")
    runs_root = tmp_path / "runs"
    ingest = ingest_source_project(project_path, runs_root=runs_root)

    store = FileArtifactStore(runs_root / "story-test" / "story" / "artifacts")
    current = load_source_document(store, ArtifactRef.from_dict(ingest["source_document_ref"]))
    payload = current.to_dict()
    payload["source"]["raw_sha256"] = "b" * 64
    fake = ImmutableArtifactEnvelope.create(
        artifact_type="source_document",
        artifact_id=source_document_artifact_id("story-test", "src_001"),
        revision=50,
        schema_version=1,
        payload=payload,
    )
    fake_ref = store.put(fake)
    pointers = FilePointerStore(runs_root / "story-test" / "story" / "pointers", store)
    pointer_id = source_pointer_id("story-test", "src_001")
    previous_pointer_ref = pointers.resolve_current_pointer_ref(pointer_id)
    pointers.compare_and_set(
        pointer_id=pointer_id,
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=previous_pointer_ref,
        target_ref=fake_ref,
    )
    with pytest.raises(StoryIntegrityError, match="ValidationReport"):
        plan_chunks_project(project_path, runs_root=runs_root, profile_path=profile_path)


def test_story_schemas_accept_runtime_payloads(monkeypatch):
    _patch_language(monkeypatch)
    repo_root = Path(__file__).resolve().parents[1]
    source_schema = json.loads((repo_root / "schemas" / "source-document.schema.json").read_text())
    chunk_schema = json.loads((repo_root / "schemas" / "source-chunk.schema.json").read_text())
    manifest_schema = json.loads((repo_root / "schemas" / "chunk-manifest.schema.json").read_text())
    profile_schema = json.loads((repo_root / "schemas" / "chunk-profile.schema.json").read_text())

    document = build_source_document(
        project_id="demo",
        document_id="src_001",
        source_type="txt",
        source_path="source/novel.txt",
        raw=b"Chapter 1\nAlpha paragraph.\n\nBeta paragraph.",
        declared_language="en-CA",
    )
    source_ref = _source_ref()
    profile = _profile()
    chunks, coverage = plan_chunks(document, source_ref, profile)
    fake_chunk_refs = tuple(
        ArtifactRef(
            artifact_type="source_chunk",
            artifact_id=f"demo.src_001.test-profile.{chunk.chunk_id.lower()}",
            revision=1,
            content_hash=("b" if index == 0 else "c") * 64,
        )
        for index, chunk in enumerate(chunks)
    )
    manifest = ChunkManifest(
        schema_version=1,
        project_id="demo",
        document_id="src_001",
        source_document_ref=source_ref,
        planner_version="a2_chunk_planner_v1",
        profile=profile,
        chunk_refs=fake_chunk_refs,
        chunk_count=len(fake_chunk_refs),
        coverage=coverage,
        state="CHUNKING_COMPLETE",
    )

    assert not list(Draft202012Validator(source_schema).iter_errors(document.to_dict()))
    assert not list(Draft202012Validator(profile_schema).iter_errors(profile.to_dict()))
    for chunk in chunks:
        assert not list(Draft202012Validator(chunk_schema).iter_errors(chunk.to_dict()))
    assert not list(Draft202012Validator(manifest_schema).iter_errors(manifest.to_dict()))
