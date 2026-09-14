from __future__ import annotations

import pytest

from short_drama.artifacts import FileArtifactStore
from short_drama.story import StoryIntegrityError, build_source_document
from short_drama.story.chunking import (
    CHUNK_PLANNER_VERSION,
    ChunkCoverage,
    ChunkManifest,
    ChunkPlanningProfile,
    ParagraphSpan,
    SourceChunk,
    estimate_paragraphs,
)
from short_drama.story.persistence import (
    load_chunk_manifest,
    persist_chunk_manifest,
    persist_source_chunk,
    persist_source_document,
)


def test_manifest_rejects_valid_but_noncanonical_chunk_partition(
    monkeypatch,
    tmp_path,
):
    import short_drama.story.source as source_module

    monkeypatch.setattr(
        source_module,
        "detect_language",
        lambda _text: ("en", "langid-1.1.6"),
    )

    source = build_source_document(
        project_id="canonical-test",
        document_id="src_001",
        source_type="txt",
        source_path="source/novel.txt",
        raw=b"Chapter One\naaa\n\nbbb\n\nccc\n",
        declared_language="en-CA",
    )
    store = FileArtifactStore(tmp_path / "artifacts")
    source_ref = persist_source_document(store, source, revision=1)

    profile = ChunkPlanningProfile(
        schema_version=1,
        profile_id="canonical-test-v1",
        token_counter="utf8-bytes-div3-v1",
        ownership_token_budget=3,
        context_overlap_token_budget=0,
        context_token_budget=3,
    )

    # The canonical greedy planner groups P0001 + P0002 together because
    # their combined estimate is exactly 3. This alternate plan deliberately
    # owns one paragraph per chunk. It still has complete exact-one coverage
    # and respects every budget, so coverage-only validation would accept it.
    chapter = source.chapters[0]
    alternate_chunks = []
    for index, paragraph in enumerate(chapter.paragraphs, start=1):
        count = estimate_paragraphs((paragraph,))
        alternate_chunks.append(
            SourceChunk(
                schema_version=1,
                chunk_id=f"CH001_C{index:03d}",
                project_id=source.project_id,
                document_id=source.document_id,
                chapter_id=chapter.chapter_id,
                source_document_ref=source_ref,
                context_span=ParagraphSpan(
                    paragraph.paragraph_id,
                    paragraph.paragraph_id,
                ),
                ownership_span=ParagraphSpan(
                    paragraph.paragraph_id,
                    paragraph.paragraph_id,
                ),
                paragraph_ids=(paragraph.paragraph_id,),
                token_count_method="utf8-bytes-div3-v1",
                context_token_count=count,
                ownership_token_count=count,
            )
        )

    chunk_refs = tuple(
        persist_source_chunk(
            store,
            chunk,
            profile_id=profile.profile_id,
            revision=99,
        )
        for chunk in alternate_chunks
    )
    manifest = ChunkManifest(
        schema_version=1,
        project_id=source.project_id,
        document_id=source.document_id,
        source_document_ref=source_ref,
        planner_version=CHUNK_PLANNER_VERSION,
        profile=profile,
        chunk_refs=chunk_refs,
        chunk_count=len(chunk_refs),
        coverage=ChunkCoverage(
            paragraphs_total=3,
            owned_once=3,
            unowned=0,
            multiply_owned=0,
        ),
        state="CHUNKING_COMPLETE",
    )
    manifest_ref = persist_chunk_manifest(store, manifest, revision=99)

    with pytest.raises(
        StoryIntegrityError,
        match="deterministic planner output",
    ):
        load_chunk_manifest(store, manifest_ref)
