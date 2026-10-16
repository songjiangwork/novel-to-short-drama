"""Thin A1--A6 production wrapper with no independent stage authority."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from short_drama.llm import LLMClient

from .consolidation_service import consolidate_evidence_project
from .extraction_batch import extract_chunks_project
from .reconciliation_service import reconcile_entities_project
from .service import ingest_source_project, plan_chunks_project
from .story_analysis_service import build_story_bible_project


def _report(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    raise TypeError("stage result must be a JSON-compatible report")


@dataclass(frozen=True, slots=True)
class PhaseAAnalysisResult:
    """Ordered reporting envelope; each nested result remains stage-owned."""

    ingest_source: dict[str, Any]
    plan_chunks: dict[str, Any]
    extract_chunks: dict[str, Any]
    reconcile_entities: dict[str, Any]
    consolidate_evidence: dict[str, Any]
    build_story_bible: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingest_source": self.ingest_source,
            "plan_chunks": self.plan_chunks,
            "extract_chunks": self.extract_chunks,
            "reconcile_entities": self.reconcile_entities,
            "consolidate_evidence": self.consolidate_evidence,
            "build_story_bible": self.build_story_bible,
        }


def analyze_story_project(
    project_path: str | Path,
    *,
    runs_root: str | Path,
    chunk_profile_path: str | Path,
    extraction_profile_path: str | Path,
    extraction_semantic_profile_path: str | Path,
    reconciliation_profile_path: str | Path,
    reconciliation_semantic_profile_path: str | Path,
    reconciliation_profile_id: str,
    consolidation_profile_path: str | Path,
    consolidation_profile_id: str,
    consolidation_semantic_profile_path: str | Path,
    story_analysis_profile_path: str | Path,
    story_analysis_semantic_profile_path: str | Path,
    llm_client: LLMClient,
) -> PhaseAAnalysisResult:
    """Execute existing A1--A6 services in dependency order.

    No error is caught here: stage-specific failure behavior reaches the CLI
    unchanged and later stages are never started after a failure.
    """
    a1 = ingest_source_project(project_path, runs_root=runs_root)
    a2 = plan_chunks_project(
        project_path, runs_root=runs_root, profile_path=chunk_profile_path
    )
    a3 = extract_chunks_project(
        project_path,
        runs_root=runs_root,
        chunk_profile_path=chunk_profile_path,
        extraction_profile_path=extraction_profile_path,
        semantic_profile_path=extraction_semantic_profile_path,
        llm_client=llm_client,
    )
    a4 = reconcile_entities_project(
        project_path,
        runs_root=runs_root,
        chunk_profile_path=chunk_profile_path,
        extraction_profile_path=extraction_profile_path,
        reconciliation_profile_path=reconciliation_profile_path,
        semantic_profile_path=reconciliation_semantic_profile_path,
        llm_client=llm_client,
    )
    a5 = consolidate_evidence_project(
        project_path,
        runs_root=runs_root,
        reconciliation_profile_id=reconciliation_profile_id,
        consolidation_profile_path=consolidation_profile_path,
        semantic_profile_path=consolidation_semantic_profile_path,
        llm_client=llm_client,
    )
    a6 = build_story_bible_project(
        project_path,
        runs_root=runs_root,
        consolidation_profile_id=consolidation_profile_id,
        story_analysis_profile_path=story_analysis_profile_path,
        semantic_profile_path=story_analysis_semantic_profile_path,
        llm_client=llm_client,
    )
    return PhaseAAnalysisResult(
        _report(a1), _report(a2), _report(a3), _report(a4), _report(a5), _report(a6)
    )


__all__ = ["PhaseAAnalysisResult", "analyze_story_project"]
