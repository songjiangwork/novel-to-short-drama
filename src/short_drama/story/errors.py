from __future__ import annotations


class StoryError(Exception):
    """Base error for deterministic v1.2 Story Analysis stages."""


class StoryConfigError(StoryError):
    """Project/profile configuration is invalid."""


class SourceIngestionError(StoryError):
    """A1 source ingestion failed."""


class SourceReadError(SourceIngestionError):
    """Source bytes cannot be read."""


class SourceDecodeError(SourceIngestionError):
    """Source text cannot be decoded under the selected policy."""


class SourceStructureError(SourceIngestionError):
    """Parsed source structure violates the A1 contract."""


class SourcePdfError(SourceIngestionError):
    """Text-based PDF extraction failed or yielded no usable text."""


class SourceLanguageError(SourceIngestionError):
    """Source-language detection failed."""


class StoryPersistenceError(StoryError):
    """Typed Story artifact persistence or resolution failed."""


class StoryIntegrityError(StoryPersistenceError):
    """Persisted Story artifact violates its typed contract."""


class ChunkPlanningError(StoryError):
    """A2 chunk planning failed."""


class ChunkProfileError(ChunkPlanningError):
    """Chunk-planning profile is invalid."""


class ChunkCoverageError(ChunkPlanningError):
    """Chunk ownership/context coverage violates A2 invariants."""
