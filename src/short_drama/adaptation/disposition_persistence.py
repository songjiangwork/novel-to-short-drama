"""B1.3-B4B1 read-only CURRENT verification and exact reuse.

This is intentionally the *reader* half of B4B.  It neither persists a B1.3
publication nor mutates a pointer.  A published B4A payload is stored directly
as the existing immutable ``b13_disposition_publication`` artifact; there is no
second cache payload or persistence format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.foundation import FilePointerStore, PointerKind, PointerNotFoundError
from short_drama.story.consolidation import ConsolidationManifest
from short_drama.story.consolidation_persistence import (
    _require_a5_validation_report, _verify_consolidation_manifest,
    a5_base_artifact_id, a5_validation_artifact_id, build_a5_validation_report,
    load_consolidation_manifest,
)
from short_drama.story.story_analysis import StoryAnalysisManifest
from short_drama.story.story_analysis_persistence import (
    StoryAnalysisPersistenceService, a6_base_artifact_id,
)

from .disposition import B13DispositionPlan
from .disposition_preparation import (
    B12StrategySelection, B13DispositionPreparation,
    build_b13_disposition_preparation_from_authority,
    prepare_b13_disposition_windows,
)
from .disposition_publication import (
    B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
    B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION, B13DispositionPublication,
    b13_disposition_publication_artifact_id, verify_b13_disposition_publication,
)
from .disposition_semantic import B13WindowSemanticPreparation, build_b13_window_semantic_preparation
from .strategy_execution import (
    B12_STRATEGY_ARTIFACT_SCHEMA_VERSION, B12StrategyArtifact,
    _B12StrategyPersistenceService, b12_strategy_pointer_id,
)
from .target import AdaptationSourceSnapshot


class B13DispositionPersistenceError(ValueError):
    """A B1.3 CURRENT, persisted evidence, or authority check failed closed."""


def b13_disposition_pointer_id(project_id: str, adaptation_target_id: str) -> str:
    """Return B1.3's dedicated stage-local CURRENT pointer identity."""
    if not isinstance(project_id, str) or not project_id:
        raise B13DispositionPersistenceError("B1.3 pointer requires a project_id")
    if not isinstance(adaptation_target_id, str) or not adaptation_target_id:
        raise B13DispositionPersistenceError("B1.3 pointer requires an adaptation_target_id")
    return f"{project_id}.b13.{adaptation_target_id}.disposition"


@dataclass(frozen=True, slots=True)
class B13VerifiedCurrentSnapshot:
    """A fully verified B1.3 CURRENT that exactly matches a fresh request."""

    publication_ref: ArtifactRef
    current_pointer_ref: ArtifactRef
    publication: B13DispositionPublication
    plan: B13DispositionPlan
    reused: bool = True


@dataclass(frozen=True, slots=True)
class _HistoricalSourceAuthority:
    source: AdaptationSourceSnapshot
    manifest: StoryAnalysisManifest
    event_analysis: Any
    arc_analysis: Any
    event_set: Any


def _typed(store: FileArtifactStore, ref: ArtifactRef, artifact_type: str, model: Any) -> Any:
    if not isinstance(ref, ArtifactRef) or ref.artifact_type != artifact_type:
        raise B13DispositionPersistenceError(f"historical source has an invalid {artifact_type} ref")
    try:
        envelope = store.get_ref(ref)
        if not isinstance(envelope.payload, dict):
            raise ValueError("payload is not an object")
        value = model.from_dict(envelope.payload)
        if value.to_dict() != envelope.payload:
            raise ValueError("payload is not canonical")
        return value
    except Exception as exc:
        raise B13DispositionPersistenceError(
            f"historical source cannot resolve exact {artifact_type}"
        ) from exc


def _verify_historical_source(
    store: FileArtifactStore, pointers: FilePointerStore, source: AdaptationSourceSnapshot,
) -> _HistoricalSourceAuthority:
    """Verify pinned historical A5/A6 artifacts without consulting live heads."""
    if not isinstance(source, AdaptationSourceSnapshot):
        raise B13DispositionPersistenceError("publication source snapshot is invalid")
    try:
        a6 = _typed(store, source.story_analysis_manifest_ref, "story_analysis_manifest", StoryAnalysisManifest)
        a6_base = a6_base_artifact_id(
            a6.project_id, a6.document_id, a6.semantic_identity.story_analysis_profile_id,
        )
        verified_a6, _characters, event_analysis, arcs, _structure, _bible, _report = (
            StoryAnalysisPersistenceService(store, pointers)
            ._load_current_integrity(a6_base, source.story_analysis_manifest_ref)
        )
        # _load_current_integrity is structurally historical despite its legacy
        # name: it never reads a pointer.  Keep the supplied pointer store out
        # of this path so an old valid B1.3 revision remains verifiable after a
        # legitimate A5/A6 head advance.
        if verified_a6 != a6 or a6.consolidation_manifest_ref != source.consolidation_manifest_ref:
            raise ValueError("A6/A5 binding differs from the pinned snapshot")
        if a6.arc_analysis_ref != source.arc_analysis_ref:
            raise ValueError("A6 arc ref differs from the pinned snapshot")

        a5 = load_consolidation_manifest(
            store, source.consolidation_manifest_ref,
            expected_artifact_id=source.consolidation_manifest_ref.artifact_id,
        )
        a5_base = a5_base_artifact_id(
            a5.project_id, a5.document_id, a5.semantic_identity.consolidation_profile_id,
        )
        _verify_consolidation_manifest(
            store, base=a5_base, manifest=a5, manifest_ref=source.consolidation_manifest_ref,
        )
        _require_a5_validation_report(
            store, artifact_id=a5_validation_artifact_id(a5_base),
            revision=source.consolidation_manifest_ref.revision,
            expected_report=build_a5_validation_report(a5, source.consolidation_manifest_ref),
        )
        if (a5.project_id, a5.document_id) != (a6.project_id, a6.document_id):
            raise ValueError("A5/A6 project/document bindings differ")
        if (
            a5.entity_map_ref != source.entity_map_ref
            or a5.canonical_fact_set_ref != source.canonical_fact_set_ref
            or a5.canonical_event_set_ref != source.canonical_event_set_ref
            or a5.canonical_relationship_set_ref != source.canonical_relationship_set_ref
        ):
            raise ValueError("A5 leaves differ from the pinned snapshot")
        # Loading the complete exact snapshot through this historical route
        # verifies every persisted ref used by B1.3.  The canonical-reference
        # universe must be exactly reproducible, not merely a trusted copy in
        # B4A evidence.
        from . import target as target_module
        from short_drama.story.consolidation import CanonicalEventSet, CanonicalFactSet, CanonicalRelationshipSet
        from short_drama.story.reconciliation import CanonicalCharacterRegistry, CanonicalLocationRegistry, EntityMap

        entity_map = _typed(store, source.entity_map_ref, "entity_map", EntityMap)
        if (
            entity_map.canonical_character_registry_ref != source.canonical_character_registry_ref
            or entity_map.canonical_location_registry_ref != source.canonical_location_registry_ref
        ):
            raise ValueError("EntityMap registry bindings differ from the pinned snapshot")
        characters = _typed(store, source.canonical_character_registry_ref, "canonical_character_registry", CanonicalCharacterRegistry)
        locations = _typed(store, source.canonical_location_registry_ref, "canonical_location_registry", CanonicalLocationRegistry)
        facts = _typed(store, source.canonical_fact_set_ref, "canonical_fact_set", CanonicalFactSet)
        events = _typed(store, source.canonical_event_set_ref, "canonical_event_set", CanonicalEventSet)
        relationships = _typed(store, source.canonical_relationship_set_ref, "canonical_relationship_set", CanonicalRelationshipSet)
        refs = list(target_module._canonical_references_from_a4(entity_map, characters.entities, locations.entities))
        refs.extend(target_module.CanonicalTargetReference("fact", item.fact_id) for item in facts.facts)
        refs.extend(target_module.CanonicalTargetReference("event", item.event_id) for item in events.events)
        refs.extend(target_module.CanonicalTargetReference("relationship", item.relationship_id) for item in relationships.relationships)
        refs.extend(target_module.CanonicalTargetReference("arc", item.arc_id) for item in arcs.arcs)
        refs.extend(target_module.CanonicalTargetReference("turning_point", item.turning_point_id) for item in arcs.turning_points)
        refs.extend(target_module.CanonicalTargetReference("reveal", item.reveal_id) for item in arcs.reveals)
        refs.extend(target_module.CanonicalTargetReference("payoff", item.payoff_id) for item in arcs.foreshadow_payoffs)
        if tuple(refs) != source.canonical_references:
            raise ValueError("canonical reference universe differs from pinned snapshot")
        return _HistoricalSourceAuthority(source, a6, event_analysis, arcs, events)
    except B13DispositionPersistenceError:
        raise
    except Exception as exc:
        raise B13DispositionPersistenceError(
            "historical A5/A6 authority is missing, corrupt, or inconsistent"
        ) from exc


def _verify_historical_strategy(
    store: FileArtifactStore, pointers: FilePointerStore, publication: B13DispositionPublication,
) -> B12StrategySelection:
    """Resolve the B4A-selected B1.2 ref, never its potentially newer head."""
    try:
        evidence = publication.to_dict()["evidence"]
        source = AdaptationSourceSnapshot.from_dict(evidence["source_snapshot"])
        selection_data = evidence["strategy_selection"]
        strategy_ref = ArtifactRef.from_dict(selection_data["strategy_ref"])
        embedded = B12StrategyArtifact.from_dict(selection_data["strategy_artifact"])
        envelope = store.get_ref(strategy_ref)
        if (
            envelope.schema_version != B12_STRATEGY_ARTIFACT_SCHEMA_VERSION
            or not isinstance(envelope.payload, dict)
        ):
            raise ValueError("B1.2 envelope is unsupported")
        actual = B12StrategyArtifact.from_dict(envelope.payload)
        if actual != embedded or actual.source_snapshot != source:
            raise ValueError("B1.2 immutable artifact disagrees with B4A evidence")
        _B12StrategyPersistenceService(store, pointers)._validate_intrinsic_artifact(actual)
        observed = selection_data["current_pointer_ref"]
        if observed is not None:
            pointer_ref = ArtifactRef.from_dict(observed)
            pointer = pointers._load_pointer_ref(pointer_ref)
            target_id = actual.strategy.resolved_target.adaptation_target_id
            manifest = _typed(store, source.story_analysis_manifest_ref, "story_analysis_manifest", StoryAnalysisManifest)
            if (
                pointer_ref.artifact_id != b12_strategy_pointer_id(manifest.project_id, target_id)
                or pointer.pointer_kind is not PointerKind.CURRENT
                or pointer.target_ref != strategy_ref
            ):
                raise ValueError("observed B1.2 CURRENT snapshot is inconsistent")
        # Reconstruct the selection to retain the B1.3 contract's strict type
        # and explicit-vs-CURRENT distinction after the store round trip.
        return B12StrategySelection(
            strategy_ref, actual,
            None if observed is None else ArtifactRef.from_dict(observed),
        )
    except Exception as exc:
        raise B13DispositionPersistenceError(
            "historical B1.2 strategy authority is missing, corrupt, or inconsistent"
        ) from exc


class B13DispositionPersistenceService:
    """Read-only B4B1 resolver.  B4B2 owns the future write/CAS half."""

    def __init__(self, store: FileArtifactStore, pointers: FilePointerStore) -> None:
        if not isinstance(store, FileArtifactStore) or not isinstance(pointers, FilePointerStore):
            raise B13DispositionPersistenceError("store/pointers must be Foundation file stores")
        self._store = store
        self._pointers = pointers

    def try_reuse_current(
        self,
        preparation: B13DispositionPreparation,
        semantic_preparations: tuple[B13WindowSemanticPreparation, ...],
    ) -> B13VerifiedCurrentSnapshot | None:
        """Return an exact verified hit, ``None`` for a healthy miss only.

        Existing CURRENT integrity is deliberately checked before request
        comparison.  There is no historical revision scan and no provider or
        persistence side effect.
        """
        self._validate_request(preparation, semantic_preparations)
        project_id, target_id = self._request_pointer_identity(preparation)
        pointer_id = b13_disposition_pointer_id(project_id, target_id)
        try:
            pointer_ref = self._pointers.resolve_current_pointer_ref(pointer_id)
        except PointerNotFoundError:
            # A missing B1.3 head is a normal cache miss, but a caller cannot
            # use that miss to bypass verification of its requested live A5/A6
            # (and selected B1.2) authority.
            self._request_current_authority(preparation)
            try:
                self._pointers.resolve_current_pointer_ref(pointer_id)
            except PointerNotFoundError:
                return None
            raise B13DispositionPersistenceError(
                "B1.3 CURRENT appeared during missing-current verification"
            )
        try:
            pointer = self._pointers._load_pointer_ref(pointer_ref)
        except Exception as exc:
            raise B13DispositionPersistenceError("B1.3 CURRENT pointer fails integrity verification") from exc
        if (
            pointer.pointer_kind is not PointerKind.CURRENT
            or pointer.target_ref.artifact_type != B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE
        ):
            raise B13DispositionPersistenceError("B1.3 CURRENT pointer has invalid identity or target")
        publication = self._load_and_verify_publication(pointer.target_ref)
        historical, historical_selection = self._verify_historical_publication(publication)
        historical_target_id = historical_selection.artifact.strategy.resolved_target.adaptation_target_id
        if b13_disposition_pointer_id(historical.manifest.project_id, historical_target_id) != pointer_id:
            raise B13DispositionPersistenceError("B1.3 CURRENT publication belongs to another project or target")
        # Only after the persisted CURRENT has survived intrinsic verification
        # may a changed live request turn it into a normal cache miss.
        self._request_current_authority(preparation)
        if not self._matches_request(publication, preparation, semantic_preparations):
            self._recheck_stability(preparation, project_id, target_id, pointer_ref)
            return None
        plan = verify_b13_disposition_publication(publication)
        # This must be the final observable operation before a successful
        # return: B4A reconstruction above is deliberately expensive.
        self._recheck_stability(preparation, project_id, target_id, pointer_ref)
        return B13VerifiedCurrentSnapshot(pointer.target_ref, pointer_ref, publication, plan)

    def _request_pointer_identity(self, preparation: B13DispositionPreparation) -> tuple[str, str]:
        """Derive the stage-local pointer without asserting live freshness."""
        target_id = preparation.strategy_selection.artifact.strategy.resolved_target.adaptation_target_id
        try:
            manifest = _typed(
                self._store, preparation.source_snapshot.story_analysis_manifest_ref,
                "story_analysis_manifest", StoryAnalysisManifest,
            )
        except Exception as exc:
            raise B13DispositionPersistenceError("B1.3 request cannot derive stage pointer identity") from exc
        return manifest.project_id, target_id

    def _load_and_verify_publication(self, ref: ArtifactRef) -> B13DispositionPublication:
        try:
            envelope = self._store.get_ref(ref)
            if (
                envelope.schema_version != B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION
                or not isinstance(envelope.payload, dict)
            ):
                raise ValueError("unsupported immutable envelope")
            publication = B13DispositionPublication.from_dict(envelope.payload)
            verify_b13_disposition_publication(publication)
            if ref.artifact_id != publication.to_dict()["stage_artifact_id"]:
                raise ValueError("immutable artifact id disagrees with publication stage identity")
            self._verify_persisted_request_evidence(publication)
            return publication
        except Exception as exc:
            raise B13DispositionPersistenceError(
                "B1.3 CURRENT publication fails immutable evidence verification"
            ) from exc

    @staticmethod
    def _verify_persisted_request_evidence(publication: B13DispositionPublication) -> None:
        """Close the retained-audit fields B4A does not need for assembly.

        B4A reconstructs the request, provenance and semantic output.  This
        persistence boundary also rejects a separately forged retained prompt
        identity instead of allowing it to look like a new caller request.
        """
        for accepted in publication.to_dict()["evidence"]["accepted_windows"]:
            provenance = accepted["invocation_provenance"]
            expected_prompt = {
                "prompt_id": provenance["prompt_id"],
                "prompt_version": provenance["prompt_version"],
                "prompt_content_hash": provenance["prompt_content_hash"],
            }
            if accepted["prompt_identity"] != expected_prompt:
                raise B13DispositionPersistenceError(
                    "B1.3 publication accepted-window prompt identity is forged"
                )

    def _verify_historical_publication(
        self, publication: B13DispositionPublication,
    ) -> tuple[_HistoricalSourceAuthority, B12StrategySelection]:
        payload = publication.to_dict()
        source = AdaptationSourceSnapshot.from_dict(payload["evidence"]["source_snapshot"])
        authority = _verify_historical_source(self._store, self._pointers, source)
        selection = _verify_historical_strategy(self._store, self._pointers, publication)
        # B4A independently rebuilds the plan.  Keep this explicit at the
        # persistence boundary so a forged plan cannot be accepted by merely
        # matching a cache key.
        try:
            assembly = verify_b13_disposition_publication(publication)
            if not isinstance(assembly.plan, B13DispositionPlan):
                raise ValueError("assembly did not produce a B13DispositionPlan")
        except Exception as exc:
            raise B13DispositionPersistenceError("B1.3 publication has a forged final plan") from exc
        try:
            historical = build_b13_disposition_preparation_from_authority(
                source_snapshot=authority.source, strategy_selection=selection,
                event_set=authority.event_set, event_analysis=authority.event_analysis,
                arc_analysis=authority.arc_analysis,
            )
            accepted = payload["evidence"]["accepted_windows"]
            if (
                payload["evidence"]["preparation_identity"] != historical.preparation_identity
                or not isinstance(accepted, list)
                or len(accepted) != len(historical.packets)
                or any(item["packet"] != packet.to_dict() for item, packet in zip(accepted, historical.packets))
            ):
                raise ValueError("retained windows differ from immutable historical authority")
        except Exception as exc:
            raise B13DispositionPersistenceError(
                "B1.3 publication retained window packets are forged or incomplete"
            ) from exc
        return authority, selection

    def _validate_request(
        self, preparation: B13DispositionPreparation,
        semantics: tuple[B13WindowSemanticPreparation, ...],
    ) -> None:
        if not isinstance(preparation, B13DispositionPreparation):
            raise B13DispositionPersistenceError("reuse requires complete deterministic B1.3 preparation")
        if not isinstance(semantics, tuple) or len(semantics) != len(preparation.packets):
            raise B13DispositionPersistenceError("reuse requires one ordered semantic request per B1.3 window")
        for packet, semantic in zip(preparation.packets, semantics):
            if not isinstance(semantic, B13WindowSemanticPreparation) or semantic.packet != packet:
                raise B13DispositionPersistenceError("semantic request does not match ordered B1.3 preparation")
            try:
                rebuilt = build_b13_window_semantic_preparation(
                    packet, runtime_config=semantic.runtime_config,
                    execution_options=semantic.execution_options,
                )
            except Exception as exc:
                raise B13DispositionPersistenceError("semantic request cannot be independently rebuilt") from exc
            if (
                semantic.request_identity != rebuilt.request_identity
                or semantic.request != rebuilt.request
                or semantic.semantic_profile != rebuilt.semantic_profile
                or semantic.output_schema != rebuilt.output_schema
            ):
                raise B13DispositionPersistenceError("semantic request identities are not pinned deterministic assets")

    def _request_current_authority(self, preparation: B13DispositionPreparation) -> tuple[str, str]:
        """Verify live A5/A6 and selected B1.2 authority for the new request."""
        source = preparation.source_snapshot
        target_id = preparation.strategy_selection.artifact.strategy.resolved_target.adaptation_target_id
        try:
            manifest = _typed(self._store, source.story_analysis_manifest_ref, "story_analysis_manifest", StoryAnalysisManifest)
            # This is intentionally live-CURRENT verification; historical B4A
            # verification above must not call it.
            bound = AdaptationSourceSnapshot.from_store(
                self._store, self._pointers, source.story_analysis_manifest_ref,
            )
            if bound != source:
                raise ValueError("source snapshot is stale")
            explicit = (None if preparation.strategy_selection.current_pointer_ref is not None
                        else preparation.strategy_selection.strategy_ref)
            rebuilt = prepare_b13_disposition_windows(
                store=self._store, pointers=self._pointers, source_snapshot=source,
                adaptation_target_id=target_id, strategy_ref=explicit,
            )
            if rebuilt != preparation:
                raise ValueError("preparation is not the exact live authority")
        except Exception as exc:
            raise B13DispositionPersistenceError(
                "new B1.3 request no longer has exact live A5/A6/B1.2 authority"
            ) from exc
        return manifest.project_id, target_id

    def _matches_request(
        self,
        publication: B13DispositionPublication,
        preparation: B13DispositionPreparation,
        semantics: tuple[B13WindowSemanticPreparation, ...],
    ) -> bool:
        evidence = publication.to_dict()["evidence"]
        if (
            evidence["source_snapshot"] != preparation.source_snapshot.to_dict()
            or evidence["strategy_selection"]["strategy_ref"] != preparation.strategy_selection.strategy_ref.to_dict()
            or evidence["strategy_selection"]["current_pointer_ref"] != (
                None if preparation.strategy_selection.current_pointer_ref is None
                else preparation.strategy_selection.current_pointer_ref.to_dict()
            )
            or evidence["preparation_identity"] != preparation.preparation_identity
        ):
            return False
        windows = evidence["accepted_windows"]
        if not isinstance(windows, list) or len(windows) != len(preparation.packets):
            return False
        for expected_packet, semantic, accepted in zip(preparation.packets, semantics, windows):
            if not isinstance(accepted, dict):
                return False
            packet = accepted.get("packet")
            if (
                packet != expected_packet.to_dict()
                or accepted.get("request_identity") != semantic.request_identity
                or accepted.get("semantic_request_hash") != semantic.request.request_hash
                or accepted.get("semantic_profile_identity") != {
                    "profile_id": semantic.semantic_profile.profile_id,
                    "profile_hash": semantic.semantic_profile.semantic_profile_hash,
                }
                or accepted.get("prompt_identity") != {
                    "prompt_id": semantic.request.rendered_prompt.prompt_id,
                    "prompt_version": semantic.request.rendered_prompt.prompt_version,
                    "prompt_content_hash": semantic.request.rendered_prompt.prompt_content_hash,
                }
                or accepted.get("output_schema_identity") != {
                    "schema_id": semantic.output_schema.schema_id,
                    "schema_version": semantic.output_schema.schema_version,
                    "schema_hash": semantic.output_schema.schema_hash,
                }
            ):
                return False
        return True

    def _recheck_stability(
        self, preparation: B13DispositionPreparation, project_id: str, target_id: str,
                           pointer_ref: ArtifactRef) -> None:
        self._request_current_authority(preparation)
        pointer_id = b13_disposition_pointer_id(project_id, target_id)
        try:
            if self._pointers.resolve_current_pointer_ref(pointer_id) != pointer_ref:
                raise B13DispositionPersistenceError("B1.3 CURRENT changed during verification")
        except PointerNotFoundError as exc:
            raise B13DispositionPersistenceError("B1.3 CURRENT changed during verification") from exc


__all__ = [
    "B13DispositionPersistenceError", "B13VerifiedCurrentSnapshot",
    "B13DispositionPersistenceService", "b13_disposition_pointer_id",
]
