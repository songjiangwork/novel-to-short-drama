"""B1.3-B4B CURRENT verification, immutable publication, and exact reuse.

B4B1's verifier remains the sole reader for a published B4A payload.  B4B2
adds the narrowly scoped immutable write and stage-local CURRENT CAS path; it
stores B4A's existing ``b13_disposition_publication`` payload directly and
does not introduce a second persistence format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from short_drama.artifacts import (
    ArtifactError, ArtifactNotFoundError, ArtifactRef, FileArtifactStore,
    ImmutableArtifactEnvelope,
)
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
class B13DispositionPublicationResult:
    """One verified B1.3 CURRENT publication, never a human approval."""

    publication_ref: ArtifactRef
    current_pointer_ref: ArtifactRef
    publication: B13DispositionPublication
    plan: B13DispositionPlan
    reused: bool = False


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
    """B1.3 immutable publication and stage-local CURRENT persistence."""

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
        current = self._resolve_verified_current(pointer_id)
        if current is None:
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
        # Only after the persisted CURRENT has survived intrinsic verification
        # may a changed live request turn it into a normal cache miss.
        self._request_current_authority(preparation)
        if not self._matches_request(current.publication, preparation, semantic_preparations):
            self._recheck_stability(preparation, project_id, target_id, current.current_pointer_ref)
            return None
        # This must be the final observable operation before a successful
        # return: B4A reconstruction above is deliberately expensive.
        self._recheck_stability(preparation, project_id, target_id, current.current_pointer_ref)
        return current

    def publish_validated(
        self,
        *,
        publication: B13DispositionPublication,
        preparation: B13DispositionPreparation,
        semantic_preparations: tuple[B13WindowSemanticPreparation, ...],
    ) -> B13DispositionPublicationResult:
        """Persist exact B4A evidence and advance B1.3 CURRENT exactly once.

        ``publication`` is not trusted merely because it has B4A's type: this
        boundary reconstructs it, binds it to the supplied deterministic B1.3
        request, verifies live A5/A6/B1.2 authority, then uses a stage-local
        CURRENT CAS.  A matching final semantic identity reuses only a healthy
        CURRENT; request-level identity alone is deliberately insufficient.
        """
        self._validate_request(preparation, semantic_preparations)
        if not isinstance(publication, B13DispositionPublication):
            raise B13DispositionPersistenceError("publication requires exact B4A publication evidence")
        try:
            plan = verify_b13_disposition_publication(publication)
        except Exception as exc:
            raise B13DispositionPersistenceError("publication B4A evidence cannot reconstruct a final plan") from exc
        if not self._matches_request(publication, preparation, semantic_preparations):
            raise B13DispositionPersistenceError(
                "publication evidence is not for the exact prepared B1.3 request"
            )
        project_id, target_id = self._request_pointer_identity(preparation)
        pointer_id = b13_disposition_pointer_id(project_id, target_id)

        # Always validate an existing head before comparing semantic identity.
        # There is intentionally no history fallback when it is corrupt.
        current = self._resolve_verified_current(pointer_id)
        observed_pointer_ref = None if current is None else current.current_pointer_ref
        self._request_current_authority(preparation)
        self._require_observed_current(pointer_id, observed_pointer_ref)
        if current is not None and current.publication.semantic_identity == publication.semantic_identity:
            # This final-semantic hit remains distinct from B4B1's pre-provider
            # request reuse: a different proposal/reconciliation decision must
            # proceed to a new immutable revision even when request identities
            # are identical.
            return B13DispositionPublicationResult(
                current.publication_ref, current.current_pointer_ref,
                current.publication, current.plan, reused=True,
            )

        artifact_id = b13_disposition_publication_artifact_id(
            preparation.source_snapshot, preparation.strategy_selection.artifact,
        )
        if publication.to_dict()["stage_artifact_id"] != artifact_id:
            raise B13DispositionPersistenceError("publication artifact identity disagrees with exact authority")
        revision = self._allocate_revision(
            artifact_id=artifact_id,
            observed_current=current,
        )
        envelope = ImmutableArtifactEnvelope.create(
            artifact_type=B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
            artifact_id=artifact_id,
            revision=revision,
            schema_version=B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
            payload=publication.to_dict(),
        )
        try:
            publication_ref = self._store.put(envelope)
        except ArtifactError as exc:
            raise B13DispositionPersistenceError("failed to persist immutable B1.3 publication") from exc

        # An orphan is acceptable if this post-write guard discovers a race;
        # returning a stale success is not.  Do not retry against a new head.
        self._recheck_stability(preparation, project_id, target_id, observed_pointer_ref)
        try:
            pointer_ref = self._pointers.compare_and_set(
                pointer_id=pointer_id,
                pointer_kind=PointerKind.CURRENT,
                expected_pointer_ref=observed_pointer_ref,
                target_ref=publication_ref,
            )
        except Exception as exc:
            raise B13DispositionPersistenceError("B1.3 CURRENT publication CAS failed") from exc
        self._verify_published_success(
            preparation=preparation,
            project_id=project_id,
            target_id=target_id,
            publication_ref=publication_ref,
            pointer_ref=pointer_ref,
        )
        return B13DispositionPublicationResult(
            publication_ref, pointer_ref, publication, plan,
        )

    def _resolve_verified_current(self, pointer_id: str) -> B13VerifiedCurrentSnapshot | None:
        """Use B4B1's single historical verifier for the stage-local head."""
        try:
            pointer_ref = self._pointers.resolve_current_pointer_ref(pointer_id)
        except PointerNotFoundError:
            return None
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
        try:
            plan = verify_b13_disposition_publication(publication)
        except Exception as exc:
            raise B13DispositionPersistenceError("B1.3 CURRENT publication has a forged final plan") from exc
        return B13VerifiedCurrentSnapshot(pointer.target_ref, pointer_ref, publication, plan)

    def _allocate_revision(
        self,
        *,
        artifact_id: str,
        observed_current: B13VerifiedCurrentSnapshot | None,
    ) -> int:
        """Allocate after the stage-local head, preserving monotonic revision.

        The immutable artifact id can change with A6, while the B1.3 CURRENT
        pointer cannot.  Start after the observed published revision and only
        skip occupied *valid* immutable slots; a corrupt slot is an error.
        """
        revision = 1 if observed_current is None else observed_current.publication_ref.revision + 1
        while True:
            try:
                envelope = self._store.get(
                    B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE, artifact_id, revision,
                )
            except ArtifactNotFoundError:
                return revision
            except ArtifactError as exc:
                raise B13DispositionPersistenceError(
                    "cannot allocate immutable B1.3 publication revision"
                ) from exc
            self._verify_occupied_publication(
                envelope=envelope, artifact_id=artifact_id, revision=revision,
            )
            revision += 1

    def _verify_occupied_publication(
        self,
        *,
        envelope: ImmutableArtifactEnvelope,
        artifact_id: str,
        revision: int,
    ) -> None:
        """Validate an occupied allocation slot without considering it for reuse.

        Foundation has already verified the envelope's canonical bytes.  An
        immutable B1.3 orphan additionally has to pass B4B1's publication and
        exact historical-authority verification before it is safe to skip.
        Historical refs deliberately need not still be CURRENT.
        """
        try:
            if (
                envelope.artifact_type != B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE
                or envelope.artifact_id != artifact_id
                or envelope.revision != revision
                or envelope.schema_version != B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION
            ):
                raise ValueError("occupied envelope has an unexpected B1.3 publication identity")
            publication = self._load_and_verify_publication(envelope.ref)
            self._verify_historical_publication(publication)
        except Exception as exc:
            raise B13DispositionPersistenceError(
                "occupied immutable B1.3 publication revision fails semantic integrity"
            ) from exc

    def _require_observed_current(self, pointer_id: str, expected: ArtifactRef | None) -> None:
        try:
            actual = self._pointers.resolve_current_pointer_ref(pointer_id)
        except PointerNotFoundError:
            actual = None
        if actual != expected:
            raise B13DispositionPersistenceError("B1.3 CURRENT changed during publication verification")

    def _verify_published_success(
        self,
        *,
        preparation: B13DispositionPreparation,
        project_id: str,
        target_id: str,
        publication_ref: ArtifactRef,
        pointer_ref: ArtifactRef,
    ) -> None:
        pointer_id = b13_disposition_pointer_id(project_id, target_id)
        try:
            pointer = self._pointers._load_pointer_ref(pointer_ref)
            if (
                pointer.pointer_kind is not PointerKind.CURRENT
                or pointer.target_ref != publication_ref
                or self._pointers.resolve_current_pointer_ref(pointer_id) != pointer_ref
            ):
                raise ValueError("returned pointer is no longer the expected B1.3 CURRENT")
        except Exception as exc:
            raise B13DispositionPersistenceError("B1.3 CURRENT changed after publication CAS") from exc
        self._request_current_authority(preparation)
        self._require_observed_current(pointer_id, pointer_ref)

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
                           pointer_ref: ArtifactRef | None) -> None:
        self._request_current_authority(preparation)
        pointer_id = b13_disposition_pointer_id(project_id, target_id)
        try:
            self._require_observed_current(pointer_id, pointer_ref)
        except B13DispositionPersistenceError as exc:
            raise B13DispositionPersistenceError("B1.3 CURRENT changed during verification") from exc


__all__ = [
    "B13DispositionPersistenceError", "B13VerifiedCurrentSnapshot",
    "B13DispositionPublicationResult",
    "B13DispositionPersistenceService", "b13_disposition_pointer_id",
]
