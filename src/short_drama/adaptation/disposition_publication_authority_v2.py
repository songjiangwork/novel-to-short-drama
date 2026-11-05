"""Unratified in-memory historical authority reconstruction for B4A v2.

Hash-linked envelopes provide internal consistency, not external authentication.
The caller must independently anchor the source/strategy references.
"""
from short_drama.artifacts import ImmutableArtifactEnvelope
from short_drama.story.consolidation import CanonicalEventSet, CanonicalFactSet, CanonicalRelationshipSet, ConsolidationManifest
from short_drama.story.reconciliation import CanonicalCharacterRegistry, CanonicalLocationRegistry, EntityMap
from short_drama.story.story_analysis import ArcAnalysis, GlobalEventAnalysis, StoryAnalysisManifest
from .disposition_preparation import build_b13_disposition_preparation_from_authority
from .target import CanonicalTargetReference, _canonical_references_from_a4


class B13PublicationAuthorityError(ValueError):
    pass


def reconstruct_publication_authority_v2(source, selection, evidence):
    """Verify embedded envelopes and reconstruct the full ordered preparation."""
    try:
        names = {'a5_manifest', 'a6_manifest', 'entity_map', 'characters', 'locations',
                 'facts', 'events', 'relationships', 'arc_analysis', 'event_analysis'}
        if type(evidence) is not dict or set(evidence) != names:
            raise ValueError
        def unpack(name, ref, model):
            envelope = ImmutableArtifactEnvelope.from_dict(evidence[name])
            if envelope.ref != ref or envelope.schema_version != (2 if name == 'entity_map' else 1):
                raise ValueError
            obj = model.from_dict(envelope.payload)
            if obj.to_dict() != envelope.payload:
                raise ValueError
            return obj
        a5 = unpack('a5_manifest', source.consolidation_manifest_ref, ConsolidationManifest)
        a6 = unpack('a6_manifest', source.story_analysis_manifest_ref, StoryAnalysisManifest)
        if (a6.consolidation_manifest_ref != source.consolidation_manifest_ref
                or a6.arc_analysis_ref != source.arc_analysis_ref
                or (a5.project_id, a5.document_id) != (a6.project_id, a6.document_id)
                or a5.entity_map_ref != source.entity_map_ref
                or a5.canonical_fact_set_ref != source.canonical_fact_set_ref
                or a5.canonical_event_set_ref != source.canonical_event_set_ref
                or a5.canonical_relationship_set_ref != source.canonical_relationship_set_ref):
            raise ValueError
        entity_map = unpack('entity_map', source.entity_map_ref, EntityMap)
        if (entity_map.canonical_character_registry_ref != source.canonical_character_registry_ref
                or entity_map.canonical_location_registry_ref != source.canonical_location_registry_ref):
            raise ValueError
        chars = unpack('characters', source.canonical_character_registry_ref, CanonicalCharacterRegistry)
        locations = unpack('locations', source.canonical_location_registry_ref, CanonicalLocationRegistry)
        facts = unpack('facts', source.canonical_fact_set_ref, CanonicalFactSet)
        events = unpack('events', source.canonical_event_set_ref, CanonicalEventSet)
        relationships = unpack('relationships', source.canonical_relationship_set_ref, CanonicalRelationshipSet)
        arcs = unpack('arc_analysis', source.arc_analysis_ref, ArcAnalysis)
        event_analysis = unpack('event_analysis', a6.global_event_analysis_ref, GlobalEventAnalysis)
        refs = list(_canonical_references_from_a4(entity_map, chars.entities, locations.entities))
        for kind, items, field in (('fact', facts.facts, 'fact_id'), ('event', events.events, 'event_id'),
                                    ('relationship', relationships.relationships, 'relationship_id'),
                                    ('arc', arcs.arcs, 'arc_id'), ('turning_point', arcs.turning_points, 'turning_point_id'),
                                    ('reveal', arcs.reveals, 'reveal_id'), ('payoff', arcs.foreshadow_payoffs, 'payoff_id')):
            refs.extend(CanonicalTargetReference(kind, getattr(item, field)) for item in items)
        if tuple(refs) != source.canonical_references:
            raise ValueError
        return build_b13_disposition_preparation_from_authority(
            source_snapshot=source, strategy_selection=selection, event_set=events,
            event_analysis=event_analysis, arc_analysis=arcs)
    except Exception:
        raise B13PublicationAuthorityError('historical publication authority is invalid') from None
