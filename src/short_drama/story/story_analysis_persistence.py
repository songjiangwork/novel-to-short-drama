"""A6F coherent immutable publication and current-only verification.

This is intentionally a small analogue of A5 persistence: one revision covers
all five leaves, the root manifest and its PASS report.  It never searches
history for a reusable result.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from short_drama.artifacts import ArtifactError, ArtifactNotFoundError, ArtifactRef, FileArtifactStore, ImmutableArtifactEnvelope
from short_drama.foundation import (FilePointerStore, LineageRef, PointerKind, PointerNotFoundError,
    ValidationReport, ValidationResult, load_validation_report, persist_validation_report)
from short_drama.llm import PromptRegistry, SemanticLLMProfile
from .errors import StoryIntegrityError, StoryPersistenceError
from .story_analysis import (A6SemanticIdentity, A6UpstreamIdentity, ArcAnalysis, CharacterAnalysisSet,
    GlobalEventAnalysis, GlobalStoryBible, GlobalStructure, StoryAnalysisCoverageSummary, StoryAnalysisManifest,
    StoryAnalysisProfile)
from .story_analysis_planning import StoryAnalysisInputSnapshot, StoryAnalysisPlan

CHARACTER_ANALYSIS_SET_ARTIFACT_TYPE = "character_analysis_set"
GLOBAL_EVENT_ANALYSIS_ARTIFACT_TYPE = "global_event_analysis"
ARC_ANALYSIS_ARTIFACT_TYPE = "arc_analysis"
GLOBAL_STRUCTURE_ARTIFACT_TYPE = "global_structure"
GLOBAL_STORY_BIBLE_ARTIFACT_TYPE = "global_story_bible"
STORY_ANALYSIS_MANIFEST_ARTIFACT_TYPE = "story_analysis_manifest"

def a6_base_artifact_id(project_id: str, document_id: str, profile_id: str) -> str: return f"{project_id}.{document_id}.a6.{profile_id}"
def character_analysis_artifact_id(base: str) -> str: return f"{base}.characters"
def global_event_analysis_artifact_id(base: str) -> str: return f"{base}.events"
def arc_analysis_artifact_id(base: str) -> str: return f"{base}.arcs"
def global_structure_artifact_id(base: str) -> str: return f"{base}.structure"
def global_story_bible_artifact_id(base: str) -> str: return f"{base}.bible"
def story_analysis_manifest_artifact_id(base: str) -> str: return f"{base}.manifest"
def a6_validation_artifact_id(base: str) -> str: return f"{base}.manifest.a6-validation"
def a6_pointer_id(project_id: str, document_id: str, profile_id: str) -> str: return f"{project_id}.a6.{document_id}.{profile_id}"

def _slots(base: str):
    return ((CHARACTER_ANALYSIS_SET_ARTIFACT_TYPE,character_analysis_artifact_id(base)),(GLOBAL_EVENT_ANALYSIS_ARTIFACT_TYPE,global_event_analysis_artifact_id(base)),(ARC_ANALYSIS_ARTIFACT_TYPE,arc_analysis_artifact_id(base)),(GLOBAL_STRUCTURE_ARTIFACT_TYPE,global_structure_artifact_id(base)),(GLOBAL_STORY_BIBLE_ARTIFACT_TYPE,global_story_bible_artifact_id(base)),(STORY_ANALYSIS_MANIFEST_ARTIFACT_TYPE,story_analysis_manifest_artifact_id(base)),("validation_report",a6_validation_artifact_id(base)))

def next_a6_revision(store: FileArtifactStore, *, base: str, current_manifest_ref: ArtifactRef | None) -> int:
    revision = 1 if current_manifest_ref is None else current_manifest_ref.revision + 1
    while True:
        if all(_missing(store, t, i, revision) for t, i in _slots(base)): return revision
        revision += 1

def _missing(store, typ, ident, rev):
    try: store.get(typ, ident, rev); return False
    except ArtifactNotFoundError: return True
    except ArtifactError as exc: raise StoryPersistenceError(f"cannot inspect A6 revision slot {typ}/{ident}") from exc

def _persist(store: FileArtifactStore, artifact_type: str, artifact_id: str, revision: int, payload: dict[str, Any]) -> ArtifactRef:
    try:
        return store.put(ImmutableArtifactEnvelope.create(artifact_type=artifact_type, artifact_id=artifact_id, revision=revision, schema_version=1, payload=payload))
    except Exception as exc: raise StoryPersistenceError(f"failed to persist {artifact_type}") from exc

def _load(store: FileArtifactStore, ref: ArtifactRef, typ: str, ident: str, cls):
    if ref.artifact_type != typ or ref.artifact_id != ident: raise StoryIntegrityError(f"A6 {typ} reference has wrong identity")
    try: envelope = store.get_ref(ref)
    except Exception as exc: raise StoryIntegrityError(f"cannot resolve A6 {typ}") from exc
    if envelope.schema_version != 1 or not isinstance(envelope.payload, dict): raise StoryIntegrityError(f"invalid persisted A6 {typ}")
    try: typed = cls.from_dict(envelope.payload)
    except Exception as exc: raise StoryIntegrityError(f"invalid persisted A6 {typ}") from exc
    if typed.to_dict() != envelope.payload: raise StoryIntegrityError(f"A6 {typ} is not canonical")
    return typed

def build_a6_semantic_identity(*, snapshot: StoryAnalysisInputSnapshot, plan: StoryAnalysisPlan,
    profile: StoryAnalysisProfile, semantic_profile: SemanticLLMProfile, characters: CharacterAnalysisSet,
    events: GlobalEventAnalysis, arcs: ArcAnalysis, structure: GlobalStructure, prompts: PromptRegistry) -> A6SemanticIdentity:
    """Reconstruct the sole A6 identity authority from actual inputs/outputs.

    This is used by both publication and reuse.  A caller may not supply hashes
    which are merely asserted to be current.
    """
    from .story_analysis_semantic import build_character_semantic_preparation
    from .story_analysis_window_semantic import build_plot_window_semantic_preparation
    from .story_analysis_global_skeleton_semantic import build_global_skeleton_semantic_preparation
    from .story_analysis_story_bible_semantic import build_story_bible_semantic_preparation
    cp = build_character_semantic_preparation(plan, profile, semantic_profile, prompts=prompts)
    wp = build_plot_window_semantic_preparation(snapshot, plan, profile, semantic_profile, prompts=prompts)
    gp = build_global_skeleton_semantic_preparation(snapshot, plan, profile, semantic_profile,
        characters.analyses, events.windows, prompts=prompts,
        character_request_identity_hashes=cp.character_request_identity_hashes,
        window_request_identity_hashes=wp.window_request_identity_hashes)
    bp = build_story_bible_semantic_preparation(snapshot, plan, profile, semantic_profile,
        characters.analyses, events, arcs, structure, prompts=prompts,
        character_request_identity_hashes=cp.character_request_identity_hashes,
        window_request_identity_hashes=wp.window_request_identity_hashes,
        global_skeleton_request_identity_hash=gp.request_identity_hash)
    return A6SemanticIdentity(profile.profile_id, profile.content_hash(), semantic_profile.profile_id,
        semantic_profile.semantic_profile_hash,
        (cp.prompt_identity, wp.prompt_identity, gp.prompt_identity, bp.prompt_identity),
        (cp.output_schema_identity, wp.output_schema_identity, gp.output_schema_identity, bp.output_schema_identity),
        plan.plan_hash, cp.character_request_identity_hashes, wp.window_request_identity_hashes,
        (gp.request_identity_hash,), (bp.request_identity_hash,))

def validate_whole_a6(snapshot: StoryAnalysisInputSnapshot, plan: StoryAnalysisPlan, characters: CharacterAnalysisSet,
    events: GlobalEventAnalysis, arcs: ArcAnalysis, structure: GlobalStructure, bible: GlobalStoryBible) -> None:
    if snapshot.consolidation_manifest_ref != plan.consolidation_manifest_ref: raise StoryIntegrityError("A6 plan is not bound to exact validated A5 CURRENT")
    expected_chars = [x.character_ref for x in plan.character_packages]
    if [x.character_ref for x in characters.analyses] != expected_chars: raise StoryIntegrityError("A6 character coverage is incomplete or reordered")
    if events.plot_window_plan_hash != plan.plan_hash or len(events.windows) != len(plan.windows): raise StoryIntegrityError("GlobalEventAnalysis plan binding/coverage mismatch")
    for got, expected in zip(events.windows, plan.windows):
        if (got.window_id, got.window_ordinal, tuple(got.owned_event_refs), tuple(got.context_event_refs)) != (expected.window_id, expected.window_ordinal, tuple(expected.owned_event_ids), tuple(expected.context_event_ids)):
            raise StoryIntegrityError("A6 window identity/order/ownership differs from deterministic plan")
    owned = [evt for window in events.windows for evt in window.owned_event_refs]
    if len(owned) != len(set(owned)) or set(owned) != set(plan.event_stream): raise StoryIntegrityError("A6 event ownership is not exact-one")
    universe = {"char": {x.canonical_id for x in snapshot.canonical_characters}, "evt": {x.event_id for x in snapshot.events}, "fact": {x.fact_id for x in snapshot.facts}, "rel": {x.relationship_id for x in snapshot.relationships}, "conf": {x.conflict_id for x in snapshot.conflicts}}
    def interp(x):
        for attr, key in (("supporting_fact_refs","fact"),("supporting_event_refs","evt"),("supporting_relationship_refs","rel"),("supporting_conflict_refs","conf")):
            if not set(getattr(x,attr)) <= universe[key]: raise StoryIntegrityError("A6 interpretation has unsupported canonical namespace/ref")
    for analysis in characters.analyses:
        if analysis.character_ref not in universe["char"]: raise StoryIntegrityError("unknown character analysis ref")
        if not set(analysis.key_event_refs)<=universe["evt"] or not set(analysis.key_fact_refs)<=universe["fact"] or not set(analysis.important_relationship_refs)<=universe["rel"]: raise StoryIntegrityError("CharacterAnalysis has dangling canonical refs")
        for value in (analysis.role, analysis.arc_summary, *analysis.goals, *analysis.motivations, *analysis.traits, *analysis.unresolved_or_conflicting_points): interp(value)
    for window in events.windows:
        for value in (window.interpretation,*window.candidate_turning_points,*window.candidate_reveals,*window.arc_continuation_markers): interp(value)
    for e in events.event_importance_overlay:
        if e.event_ref not in universe["evt"]: raise StoryIntegrityError("unknown event overlay ref")
        interp(e.interpretation)
    arc_ids={x.arc_id for x in arcs.arcs}; turn_ids={x.turning_point_id for x in arcs.turning_points}; reveal_ids={x.reveal_id for x in arcs.reveals}
    for arc in arcs.arcs:
        if not set((arc.start_event_ref,arc.end_event_ref,*arc.supporting_event_refs))<=universe["evt"] or not set(arc.involved_character_refs)<=universe["char"] or not set(arc.involved_relationship_refs)<=universe["rel"] or not set(arc.supporting_fact_refs)<=universe["fact"]: raise StoryIntegrityError("StoryArc has dangling refs")
        interp(arc.interpretation)
    for turn in arcs.turning_points:
        if turn.event_ref not in universe["evt"] or not set(turn.supporting_fact_refs)<=universe["fact"] or not set(turn.supporting_relationship_refs)<=universe["rel"]: raise StoryIntegrityError("TurningPoint has dangling refs")
        interp(turn.interpretation)
    for reveal in arcs.reveals:
        if not set((*reveal.reveal_event_refs,*reveal.setup_event_refs))<=universe["evt"] or not set(reveal.supporting_fact_refs)<=universe["fact"] or not set(reveal.affected_character_refs)<=universe["char"]: raise StoryIntegrityError("Reveal has dangling refs")
        interp(reveal.interpretation)
    for payoff in arcs.foreshadow_payoffs:
        if not set((*payoff.setup_event_refs,*payoff.payoff_event_refs))<=universe["evt"]: raise StoryIntegrityError("ForeshadowPayoff has dangling refs")
        interp(payoff.interpretation)
    if not set(structure.main_character_refs)<=universe["char"] or not set(structure.major_arc_refs)<=arc_ids or not set(structure.major_turning_point_refs)<=turn_ids or not set(structure.major_reveal_refs)<=reveal_ids: raise StoryIntegrityError("GlobalStructure has dangling A6 refs")
    if not set(bible.main_character_refs)<=universe["char"] or not set(bible.major_arc_refs)<=arc_ids or not set(bible.major_turning_point_refs)<=turn_ids or not set(bible.major_reveal_refs)<=reveal_ids: raise StoryIntegrityError("StoryBible has dangling A6 refs")
    for value in (structure.main_conflict,structure.main_plot,structure.ending_state,*structure.secondary_conflicts,*structure.subplots): interp(value)
    for section in structure.global_sections:
        if not set(section.event_refs)<=universe["evt"]: raise StoryIntegrityError("GlobalStructure section has dangling event refs")
        interp(section.interpretation)
    for value in (bible.premise_zh,bible.synopsis_zh,bible.setting,bible.unresolved_summary,bible.source_conflict_summary,*bible.themes): interp(value)

def build_a6_validation_report(manifest: StoryAnalysisManifest, manifest_ref: ArtifactRef) -> ValidationReport:
    refs=(LineageRef("a5_manifest",manifest.consolidation_manifest_ref),LineageRef("character_analysis",manifest.character_analysis_ref),LineageRef("global_event_analysis",manifest.global_event_analysis_ref),LineageRef("arc_analysis",manifest.arc_analysis_ref),LineageRef("global_structure",manifest.global_structure_ref),LineageRef("global_story_bible",manifest.global_story_bible_ref),LineageRef("story_analysis_manifest",manifest_ref))
    return ValidationReport(validated_refs=refs, findings=())

@dataclass(frozen=True, slots=True)
class StoryAnalysisPublication:
    character_analysis_ref: ArtifactRef; global_event_analysis_ref: ArtifactRef; arc_analysis_ref: ArtifactRef; global_structure_ref: ArtifactRef; global_story_bible_ref: ArtifactRef; story_analysis_manifest_ref: ArtifactRef; validation_report_ref: ArtifactRef; current_pointer_ref: ArtifactRef; reused: bool=False

class StoryAnalysisPersistenceService:
    def __init__(self, store: FileArtifactStore, pointers: FilePointerStore): self._store,self._pointers=store,pointers
    def publish_validated(self, *, project_id: str, document_id: str, profile: StoryAnalysisProfile,
        semantic_profile: SemanticLLMProfile, prompts: PromptRegistry, snapshot: StoryAnalysisInputSnapshot,
        plan: StoryAnalysisPlan, characters: CharacterAnalysisSet, events: GlobalEventAnalysis,
        arcs: ArcAnalysis, structure: GlobalStructure, bible: GlobalStoryBible) -> StoryAnalysisPublication:
        validate_whole_a6(snapshot,plan,characters,events,arcs,structure,bible)
        if (snapshot.consolidation_manifest.project_id != project_id or
                snapshot.consolidation_manifest.document_id != document_id):
            raise StoryIntegrityError("publication project/document disagree with authoritative A5 snapshot")
        if profile.planning_policy != plan.planning_policy: raise StoryIntegrityError("A6 profile and deterministic plan disagree")
        semantic_identity=build_a6_semantic_identity(snapshot=snapshot,plan=plan,profile=profile,semantic_profile=semantic_profile,characters=characters,events=events,arcs=arcs,structure=structure,prompts=prompts)
        base=a6_base_artifact_id(project_id,document_id,profile.profile_id); pointer=a6_pointer_id(project_id,document_id,profile.profile_id)
        try: old_ptr=self._pointers.resolve_current_pointer_ref(pointer); old_manifest=self._pointers.resolve_current(pointer).target_ref
        except PointerNotFoundError: old_ptr=old_manifest=None
        if old_manifest is not None:
            self._load_current_integrity(base, old_manifest)
        revision=next_a6_revision(self._store,base=base,current_manifest_ref=old_manifest)
        cr=_persist(self._store,CHARACTER_ANALYSIS_SET_ARTIFACT_TYPE,character_analysis_artifact_id(base),revision,characters.to_dict()); er=_persist(self._store,GLOBAL_EVENT_ANALYSIS_ARTIFACT_TYPE,global_event_analysis_artifact_id(base),revision,events.to_dict()); ar=_persist(self._store,ARC_ANALYSIS_ARTIFACT_TYPE,arc_analysis_artifact_id(base),revision,arcs.to_dict()); sr=_persist(self._store,GLOBAL_STRUCTURE_ARTIFACT_TYPE,global_structure_artifact_id(base),revision,structure.to_dict()); br=_persist(self._store,GLOBAL_STORY_BIBLE_ARTIFACT_TYPE,global_story_bible_artifact_id(base),revision,bible.to_dict())
        coverage=StoryAnalysisCoverageSummary(len(snapshot.canonical_characters),len(characters.analyses),len(snapshot.events),len({e for w in events.windows for e in w.owned_event_refs}),len(events.windows),len(events.event_importance_overlay),len(arcs.arcs),len(arcs.turning_points),len(arcs.reveals),len(arcs.foreshadow_payoffs))
        manifest=StoryAnalysisManifest(1,project_id,document_id,snapshot.consolidation_manifest_ref,cr,er,ar,sr,br,semantic_identity,A6UpstreamIdentity(snapshot.consolidation_manifest_ref),coverage)
        mr=_persist(self._store,STORY_ANALYSIS_MANIFEST_ARTIFACT_TYPE,story_analysis_manifest_artifact_id(base),revision,manifest.to_dict()); report=build_a6_validation_report(manifest,mr)
        if report.summary.result is not ValidationResult.PASS: raise StoryIntegrityError("A6 report not PASS")
        rr=persist_validation_report(self._store,report,artifact_id=a6_validation_artifact_id(base),revision=revision)
        # An A5 advance between the input snapshot and pointer mutation is fatal.
        if snapshot.a5_current_pointer_ref != self._pointers.resolve_current_pointer_ref(snapshot.a5_current_pointer_ref.artifact_id): raise StoryPersistenceError("A5 CURRENT changed during A6 publication")
        try: pr=self._pointers.compare_and_set(pointer_id=pointer,pointer_kind=PointerKind.CURRENT,expected_pointer_ref=old_ptr,target_ref=mr)
        except Exception as exc: raise StoryPersistenceError("A6 CURRENT CAS failed") from exc
        return StoryAnalysisPublication(cr,er,ar,sr,br,mr,rr,pr)

    def try_reuse_current(self, *, project_id: str, document_id: str, profile: StoryAnalysisProfile,
        semantic_profile: SemanticLLMProfile, prompts: PromptRegistry, snapshot: StoryAnalysisInputSnapshot,
        plan: StoryAnalysisPlan) -> StoryAnalysisPublication | None:
        """Return only an exact, currently-published, fully verified A6 hit.

        A missing CURRENT is a normal fresh-run miss.  A malformed CURRENT is
        an integrity failure; this deliberately never scans historical revisions.
        """
        base=a6_base_artifact_id(project_id,document_id,profile.profile_id); pointer=a6_pointer_id(project_id,document_id,profile.profile_id)
        try:
            pointer_ref=self._pointers.resolve_current_pointer_ref(pointer)
            manifest_ref=self._pointers.resolve_current(pointer).target_ref
        except PointerNotFoundError: return None
        manifest, chars, events, arcs, structure, bible, report_ref = self._load_current_integrity(base, manifest_ref)
        if manifest.project_id!=project_id or manifest.document_id!=document_id: raise StoryIntegrityError("A6 CURRENT manifest project/document mismatch")
        revision=manifest_ref.revision
        refs=(manifest.character_analysis_ref,manifest.global_event_analysis_ref,manifest.arc_analysis_ref,manifest.global_structure_ref,manifest.global_story_bible_ref)
        # A different A5 authority or deterministic plan is a normal miss only
        # after the persisted tree/report have passed intrinsic integrity checks.
        # Do not validate old leaves against the new plan: that would turn a
        # legitimate profile/plan change into a false corruption finding.
        if manifest.consolidation_manifest_ref != snapshot.consolidation_manifest_ref: return None
        if manifest.semantic_identity.plan_hash != plan.plan_hash: return None
        if (manifest.semantic_identity.story_analysis_profile_id != profile.profile_id or
                manifest.semantic_identity.story_analysis_profile_hash != profile.content_hash() or
                manifest.semantic_identity.semantic_profile_id != semantic_profile.profile_id or
                manifest.semantic_identity.semantic_profile_hash != semantic_profile.semantic_profile_hash):
            return None
        validate_whole_a6(snapshot,plan,chars,events,arcs,structure,bible)
        identity=build_a6_semantic_identity(snapshot=snapshot,plan=plan,profile=profile,semantic_profile=semantic_profile,characters=chars,events=events,arcs=arcs,structure=structure,prompts=prompts)
        if manifest.semantic_identity != identity: return None
        if snapshot.a5_current_pointer_ref != self._pointers.resolve_current_pointer_ref(snapshot.a5_current_pointer_ref.artifact_id): raise StoryPersistenceError("A5 CURRENT pointer changed during reuse eligibility check")
        return StoryAnalysisPublication(*refs,manifest_ref,report_ref,pointer_ref,reused=True)

    def _load_current_integrity(self, base: str, manifest_ref: ArtifactRef):
        manifest=_load(self._store,manifest_ref,STORY_ANALYSIS_MANIFEST_ARTIFACT_TYPE,story_analysis_manifest_artifact_id(base),StoryAnalysisManifest)
        revision=manifest_ref.revision; refs=(manifest.character_analysis_ref,manifest.global_event_analysis_ref,manifest.arc_analysis_ref,manifest.global_structure_ref,manifest.global_story_bible_ref)
        if any(ref.revision != revision for ref in refs): raise StoryIntegrityError("A6 CURRENT mixes revisions")
        chars=_load(self._store,refs[0],CHARACTER_ANALYSIS_SET_ARTIFACT_TYPE,character_analysis_artifact_id(base),CharacterAnalysisSet); events=_load(self._store,refs[1],GLOBAL_EVENT_ANALYSIS_ARTIFACT_TYPE,global_event_analysis_artifact_id(base),GlobalEventAnalysis); arcs=_load(self._store,refs[2],ARC_ANALYSIS_ARTIFACT_TYPE,arc_analysis_artifact_id(base),ArcAnalysis); structure=_load(self._store,refs[3],GLOBAL_STRUCTURE_ARTIFACT_TYPE,global_structure_artifact_id(base),GlobalStructure); bible=_load(self._store,refs[4],GLOBAL_STORY_BIBLE_ARTIFACT_TYPE,global_story_bible_artifact_id(base),GlobalStoryBible)
        coverage=manifest.coverage_summary
        if (coverage.character_analysis_count, coverage.owned_event_count, coverage.plot_window_count, coverage.global_event_importance_count, coverage.arc_count, coverage.turning_point_count, coverage.reveal_count, coverage.foreshadow_payoff_count) != (len(chars.analyses), len({e for w in events.windows for e in w.owned_event_refs}), len(events.windows), len(events.event_importance_overlay), len(arcs.arcs), len(arcs.turning_points), len(arcs.reveals), len(arcs.foreshadow_payoffs)):
            raise StoryIntegrityError("A6 manifest coverage summary differs from leaves")
        try: report_ref=self._store.get("validation_report",a6_validation_artifact_id(base),revision).ref; report=load_validation_report(self._store,report_ref)
        except Exception as exc: raise StoryIntegrityError("A6 CURRENT lacks valid PASS ValidationReport") from exc
        if report != build_a6_validation_report(manifest,manifest_ref) or report.summary.result is not ValidationResult.PASS: raise StoryIntegrityError("A6 ValidationReport does not exactly validate CURRENT manifest")
        return manifest, chars, events, arcs, structure, bible, report_ref
