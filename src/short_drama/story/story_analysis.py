"""A6A static Global Story Bible contracts; no planning, provider, or persistence."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from short_drama.artifacts import ArtifactRef, content_hash
from short_drama.story.consolidation import OutputSchemaAssetIdentity, PromptAssetIdentity
from short_drama.story.errors import StoryAnalysisModelError

STORY_ANALYSIS_SCHEMA_VERSION = 1
EVIDENCE_MODES = frozenset(("explicit", "inferred"))
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_REF = {k: re.compile(v) for k, v in {
    "char": r"^char_[0-9]{4,}$", "loc": r"^loc_[0-9]{4,}$",
    "unres": r"^unres_[0-9]{4,}$", "fact": r"^fact_[0-9]{6,}$",
    "evt": r"^evt_[0-9]{6,}$", "rel": r"^rel_[0-9]{6,}$",
    "trans": r"^trans_[0-9]{6,}$", "conf": r"^conf_[0-9]{6,}$",
    "window": r"^window_[0-9]{4,}$", "arc": r"^arc_[0-9]{6,}$",
    "turn": r"^turn_[0-9]{6,}$", "reveal": r"^reveal_[0-9]{6,}$",
    "payoff": r"^payoff_[0-9]{6,}$",
}.items()}


def _text(v: object, n: str) -> str:
    if not isinstance(v, str) or not v.strip() or "\0" in v: raise StoryAnalysisModelError(f"{n} must be non-empty text")
    return v
def _id(v: object, n: str) -> str:
    v = _text(v, n)
    if not _ID.fullmatch(v): raise StoryAnalysisModelError(f"{n} must be a safe identifier")
    return v
def _pos(v: object, n: str, *, zero: bool = False) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < (0 if zero else 1): raise StoryAnalysisModelError(f"{n} must be a {'non-negative' if zero else 'positive'} integer")
    return v
def _sha(v: object, n: str) -> str:
    if not isinstance(v, str) or not _SHA.fullmatch(v): raise StoryAnalysisModelError(f"{n} must be sha256")
    return v
def _keys(v: object, expected: set[str], n: str) -> dict[str, Any]:
    if not isinstance(v, dict) or set(v) != expected: raise StoryAnalysisModelError(f"{n} must have exact keys {sorted(expected)!r}")
    return v
def _refs(v: object, ns: str, n: str) -> tuple[str, ...]:
    if not isinstance(v, (tuple, list)): raise StoryAnalysisModelError(f"{n} must be an array")
    items = tuple(v)
    if any(not isinstance(x, str) or not _REF[ns].fullmatch(x) for x in items) or len(set(items)) != len(items): raise StoryAnalysisModelError(f"{n} contains invalid or duplicate {ns} refs")
    return items
def _interp(v: object, n: str) -> tuple["EvidenceBackedInterpretation", ...]:
    if not isinstance(v, (tuple, list)): raise StoryAnalysisModelError(f"{n} must be an array")
    items = tuple(v)
    if any(not isinstance(x, EvidenceBackedInterpretation) for x in items): raise StoryAnalysisModelError(f"{n} must contain interpretations")
    return items
def _count_map(v: object, n: str) -> tuple[tuple[str, int], ...]:
    if isinstance(v, dict): v = tuple(v.items())
    if not isinstance(v, (tuple, list)): raise StoryAnalysisModelError(f"{n} must be a count object")
    items = tuple(v)
    if any(not isinstance(x, tuple) or len(x) != 2 for x in items): raise StoryAnalysisModelError(f"{n} must be count pairs")
    for k, c in items: _id(k, f"{n}.key"); _pos(c, f"{n}.{k}", zero=True)
    if len({k for k, _ in items}) != len(items): raise StoryAnalysisModelError(f"{n} has duplicate keys")
    return items


@dataclass(frozen=True, slots=True)
class StoryAnalysisSemanticPass:
    semantic_profile_id: str; prompt_id: str; prompt_version: int; output_schema_id: str; output_schema_version: int
    def __post_init__(self):
        _id(self.semantic_profile_id,"semantic_profile_id"); _text(self.prompt_id,"prompt_id"); _pos(self.prompt_version,"prompt_version"); _id(self.output_schema_id,"output_schema_id"); _pos(self.output_schema_version,"output_schema_version")
    def to_dict(self): return {n:getattr(self,n) for n in self.__dataclass_fields__}
    @classmethod
    def from_dict(cls,v): return cls(**_keys(v,set(cls.__dataclass_fields__),cls.__name__))

@dataclass(frozen=True, slots=True)
class StoryAnalysisPlanningPolicy:
    character_packet_max_estimated_tokens: int; plot_window_packet_max_estimated_tokens: int; plot_window_owned_event_target: int; plot_window_context_event_count: int; global_skeleton_packet_max_estimated_tokens: int; story_bible_packet_max_estimated_tokens: int
    def __post_init__(self):
        for n in self.__dataclass_fields__: _pos(getattr(self,n),n,zero=n=="plot_window_context_event_count")
    def to_dict(self): return {n:getattr(self,n) for n in self.__dataclass_fields__}
    @classmethod
    def from_dict(cls,v): return cls(**_keys(v,set(cls.__dataclass_fields__),cls.__name__))

@dataclass(frozen=True, slots=True)
class StoryAnalysisProfile:
    """Future A6 profile, distinct from A2 ``story_analysis_v1.yaml``."""
    schema_version: int; profile_id: str; working_language: str; character_analysis_policy_id: str; plot_window_policy_id: str; global_skeleton_policy_id: str; story_bible_policy_id: str; max_generation_rounds: int; planning_policy: StoryAnalysisPlanningPolicy; character_analysis: StoryAnalysisSemanticPass; plot_window_analysis: StoryAnalysisSemanticPass; global_skeleton: StoryAnalysisSemanticPass; story_bible: StoryAnalysisSemanticPass
    def __post_init__(self):
        if self.schema_version != 1: raise StoryAnalysisModelError("unsupported schema_version")
        for n in ("profile_id","character_analysis_policy_id","plot_window_policy_id","global_skeleton_policy_id","story_bible_policy_id"): _id(getattr(self,n),n)
        _text(self.working_language,"working_language"); _pos(self.max_generation_rounds,"max_generation_rounds")
        if not isinstance(self.planning_policy,StoryAnalysisPlanningPolicy) or any(not isinstance(getattr(self,n),StoryAnalysisSemanticPass) for n in ("character_analysis","plot_window_analysis","global_skeleton","story_bible")): raise StoryAnalysisModelError("invalid profile nested contract")
    def to_dict(self):
        scalar=("schema_version","profile_id","working_language","character_analysis_policy_id","plot_window_policy_id","global_skeleton_policy_id","story_bible_policy_id","max_generation_rounds")
        return {**{n:getattr(self,n) for n in scalar},"planning_policy":self.planning_policy.to_dict(),**{n:getattr(self,n).to_dict() for n in ("character_analysis","plot_window_analysis","global_skeleton","story_bible")}}
    def content_hash(self): return content_hash(self.to_dict())
    @classmethod
    def from_dict(cls,v):
        v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); return cls(**{**v,"planning_policy":StoryAnalysisPlanningPolicy.from_dict(v["planning_policy"]),**{n:StoryAnalysisSemanticPass.from_dict(v[n]) for n in ("character_analysis","plot_window_analysis","global_skeleton","story_bible")}})

def load_story_analysis_profile(path: str | Path) -> StoryAnalysisProfile:
    try: raw=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError,yaml.YAMLError) as exc: raise StoryAnalysisModelError(f"invalid story analysis profile: {path}") from exc
    return StoryAnalysisProfile.from_dict(raw)

@dataclass(frozen=True, slots=True)
class EvidenceBackedInterpretation:
    text_zh: str; evidence_mode: str; supporting_fact_refs: tuple[str,...]=(); supporting_event_refs: tuple[str,...]=(); supporting_relationship_refs: tuple[str,...]=(); supporting_conflict_refs: tuple[str,...]=()
    def __post_init__(self):
        _text(self.text_zh,"text_zh")
        if self.evidence_mode not in EVIDENCE_MODES: raise StoryAnalysisModelError("invalid evidence_mode")
        for n,ns in (("supporting_fact_refs","fact"),("supporting_event_refs","evt"),("supporting_relationship_refs","rel"),("supporting_conflict_refs","conf")): object.__setattr__(self,n,_refs(getattr(self,n),ns,n))
    def to_dict(self): return {"text_zh":self.text_zh,"evidence_mode":self.evidence_mode,"supporting_fact_refs":list(self.supporting_fact_refs),"supporting_event_refs":list(self.supporting_event_refs),"supporting_relationship_refs":list(self.supporting_relationship_refs),"supporting_conflict_refs":list(self.supporting_conflict_refs)}
    @classmethod
    def from_dict(cls,v): return cls(**_keys(v,set(cls.__dataclass_fields__),cls.__name__))

@dataclass(frozen=True, slots=True)
class CharacterAnalysis:
    character_ref: str; role: EvidenceBackedInterpretation; goals: tuple[EvidenceBackedInterpretation,...]; motivations: tuple[EvidenceBackedInterpretation,...]; traits: tuple[EvidenceBackedInterpretation,...]; key_event_refs: tuple[str,...]; key_fact_refs: tuple[str,...]; important_relationship_refs: tuple[str,...]; arc_summary: EvidenceBackedInterpretation; unresolved_or_conflicting_points: tuple[EvidenceBackedInterpretation,...]
    def __post_init__(self):
        _refs((self.character_ref,),"char","character_ref")
        for n in ("role","arc_summary"):
            if not isinstance(getattr(self,n),EvidenceBackedInterpretation): raise StoryAnalysisModelError(f"{n} invalid")
        for n in ("goals","motivations","traits","unresolved_or_conflicting_points"): object.__setattr__(self,n,_interp(getattr(self,n),n))
        for n,ns in (("key_event_refs","evt"),("key_fact_refs","fact"),("important_relationship_refs","rel")): object.__setattr__(self,n,_refs(getattr(self,n),ns,n))
    def to_dict(self): return {"character_ref":self.character_ref,"role":self.role.to_dict(),"goals":[x.to_dict() for x in self.goals],"motivations":[x.to_dict() for x in self.motivations],"traits":[x.to_dict() for x in self.traits],"key_event_refs":list(self.key_event_refs),"key_fact_refs":list(self.key_fact_refs),"important_relationship_refs":list(self.important_relationship_refs),"arc_summary":self.arc_summary.to_dict(),"unresolved_or_conflicting_points":[x.to_dict() for x in self.unresolved_or_conflicting_points]}
    @classmethod
    def from_dict(cls,v):
        v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); return cls(**{**v,"role":EvidenceBackedInterpretation.from_dict(v["role"]),"arc_summary":EvidenceBackedInterpretation.from_dict(v["arc_summary"]),**{n:tuple(EvidenceBackedInterpretation.from_dict(x) for x in v[n]) for n in ("goals","motivations","traits","unresolved_or_conflicting_points")}})

@dataclass(frozen=True, slots=True)
class CharacterAnalysisSet:
    schema_version: int; analyses: tuple[CharacterAnalysis,...]
    def __post_init__(self):
        if self.schema_version != 1: raise StoryAnalysisModelError("unsupported schema_version")
        object.__setattr__(self,"analyses",tuple(self.analyses))
        if any(not isinstance(x,CharacterAnalysis) for x in self.analyses) or len({x.character_ref for x in self.analyses}) != len(self.analyses): raise StoryAnalysisModelError("invalid CharacterAnalysisSet")
    def to_dict(self): return {"schema_version":self.schema_version,"analyses":[x.to_dict() for x in self.analyses]}
    @classmethod
    def from_dict(cls,v): v=_keys(v,{"schema_version","analyses"},cls.__name__); return cls(v["schema_version"],tuple(CharacterAnalysis.from_dict(x) for x in v["analyses"]))

@dataclass(frozen=True, slots=True)
class PlotWindowAnalysis:
    window_id: str; window_ordinal: int; owned_event_refs: tuple[str,...]; context_event_refs: tuple[str,...]; interpretation: EvidenceBackedInterpretation; candidate_turning_points: tuple[EvidenceBackedInterpretation,...]=(); candidate_reveals: tuple[EvidenceBackedInterpretation,...]=(); arc_continuation_markers: tuple[EvidenceBackedInterpretation,...]=()
    def __post_init__(self):
        _refs((self.window_id,),"window","window_id"); _pos(self.window_ordinal,"window_ordinal")
        for n in ("owned_event_refs","context_event_refs"): object.__setattr__(self,n,_refs(getattr(self,n),"evt",n))
        if not isinstance(self.interpretation,EvidenceBackedInterpretation): raise StoryAnalysisModelError("interpretation invalid")
        for n in ("candidate_turning_points","candidate_reveals","arc_continuation_markers"): object.__setattr__(self,n,_interp(getattr(self,n),n))
    def to_dict(self): return {"window_id":self.window_id,"window_ordinal":self.window_ordinal,"owned_event_refs":list(self.owned_event_refs),"context_event_refs":list(self.context_event_refs),"interpretation":self.interpretation.to_dict(),"candidate_turning_points":[x.to_dict() for x in self.candidate_turning_points],"candidate_reveals":[x.to_dict() for x in self.candidate_reveals],"arc_continuation_markers":[x.to_dict() for x in self.arc_continuation_markers]}
    @classmethod
    def from_dict(cls,v):
        v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); return cls(**{**v,"interpretation":EvidenceBackedInterpretation.from_dict(v["interpretation"]),**{n:tuple(EvidenceBackedInterpretation.from_dict(x) for x in v[n]) for n in ("candidate_turning_points","candidate_reveals","arc_continuation_markers")}})

@dataclass(frozen=True, slots=True)
class GlobalEventImportance:
    event_ref: str; interpretation: EvidenceBackedInterpretation
    def __post_init__(self):
        _refs((self.event_ref,),"evt","event_ref")
        if not isinstance(self.interpretation,EvidenceBackedInterpretation): raise StoryAnalysisModelError("interpretation invalid")
    def to_dict(self): return {"event_ref":self.event_ref,"interpretation":self.interpretation.to_dict()}
    @classmethod
    def from_dict(cls,v): v=_keys(v,{"event_ref","interpretation"},cls.__name__); return cls(v["event_ref"],EvidenceBackedInterpretation.from_dict(v["interpretation"]))

@dataclass(frozen=True, slots=True)
class GlobalEventAnalysis:
    schema_version: int; plot_window_plan_hash: str; windows: tuple[PlotWindowAnalysis,...]; event_importance_overlay: tuple[GlobalEventImportance,...]; coverage_metadata: tuple[tuple[str,int],...]
    def __post_init__(self):
        if self.schema_version != 1: raise StoryAnalysisModelError("unsupported schema_version")
        _sha(self.plot_window_plan_hash,"plot_window_plan_hash"); object.__setattr__(self,"windows",tuple(self.windows)); object.__setattr__(self,"event_importance_overlay",tuple(self.event_importance_overlay))
        if any(not isinstance(x,PlotWindowAnalysis) for x in self.windows) or len({x.window_id for x in self.windows}) != len(self.windows): raise StoryAnalysisModelError("windows invalid")
        if any(not isinstance(x,GlobalEventImportance) for x in self.event_importance_overlay) or len({x.event_ref for x in self.event_importance_overlay}) != len(self.event_importance_overlay): raise StoryAnalysisModelError("event importance invalid")
        object.__setattr__(self,"coverage_metadata",_count_map(self.coverage_metadata,"coverage_metadata"))
    def to_dict(self): return {"schema_version":self.schema_version,"plot_window_plan_hash":self.plot_window_plan_hash,"windows":[x.to_dict() for x in self.windows],"event_importance_overlay":[x.to_dict() for x in self.event_importance_overlay],"coverage_metadata":dict(self.coverage_metadata)}
    @classmethod
    def from_dict(cls,v): v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); return cls(v["schema_version"],v["plot_window_plan_hash"],tuple(PlotWindowAnalysis.from_dict(x) for x in v["windows"]),tuple(GlobalEventImportance.from_dict(x) for x in v["event_importance_overlay"]),v["coverage_metadata"])

@dataclass(frozen=True, slots=True)
class StoryArc:
    arc_id: str; arc_kind: str; involved_character_refs: tuple[str,...]; involved_relationship_refs: tuple[str,...]; supporting_event_refs: tuple[str,...]; supporting_fact_refs: tuple[str,...]; start_event_ref: str; end_event_ref: str; interpretation: EvidenceBackedInterpretation
    def __post_init__(self):
        _refs((self.arc_id,),"arc","arc_id"); _text(self.arc_kind,"arc_kind")
        for n,ns in (("involved_character_refs","char"),("involved_relationship_refs","rel"),("supporting_event_refs","evt"),("supporting_fact_refs","fact")): object.__setattr__(self,n,_refs(getattr(self,n),ns,n))
        _refs((self.start_event_ref,),"evt","start_event_ref"); _refs((self.end_event_ref,),"evt","end_event_ref")
        if not isinstance(self.interpretation,EvidenceBackedInterpretation): raise StoryAnalysisModelError("interpretation invalid")
    def to_dict(self): return {"arc_id":self.arc_id,"arc_kind":self.arc_kind,"involved_character_refs":list(self.involved_character_refs),"involved_relationship_refs":list(self.involved_relationship_refs),"supporting_event_refs":list(self.supporting_event_refs),"supporting_fact_refs":list(self.supporting_fact_refs),"start_event_ref":self.start_event_ref,"end_event_ref":self.end_event_ref,"interpretation":self.interpretation.to_dict()}
    @classmethod
    def from_dict(cls,v): v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); return cls(**{**v,"interpretation":EvidenceBackedInterpretation.from_dict(v["interpretation"])})

@dataclass(frozen=True, slots=True)
class TurningPoint:
    turning_point_id: str; event_ref: str; supporting_fact_refs: tuple[str,...]; supporting_relationship_refs: tuple[str,...]; interpretation: EvidenceBackedInterpretation
    def __post_init__(self):
        _refs((self.turning_point_id,),"turn","turning_point_id"); _refs((self.event_ref,),"evt","event_ref")
        for n,ns in (("supporting_fact_refs","fact"),("supporting_relationship_refs","rel")): object.__setattr__(self,n,_refs(getattr(self,n),ns,n))
        if not isinstance(self.interpretation,EvidenceBackedInterpretation): raise StoryAnalysisModelError("interpretation invalid")
    def to_dict(self): return {"turning_point_id":self.turning_point_id,"event_ref":self.event_ref,"supporting_fact_refs":list(self.supporting_fact_refs),"supporting_relationship_refs":list(self.supporting_relationship_refs),"interpretation":self.interpretation.to_dict()}
    @classmethod
    def from_dict(cls,v): v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); return cls(**{**v,"interpretation":EvidenceBackedInterpretation.from_dict(v["interpretation"])})

@dataclass(frozen=True, slots=True)
class Reveal:
    reveal_id: str; reveal_event_refs: tuple[str,...]; supporting_fact_refs: tuple[str,...]; affected_character_refs: tuple[str,...]; setup_event_refs: tuple[str,...]; interpretation: EvidenceBackedInterpretation
    def __post_init__(self):
        _refs((self.reveal_id,),"reveal","reveal_id")
        for n,ns in (("reveal_event_refs","evt"),("supporting_fact_refs","fact"),("affected_character_refs","char"),("setup_event_refs","evt")): object.__setattr__(self,n,_refs(getattr(self,n),ns,n))
        if not isinstance(self.interpretation,EvidenceBackedInterpretation): raise StoryAnalysisModelError("interpretation invalid")
    def to_dict(self): return {"reveal_id":self.reveal_id,"reveal_event_refs":list(self.reveal_event_refs),"supporting_fact_refs":list(self.supporting_fact_refs),"affected_character_refs":list(self.affected_character_refs),"setup_event_refs":list(self.setup_event_refs),"interpretation":self.interpretation.to_dict()}
    @classmethod
    def from_dict(cls,v): v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); return cls(**{**v,"interpretation":EvidenceBackedInterpretation.from_dict(v["interpretation"])})

@dataclass(frozen=True, slots=True)
class ForeshadowPayoff:
    payoff_id: str; setup_event_refs: tuple[str,...]; payoff_event_refs: tuple[str,...]; interpretation: EvidenceBackedInterpretation
    def __post_init__(self):
        _refs((self.payoff_id,),"payoff","payoff_id")
        for n in ("setup_event_refs","payoff_event_refs"): object.__setattr__(self,n,_refs(getattr(self,n),"evt",n))
        if not isinstance(self.interpretation,EvidenceBackedInterpretation): raise StoryAnalysisModelError("interpretation invalid")
    def to_dict(self): return {"payoff_id":self.payoff_id,"setup_event_refs":list(self.setup_event_refs),"payoff_event_refs":list(self.payoff_event_refs),"interpretation":self.interpretation.to_dict()}
    @classmethod
    def from_dict(cls,v): v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); return cls(**{**v,"interpretation":EvidenceBackedInterpretation.from_dict(v["interpretation"])})

@dataclass(frozen=True, slots=True)
class ArcAnalysis:
    schema_version: int; arcs: tuple[StoryArc,...]; turning_points: tuple[TurningPoint,...]; reveals: tuple[Reveal,...]; foreshadow_payoffs: tuple[ForeshadowPayoff,...]
    def __post_init__(self):
        if self.schema_version != 1: raise StoryAnalysisModelError("unsupported schema_version")
        for n,t,key in (("arcs",StoryArc,"arc_id"),("turning_points",TurningPoint,"turning_point_id"),("reveals",Reveal,"reveal_id"),("foreshadow_payoffs",ForeshadowPayoff,"payoff_id")):
            v=tuple(getattr(self,n)); object.__setattr__(self,n,v)
            if any(not isinstance(x,t) for x in v) or len({getattr(x,key) for x in v}) != len(v): raise StoryAnalysisModelError(f"{n} invalid")
    def to_dict(self): return {"schema_version":self.schema_version,"arcs":[x.to_dict() for x in self.arcs],"turning_points":[x.to_dict() for x in self.turning_points],"reveals":[x.to_dict() for x in self.reveals],"foreshadow_payoffs":[x.to_dict() for x in self.foreshadow_payoffs]}
    @classmethod
    def from_dict(cls,v): v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); return cls(v["schema_version"],tuple(StoryArc.from_dict(x) for x in v["arcs"]),tuple(TurningPoint.from_dict(x) for x in v["turning_points"]),tuple(Reveal.from_dict(x) for x in v["reveals"]),tuple(ForeshadowPayoff.from_dict(x) for x in v["foreshadow_payoffs"]))

@dataclass(frozen=True, slots=True)
class GlobalSection:
    section_ordinal: int; label_zh: str; event_refs: tuple[str,...]; interpretation: EvidenceBackedInterpretation
    def __post_init__(self):
        _pos(self.section_ordinal,"section_ordinal"); _text(self.label_zh,"label_zh"); object.__setattr__(self,"event_refs",_refs(self.event_refs,"evt","event_refs"))
        if not isinstance(self.interpretation,EvidenceBackedInterpretation): raise StoryAnalysisModelError("interpretation invalid")
    def to_dict(self): return {"section_ordinal":self.section_ordinal,"label_zh":self.label_zh,"event_refs":list(self.event_refs),"interpretation":self.interpretation.to_dict()}
    @classmethod
    def from_dict(cls,v): v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); return cls(**{**v,"interpretation":EvidenceBackedInterpretation.from_dict(v["interpretation"])})

@dataclass(frozen=True, slots=True)
class GlobalStructure:
    schema_version: int; main_conflict: EvidenceBackedInterpretation; secondary_conflicts: tuple[EvidenceBackedInterpretation,...]; main_plot: EvidenceBackedInterpretation; subplots: tuple[EvidenceBackedInterpretation,...]; ending_state: EvidenceBackedInterpretation; global_sections: tuple[GlobalSection,...]; main_character_refs: tuple[str,...]; major_arc_refs: tuple[str,...]; major_turning_point_refs: tuple[str,...]; major_reveal_refs: tuple[str,...]
    def __post_init__(self):
        if self.schema_version != 1: raise StoryAnalysisModelError("unsupported schema_version")
        for n in ("main_conflict","main_plot","ending_state"):
            if not isinstance(getattr(self,n),EvidenceBackedInterpretation): raise StoryAnalysisModelError(f"{n} invalid")
        for n in ("secondary_conflicts","subplots"): object.__setattr__(self,n,_interp(getattr(self,n),n))
        s=tuple(self.global_sections); object.__setattr__(self,"global_sections",s)
        if any(not isinstance(x,GlobalSection) for x in s) or len({x.section_ordinal for x in s}) != len(s): raise StoryAnalysisModelError("global_sections invalid")
        for n,ns in (("main_character_refs","char"),("major_arc_refs","arc"),("major_turning_point_refs","turn"),("major_reveal_refs","reveal")): object.__setattr__(self,n,_refs(getattr(self,n),ns,n))
    def to_dict(self): return {"schema_version":self.schema_version,"main_conflict":self.main_conflict.to_dict(),"secondary_conflicts":[x.to_dict() for x in self.secondary_conflicts],"main_plot":self.main_plot.to_dict(),"subplots":[x.to_dict() for x in self.subplots],"ending_state":self.ending_state.to_dict(),"global_sections":[x.to_dict() for x in self.global_sections],"main_character_refs":list(self.main_character_refs),"major_arc_refs":list(self.major_arc_refs),"major_turning_point_refs":list(self.major_turning_point_refs),"major_reveal_refs":list(self.major_reveal_refs)}
    @classmethod
    def from_dict(cls,v):
        v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); return cls(**{**v,**{n:EvidenceBackedInterpretation.from_dict(v[n]) for n in ("main_conflict","main_plot","ending_state")},**{n:tuple(EvidenceBackedInterpretation.from_dict(x) for x in v[n]) for n in ("secondary_conflicts","subplots")},"global_sections":tuple(GlobalSection.from_dict(x) for x in v["global_sections"])})

@dataclass(frozen=True, slots=True)
class GlobalStoryBible:
    schema_version: int; premise_zh: EvidenceBackedInterpretation; synopsis_zh: EvidenceBackedInterpretation; genre: str; tone: str; setting: EvidenceBackedInterpretation; themes: tuple[EvidenceBackedInterpretation,...]; main_character_refs: tuple[str,...]; major_arc_refs: tuple[str,...]; major_turning_point_refs: tuple[str,...]; major_reveal_refs: tuple[str,...]; unresolved_summary: EvidenceBackedInterpretation; source_conflict_summary: EvidenceBackedInterpretation; coverage_metadata: tuple[tuple[str,int],...]
    def __post_init__(self):
        if self.schema_version != 1: raise StoryAnalysisModelError("unsupported schema_version")
        for n in ("premise_zh","synopsis_zh","setting","unresolved_summary","source_conflict_summary"):
            if not isinstance(getattr(self,n),EvidenceBackedInterpretation): raise StoryAnalysisModelError(f"{n} invalid")
        _text(self.genre,"genre"); _text(self.tone,"tone"); object.__setattr__(self,"themes",_interp(self.themes,"themes"))
        for n,ns in (("main_character_refs","char"),("major_arc_refs","arc"),("major_turning_point_refs","turn"),("major_reveal_refs","reveal")): object.__setattr__(self,n,_refs(getattr(self,n),ns,n))
        object.__setattr__(self,"coverage_metadata",_count_map(self.coverage_metadata,"coverage_metadata"))
    def to_dict(self): return {"schema_version":self.schema_version,"premise_zh":self.premise_zh.to_dict(),"synopsis_zh":self.synopsis_zh.to_dict(),"genre":self.genre,"tone":self.tone,"setting":self.setting.to_dict(),"themes":[x.to_dict() for x in self.themes],"main_character_refs":list(self.main_character_refs),"major_arc_refs":list(self.major_arc_refs),"major_turning_point_refs":list(self.major_turning_point_refs),"major_reveal_refs":list(self.major_reveal_refs),"unresolved_summary":self.unresolved_summary.to_dict(),"source_conflict_summary":self.source_conflict_summary.to_dict(),"coverage_metadata":dict(self.coverage_metadata)}
    @classmethod
    def from_dict(cls,v):
        v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); return cls(**{**v,**{n:EvidenceBackedInterpretation.from_dict(v[n]) for n in ("premise_zh","synopsis_zh","setting","unresolved_summary","source_conflict_summary")},"themes":tuple(EvidenceBackedInterpretation.from_dict(x) for x in v["themes"])})

@dataclass(frozen=True, slots=True)
class A6UpstreamIdentity:
    consolidation_manifest_ref: ArtifactRef
    def __post_init__(self):
        if not isinstance(self.consolidation_manifest_ref,ArtifactRef): raise StoryAnalysisModelError("consolidation_manifest_ref must be ArtifactRef")
    def to_dict(self): return {"consolidation_manifest_ref":self.consolidation_manifest_ref.to_dict()}
    @classmethod
    def from_dict(cls,v): v=_keys(v,{"consolidation_manifest_ref"},cls.__name__); return cls(ArtifactRef.from_dict(v["consolidation_manifest_ref"]))

@dataclass(frozen=True, slots=True)
class A6SemanticIdentity:
    story_analysis_profile_id: str; story_analysis_profile_hash: str; semantic_profile_id: str; semantic_profile_hash: str; prompt_identities: tuple[PromptAssetIdentity,...]; output_schema_identities: tuple[OutputSchemaAssetIdentity,...]; plan_hash: str; character_request_hashes: tuple[str,...]; plot_window_request_hashes: tuple[str,...]; global_skeleton_request_hashes: tuple[str,...]; story_bible_request_hashes: tuple[str,...]
    def __post_init__(self):
        _id(self.story_analysis_profile_id,"story_analysis_profile_id"); _sha(self.story_analysis_profile_hash,"story_analysis_profile_hash"); _id(self.semantic_profile_id,"semantic_profile_id"); _sha(self.semantic_profile_hash,"semantic_profile_hash"); _sha(self.plan_hash,"plan_hash")
        p=tuple(self.prompt_identities); o=tuple(self.output_schema_identities)
        if any(not isinstance(x,PromptAssetIdentity) for x in p) or any(not isinstance(x,OutputSchemaAssetIdentity) for x in o): raise StoryAnalysisModelError("invalid asset identities")
        object.__setattr__(self,"prompt_identities",p); object.__setattr__(self,"output_schema_identities",o)
        for n in ("character_request_hashes","plot_window_request_hashes","global_skeleton_request_hashes","story_bible_request_hashes"): object.__setattr__(self,n,tuple(_sha(x,n) for x in getattr(self,n)))
    def to_dict(self): return {"story_analysis_profile_id":self.story_analysis_profile_id,"story_analysis_profile_hash":self.story_analysis_profile_hash,"semantic_profile_id":self.semantic_profile_id,"semantic_profile_hash":self.semantic_profile_hash,"prompt_identities":[x.to_dict() for x in self.prompt_identities],"output_schema_identities":[x.to_dict() for x in self.output_schema_identities],"plan_hash":self.plan_hash,"character_request_hashes":list(self.character_request_hashes),"plot_window_request_hashes":list(self.plot_window_request_hashes),"global_skeleton_request_hashes":list(self.global_skeleton_request_hashes),"story_bible_request_hashes":list(self.story_bible_request_hashes)}
    @classmethod
    def from_dict(cls,v):
        v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); return cls(**{**v,"prompt_identities":tuple(PromptAssetIdentity.from_dict(x) for x in v["prompt_identities"]),"output_schema_identities":tuple(OutputSchemaAssetIdentity.from_dict(x) for x in v["output_schema_identities"])})

@dataclass(frozen=True, slots=True)
class StoryAnalysisCoverageSummary:
    canonical_character_count: int; character_analysis_count: int; canonical_event_count: int; owned_event_count: int; plot_window_count: int; global_event_importance_count: int; arc_count: int; turning_point_count: int; reveal_count: int; foreshadow_payoff_count: int
    def __post_init__(self):
        for n in self.__dataclass_fields__: _pos(getattr(self,n),n,zero=True)
    def to_dict(self): return {n:getattr(self,n) for n in self.__dataclass_fields__}
    @classmethod
    def from_dict(cls,v): return cls(**_keys(v,set(cls.__dataclass_fields__),cls.__name__))

@dataclass(frozen=True, slots=True)
class StoryAnalysisManifest:
    schema_version: int; project_id: str; document_id: str; consolidation_manifest_ref: ArtifactRef; character_analysis_ref: ArtifactRef; global_event_analysis_ref: ArtifactRef; arc_analysis_ref: ArtifactRef; global_structure_ref: ArtifactRef; global_story_bible_ref: ArtifactRef; semantic_identity: A6SemanticIdentity; upstream_identity: A6UpstreamIdentity; coverage_summary: StoryAnalysisCoverageSummary
    def __post_init__(self):
        if self.schema_version != 1: raise StoryAnalysisModelError("unsupported schema_version")
        _id(self.project_id,"project_id"); _id(self.document_id,"document_id")
        for n in ("consolidation_manifest_ref","character_analysis_ref","global_event_analysis_ref","arc_analysis_ref","global_structure_ref","global_story_bible_ref"):
            if not isinstance(getattr(self,n),ArtifactRef): raise StoryAnalysisModelError(f"{n} must be ArtifactRef")
        if not isinstance(self.semantic_identity,A6SemanticIdentity) or not isinstance(self.upstream_identity,A6UpstreamIdentity) or not isinstance(self.coverage_summary,StoryAnalysisCoverageSummary): raise StoryAnalysisModelError("invalid manifest nested contract")
        if self.consolidation_manifest_ref != self.upstream_identity.consolidation_manifest_ref: raise StoryAnalysisModelError("manifest upstream pin mismatch")
    def to_dict(self): return {"schema_version":self.schema_version,"project_id":self.project_id,"document_id":self.document_id,**{n:getattr(self,n).to_dict() for n in ("consolidation_manifest_ref","character_analysis_ref","global_event_analysis_ref","arc_analysis_ref","global_structure_ref","global_story_bible_ref")},"semantic_identity":self.semantic_identity.to_dict(),"upstream_identity":self.upstream_identity.to_dict(),"coverage_summary":self.coverage_summary.to_dict()}
    @classmethod
    def from_dict(cls,v):
        v=_keys(v,set(cls.__dataclass_fields__),cls.__name__); refs=("consolidation_manifest_ref","character_analysis_ref","global_event_analysis_ref","arc_analysis_ref","global_structure_ref","global_story_bible_ref")
        return cls(**{**v,**{n:ArtifactRef.from_dict(v[n]) for n in refs},"semantic_identity":A6SemanticIdentity.from_dict(v["semantic_identity"]),"upstream_identity":A6UpstreamIdentity.from_dict(v["upstream_identity"]),"coverage_summary":StoryAnalysisCoverageSummary.from_dict(v["coverage_summary"])})

__all__ = ["STORY_ANALYSIS_SCHEMA_VERSION","EVIDENCE_MODES","StoryAnalysisSemanticPass","StoryAnalysisPlanningPolicy","StoryAnalysisProfile","load_story_analysis_profile","EvidenceBackedInterpretation","CharacterAnalysis","CharacterAnalysisSet","PlotWindowAnalysis","GlobalEventImportance","GlobalEventAnalysis","StoryArc","TurningPoint","Reveal","ForeshadowPayoff","ArcAnalysis","GlobalSection","GlobalStructure","GlobalStoryBible","A6UpstreamIdentity","A6SemanticIdentity","StoryAnalysisCoverageSummary","StoryAnalysisManifest"]
