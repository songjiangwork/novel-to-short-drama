from ..io import load_yaml
from .registry import asset_map,load_characters,load_locations
ROLE_ORDER={"closeup":["face_front","face_three_quarter"],"medium":["face_front","face_three_quarter","full_body"],"full_body":["face_front","face_three_quarter","full_body"]}
def resolve_references(shot_path,assets_path,characters_dir,locations_dir):
    shot=load_yaml(shot_path); assets=asset_map(assets_path); chars=load_characters(characters_dir); locs=load_locations(locations_dir)
    def req(aid):
        if aid not in assets: raise ValueError(f"Missing asset registry entry: {aid}")
        a=assets[aid]
        if not a.get("enabled",True): raise ValueError(f"Asset disabled: {aid}")
        return a
    images=[];idx=1
    for sc in shot["characters"]:
        cid=sc["character_id"]; look=sc["look_id"]
        if cid not in chars: raise ValueError(f"Unknown character: {cid}")
        la=chars[cid]["looks"][look]["assets"]
        for role in ROLE_ORDER[shot["camera"]["framing"]]:
            if role not in la: raise ValueError(f"Missing required role '{role}' for {cid}/{look}")
            aid=la[role]; a=req(aid); images.append({"index":idx,"asset_id":aid,"owner_type":"character","owner_id":cid,"role":role,"source_path":a["source_path"]}); idx+=1
    lid=shot["location"]["location_id"]; view=shot["location"]["coverage_view"]
    if lid not in locs: raise ValueError(f"Unknown location: {lid}")
    if view not in locs[lid]["coverage_views"]: raise ValueError(f"Missing canonical coverage '{view}' for '{lid}'")
    aid=locs[lid]["coverage_views"][view]["asset"]; a=req(aid); images.append({"index":idx,"asset_id":aid,"owner_type":"location","owner_id":lid,"role":"scene","coverage_view":view,"source_path":a["source_path"]})
    speakers=[];langs={}
    for t in sorted(shot.get("dialogue",[]),key=lambda x:x["order"]):
        langs.setdefault(t["speaker"],t["language"]);
        if t["speaker"] not in speakers:speakers.append(t["speaker"])
    audios=[]
    for i,sp in enumerate(speakers,1):
        lang=langs[sp]; aid=chars[sp]["voices"][lang]["asset"]; a=req(aid); audios.append({"index":i,"asset_id":aid,"owner_type":"character","owner_id":sp,"role":"voice","language":lang,"source_path":a["source_path"]})
    if len(images)>9: raise ValueError(f"H3 image reference limit exceeded: {len(images)} > 9")
    if len(audios)>2: raise ValueError(f"v1 audio reference limit exceeded: {len(audios)} > 2")
    return {"schema_version":1,"shot_id":shot["shot_id"],"images":images,"audios":audios}
