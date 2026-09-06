from ..io import load_yaml,load_json
from ..paths import TEMPLATES_DIR
def plist(ids):
    xs=[f"<Picture {i}>" for i in ids]
    return xs[0] if len(xs)==1 else (f"{xs[0]} and {xs[1]}" if len(xs)==2 else ", ".join(xs[:-1])+f", and {xs[-1]}")
def build_prompt(shot_path,refs_path):
    s=load_yaml(shot_path);r=load_json(refs_path);tpl=(TEMPLATES_DIR/"h3_prompt_v1.txt").read_text(encoding="utf-8")
    groups={}
    for x in r["images"]:
        if x["owner_type"]=="character":groups.setdefault(x["owner_id"],[]).append(x["index"])
    amap={x["owner_id"]:x["index"] for x in r["audios"]}; scene=next(x for x in r["images"] if x["owner_type"]=="location")
    mapping=[]
    for c in s["characters"]:
        cid=c["character_id"];mapping.append(f"- {cid} is the same person shown in {plist(groups[cid])}. Use those references only for {cid}'s identity, face, hairstyle, persistent attributes, wardrobe, and body proportions.")
    mapping.append(f"- <Picture {scene['index']}> is the canonical environment for {s['location']['location_id']} / {s['location']['coverage_view']}.")
    for sp,i in amap.items():mapping.append(f"- <Audio {i}> is {sp}'s VOICE IDENTITY ONLY.")
    shot=f"{s['camera']['framing']} shot at {s['location']['location_id']} using canonical coverage '{s['location']['coverage_view']}'. "+" ".join(f"{c['character_id']} is {c['screen_position']}." for c in s["characters"])
    presence="\n".join(f"- {c['character_id']} must be visible FROM THE FIRST FRAME and remain visible throughout the entire clip." for c in s["characters"] if c["presence"]["first_frame"] and c["presence"]["entire_shot"]) or "No special presence constraints."
    actions="\n".join(f"- {c['character_id']}: {c['action']}" for c in s["characters"])
    camera="Locked static camera. No pan, no dolly, no orbit, no zoom, no reframing, no cut, and no camera-angle transition." if s["camera"]["movement"]=="locked" else "Use only the simple camera movement explicitly described by the shot."
    turns=sorted(s.get("dialogue",[]),key=lambda x:x["order"]); dl=[]
    for i,t in enumerate(turns):
        sp=t["speaker"]; ord="FIRST" if i==0 else "SECOND"
        dl.append(f"{i+1}. {sp} speaks {ord}, using the voice identity from <Audio {amap[sp]}>:\n<d>[{t['language']}]{t['text']}</d>\nWhile {sp} speaks, only {sp}'s mouth should articulate speech.")
        if i<len(turns)-1:dl.append(f"{sp} must then STOP COMPLETELY. No overlap with the next speaker.")
    constraints=["Keep character identities strictly separated.","Do not merge faces or swap identities.","Do not swap wardrobes or persistent attributes.","Keep the canonical location identity stable."]
    if s["constraints"].get("no_speech_overlap"):constraints.append("No simultaneous dialogue.")
    if s["constraints"].get("stable_scene"):constraints.append("No major camera drift or location change.")
    repl={"{{REFERENCE_MAPPING}}":"\n".join(mapping),"{{SHOT_DESCRIPTION}}":shot,"{{PRESENCE_REQUIREMENTS}}":presence,"{{ACTION_DESCRIPTION}}":actions,"{{CAMERA_DESCRIPTION}}":camera,"{{DIALOGUE_BLOCK}}":"\n\n".join(dl) if dl else "No spoken dialogue. Do not generate speech.","{{OUTPUT_LANGUAGE}}":turns[0]["language"] if turns else "target-language","{{CONSTRAINT_BLOCK}}":"\n".join("- "+x for x in constraints)}
    for k,v in repl.items():tpl=tpl.replace(k,v)
    return tpl.strip()+"\n"
