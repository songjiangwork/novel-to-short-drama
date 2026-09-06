from jsonschema import Draft202012Validator
from .io import load_json,load_yaml
from .paths import SCHEMAS_DIR
MAP={"project":"project.schema.json","character":"character.schema.json","location":"location.schema.json","shot":"shot.schema.json","qc":"qc.schema.json"}
def validate_file(path,kind):
    data=load_yaml(path); schema=load_json(SCHEMAS_DIR/MAP[kind]); v=Draft202012Validator(schema)
    return [f"{'/'.join(map(str,e.absolute_path)) or '<root>'}: {e.message}" for e in sorted(v.iter_errors(data),key=lambda e:list(e.absolute_path))]
def validate_shot(path):
    errs=validate_file(path,"shot")
    if errs:return errs
    s=load_yaml(path); ids=[c["character_id"] for c in s["characters"]]
    if len(ids)!=len(set(ids)): errs.append("characters: duplicate character_id")
    for t in s.get("dialogue",[]):
        if t["speaker"] not in ids: errs.append(f"dialogue: speaker {t['speaker']} is not present")
    if s.get("dialogue") and s["camera"]["movement"]!="locked": errs.append("camera: v1 dialogue shots should be locked")
    if s["duration_seconds"]<=5:
        n=sum(sum('\u4e00'<=ch<='\u9fff' for ch in t['text']) for t in s.get('dialogue',[]))
        if n>20: errs.append(f"dialogue: {n} Chinese characters exceeds 20-char/5s guardrail")
    return errs
