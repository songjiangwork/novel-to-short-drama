from pathlib import Path
from ..io import load_yaml
def asset_map(path): return load_yaml(path).get("assets",{})
def load_characters(directory):
    out={}
    for p in sorted(Path(directory).glob("*.yaml")):
        d=load_yaml(p);out[d["character_id"]]=d
    return out
def load_locations(directory):
    out={}
    for p in sorted(Path(directory).glob("*.yaml")):
        d=load_yaml(p);out[d["location_id"]]=d
    return out
