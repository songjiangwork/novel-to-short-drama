from __future__ import annotations
import json
from pathlib import Path
import yaml

def load_yaml(path):
    with open(path, encoding="utf-8") as f: return yaml.safe_load(f)
def load_json(path):
    with open(path, encoding="utf-8") as f: return json.load(f)
def dump_json(data,path):
    with open(path,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2); f.write("\n")
