from pathlib import Path
from ..io import load_yaml
def build_concat_plan(path):
    t=load_yaml(path);clips=t.get("clips",[])
    if not clips:raise ValueError("timeline contains no clips")
    return {"episode_id":t["episode_id"],"clips":[x["path"] for x in clips]}
def render_concat_file(plan,out):
    Path(out).write_text("".join("file '"+str(p).replace("'", "'\''")+"'\n" for p in plan["clips"]),encoding="utf-8")
