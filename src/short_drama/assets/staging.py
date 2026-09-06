from pathlib import Path
import shutil
def staged_name(asset_id,source_path): return f"{asset_id.replace('.', '__')}{Path(source_path).suffix.lower()}"
def build_stage_plan(refs,destination_root):
    out=[];root=Path(destination_root)
    for kind in ("images","audios"):
        for ref in refs.get(kind,[]): out.append({"asset_id":ref["asset_id"],"source":ref["source_path"],"destination":str(root/staged_name(ref["asset_id"],ref["source_path"]))})
    return out
def execute_stage_plan(plan):
    for x in plan:
        src=Path(x["source"]);dst=Path(x["destination"])
        if not src.exists(): raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
