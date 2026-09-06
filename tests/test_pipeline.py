from pathlib import Path
import json
from short_drama.assets.selector import resolve_references
from short_drama.prompting.builder import build_prompt
from short_drama.seeds import deterministic_seed
from short_drama.validation import validate_shot
from short_drama.workflows.request import build_generation_request
from short_drama.workflows.retry import build_retry
F=Path(__file__).parent/"fixtures"
def refs(tmp):
 q=resolve_references(str(F/"shot.yaml"),str(F/"assets.yaml"),str(F/"characters"),str(F/"locations"));p=tmp/"refs.json";p.write_text(json.dumps(q,ensure_ascii=False),encoding="utf-8");return q,p
def test_valid():assert validate_shot(str(F/"shot.yaml"))==[]
def test_refs(tmp_path):
 q,p=refs(tmp_path);assert [x["owner_id"] for x in q["images"]]==["song","song","song","linsey","linsey","linsey","classroom"];assert [x["index"] for x in q["images"]]==list(range(1,8));assert [x["owner_id"] for x in q["audios"]]==["song","linsey"]
def test_prompt(tmp_path):
 q,p=refs(tmp_path);s=build_prompt(str(F/"shot.yaml"),str(p));assert all(x in s for x in ["<Picture 1>","<Picture 7>","<Audio 1>","<Audio 2>","STOP COMPLETELY"])
def test_seed():assert deterministic_seed("p","s",1)==deterministic_seed("p","s",1) and deterministic_seed("p","s",1)!=deterministic_seed("p","s",2)
def test_retry_seed_only(tmp_path):
 q,p=refs(tmp_path);prompt=build_prompt(str(F/"shot.yaml"),str(p));a=build_generation_request(str(F/"shot.yaml"),str(p),"classroom-demo",1,prompt);b=build_retry(a);assert b["attempt"]==2 and b["seed"]!=a["seed"]
 for k in ["profile","prompt_template_version","references","prompt","control_after_generate","runtime_patch_whitelist","adapter_status","project_id","shot_id","schema_version"]:assert b[k]==a[k]
def test_attempt3():assert build_retry({"project_id":"p","shot_id":"s","attempt":3})["state"]=="MANUAL_INTERVENTION"
