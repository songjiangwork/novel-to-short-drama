from pathlib import Path
import json
from short_drama.assets.selector import resolve_references
from short_drama.prompting.builder import build_prompt
from short_drama.seeds import deterministic_seed
from short_drama.validation import validate_shot
from short_drama.workflows.request import build_generation_request
from short_drama.qc.state import evaluate_qc, record_qc_result
from short_drama.workflows.retry import build_gated_retry, build_retry
F=Path(__file__).parent/"fixtures"
def refs(tmp):
 q=resolve_references(str(F/"shot.yaml"),str(F/"assets.yaml"),str(F/"characters"),str(F/"locations"));p=tmp/"refs.json";p.write_text(json.dumps(q,ensure_ascii=False),encoding="utf-8");return q,p
def test_valid():assert validate_shot(str(F/"shot.yaml"))==[]
def test_refs(tmp_path):
 q,p=refs(tmp_path);assert [x["owner_id"] for x in q["images"]]==["song","song","song","linsey","linsey","linsey","classroom"];assert [x["index"] for x in q["images"]]==list(range(1,8));assert [x["owner_id"] for x in q["audios"]]==["song","linsey"]
def test_prompt(tmp_path):
 q,p=refs(tmp_path);s=build_prompt(str(F/"shot.yaml"),str(p))
 assert all(x in s for x in [
  "<Picture 1>","<Picture 7>","<Audio 1>","<Audio 2>",
  "A natural medium two-shot in the classroom from <Picture 7>",
  "must control the scene/background",
  "2. Song must then STOP COMPLETELY. No overlap.",
  "3. Linsey speaks SECOND",
  "<d>[Chinese]你觉得这样可以吗？</d>",
  "Linsey listens silently with a closed/resting mouth",
  "Natural Mandarin speech"
 ])
 assert all(x not in s for x in [
  "MUST CONTROL THE SCENE/BACKGROUND",
  "Do not reproduce walls, furniture, floors",
  "canonical coverage",
  "CHARACTER PRESENCE:",
  "ACTION:",
  "CONTINUITY / NEGATIVE CONSTRAINTS:"
 ])
def test_seed():assert deterministic_seed("p","s",1)==deterministic_seed("p","s",1) and deterministic_seed("p","s",1)!=deterministic_seed("p","s",2)
def test_retry_seed_only(tmp_path):
 q,p=refs(tmp_path);prompt=build_prompt(str(F/"shot.yaml"),str(p));a=build_generation_request(str(F/"shot.yaml"),str(p),"classroom-demo",1,prompt);b=build_retry(a);assert b["attempt"]==2 and b["seed"]!=a["seed"]
 for k in ["profile","prompt_template_version","references","prompt","control_after_generate","runtime_patch_whitelist","adapter_status","project_id","shot_id","schema_version"]:assert b[k]==a[k]
def test_attempt3():assert build_retry({"project_id":"p","shot_id":"s","attempt":3})["state"]=="MANUAL_INTERVENTION"



def _qc_file(tmp_path, decision="retry", attempt=1, scene="fail"):
 p=tmp_path/"qc.yaml"
 p.write_text(f"""schema_version: 1
project_id: classroom-demo
shot_id: E01_S003_SH002
attempt: {attempt}
decision: {decision}
checks:
  character_presence: pass
  identity_consistency: pass
  wardrobe_consistency: pass
  scene_identity: {scene}
  dialogue_order: pass
  speaker_attribution: pass
  speech_overlap: pass
  lip_sync: pass
  camera_stability: pass
failure_reasons:
  - wrong_location
notes: test
""",encoding="utf-8")
 return p


def _result_file(tmp_path, state="QC_PENDING", attempt=1):
 p=tmp_path/"result.json"
 p.write_text(json.dumps({
  "schema_version":1,
  "project_id":"classroom-demo",
  "shot_id":"E01_S003_SH002",
  "attempt":attempt,
  "state":state,
  "outputs":[]
 }),encoding="utf-8")
 return p


def test_qc_rejects_contradictory_approve(tmp_path):
 p=_qc_file(tmp_path,decision="approve",scene="fail")
 q=evaluate_qc(str(p))
 assert not q["valid"] and "inconsistent" in q["errors"][0]


def test_record_qc_persists_retry_required(tmp_path):
 qc=_qc_file(tmp_path)
 result=_result_file(tmp_path)
 q=record_qc_result(str(qc),str(result))
 saved=json.loads(result.read_text(encoding="utf-8"))
 assert q["valid"] and q["state"]=="RETRY_REQUIRED"
 assert saved["state"]=="RETRY_REQUIRED"
 assert saved["qc"]["checks"]["scene_identity"]=="fail"


def test_retry_gate_requires_retry_required():
 request={"project_id":"classroom-demo","shot_id":"E01_S003_SH002","attempt":1}
 blocked=build_gated_retry(request,{"project_id":"classroom-demo","shot_id":"E01_S003_SH002","attempt":1,"state":"APPROVED"})
 assert not blocked["valid"]


def test_retry_gate_seed_only():
 request={"project_id":"classroom-demo","shot_id":"E01_S003_SH002","attempt":1,"seed":11,"prompt":"same","references":{"images":[],"audios":[]}}
 result={"project_id":"classroom-demo","shot_id":"E01_S003_SH002","attempt":1,"state":"RETRY_REQUIRED"}
 gated=build_gated_retry(request,result)
 retry=gated["request"]
 assert gated["valid"] and retry["attempt"]==2 and retry["seed"]!=request["seed"]
 assert retry["prompt"]=="same" and retry["references"]==request["references"]
 assert retry["retry_policy"]=="seed_only"


def test_record_qc_attempt_mismatch_is_rejected(tmp_path):
 qc=_qc_file(tmp_path,attempt=2)
 result=_result_file(tmp_path,attempt=1)
 q=record_qc_result(str(qc),str(result))
 assert not q["valid"] and "attempt mismatch" in q["errors"][0]


def test_retry_gate_rejects_project_mismatch():
 request={"project_id":"project-a","shot_id":"E01_S003_SH002","attempt":1}
 result={"project_id":"project-b","shot_id":"E01_S003_SH002","attempt":1,"state":"RETRY_REQUIRED"}
 gated=build_gated_retry(request,result)
 assert not gated["valid"] and "project_id mismatch" in gated["errors"][0]
