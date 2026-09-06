import argparse,json
from pathlib import Path
from .assets.selector import resolve_references
from .assembly.concat import build_concat_plan,render_concat_file
from .io import dump_json,load_json
from .prompting.builder import build_prompt
from .qc.state import evaluate_qc
from .validation import validate_file,validate_shot
from .workflows.request import build_generation_request
from .workflows.retry import build_retry
def out(x):print(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 p=argparse.ArgumentParser(prog="short-drama");s=p.add_subparsers(dest="cmd",required=True)
 a=s.add_parser("validate");a.add_argument("path");a.add_argument("--kind",choices=["project","character","location","shot","qc"])
 a=s.add_parser("select-refs");a.add_argument("shot");a.add_argument("assets");a.add_argument("characters_dir");a.add_argument("locations_dir");a.add_argument("-o","--output",default="resolved_refs.json")
 a=s.add_parser("build-prompt");a.add_argument("shot");a.add_argument("refs");a.add_argument("-o","--output",default="prompt.txt")
 a=s.add_parser("build-request");a.add_argument("shot");a.add_argument("refs");a.add_argument("--project-id",required=True);a.add_argument("--attempt",type=int,default=1);a.add_argument("--prompt");a.add_argument("-o","--output",default="request.json")
 a=s.add_parser("record-qc");a.add_argument("qc")
 a=s.add_parser("retry");a.add_argument("request");a.add_argument("-o","--output",default="retry_request.json")
 a=s.add_parser("assemble");a.add_argument("timeline");a.add_argument("--concat-file",default="concat.txt")
 x=p.parse_args()
 if x.cmd=="validate":
  e=validate_shot(x.path) if x.kind=="shot" else validate_file(x.path,x.kind);out({"valid":not e,"errors":e});return 0 if not e else 2
 if x.cmd=="select-refs":
  q=resolve_references(x.shot,x.assets,x.characters_dir,x.locations_dir);dump_json(q,x.output);out({"output":x.output,"images":len(q["images"]),"audios":len(q["audios"])});return 0
 if x.cmd=="build-prompt":
  q=build_prompt(x.shot,x.refs);Path(x.output).write_text(q,encoding="utf-8");out({"output":x.output});return 0
 if x.cmd=="build-request":
  q=Path(x.prompt).read_text(encoding="utf-8") if x.prompt else build_prompt(x.shot,x.refs);req=build_generation_request(x.shot,x.refs,x.project_id,x.attempt,q);dump_json(req,x.output);out({"output":x.output,"seed":req["seed"],"attempt":req["attempt"]});return 0
 if x.cmd=="record-qc":out(evaluate_qc(x.qc));return 0
 if x.cmd=="retry":q=build_retry(load_json(x.request));dump_json(q,x.output);out({"output":x.output,"state":q.get("state","READY")});return 0
 if x.cmd=="assemble":q=build_concat_plan(x.timeline);render_concat_file(q,x.concat_file);out({"concat_file":x.concat_file,"clips":len(q["clips"])});return 0
 return 1
