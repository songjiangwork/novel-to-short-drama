import argparse, json
from pathlib import Path
from .assets.selector import resolve_references
from .assets.staging import stage_references
from .assembly.concat import build_concat_plan, render_concat_file
from .comfyui.adapter import generate_with_comfyui
from .comfyui.api_workflow import REQUIRED_NODE_CLASSES, build_api_workflow
from .comfyui.client import ComfyUIClient
from .io import dump_json, load_json, load_yaml
from .prompting.builder import build_prompt
from .qc.state import record_qc_result
from .validation import validate_file, validate_shot
from .workflows.request import build_generation_request
from .workflows.retry import build_gated_retry

def out(x): print(json.dumps(x, ensure_ascii=False, indent=2))

def main():
    p=argparse.ArgumentParser(prog="short-drama"); s=p.add_subparsers(dest="cmd",required=True)
    a=s.add_parser("validate"); a.add_argument("path"); a.add_argument("--kind",choices=["project","character","location","shot","qc"])
    a=s.add_parser("select-refs"); a.add_argument("shot"); a.add_argument("assets"); a.add_argument("characters_dir"); a.add_argument("locations_dir"); a.add_argument("-o","--output",default="resolved_refs.json")
    a=s.add_parser("build-prompt"); a.add_argument("shot"); a.add_argument("refs"); a.add_argument("-o","--output",default="prompt.txt")
    a=s.add_parser("build-request"); a.add_argument("shot"); a.add_argument("refs"); a.add_argument("--project-id",required=True); a.add_argument("--attempt",type=int,default=1); a.add_argument("--prompt"); a.add_argument("-o","--output",default="request.json")
    a=s.add_parser("build-api-workflow"); a.add_argument("request"); a.add_argument("--input-dir",required=True); a.add_argument("-o","--output",default="workflow_api.json")
    a=s.add_parser("comfyui-check"); a.add_argument("runtime_config")
    a=s.add_parser("generate"); a.add_argument("request"); a.add_argument("runtime_config"); a.add_argument("--result",default="result.json"); a.add_argument("--workflow",required=True)
    a=s.add_parser("record-qc"); a.add_argument("qc"); a.add_argument("--result",required=True)
    a=s.add_parser("retry"); a.add_argument("request"); a.add_argument("--result",required=True); a.add_argument("-o","--output",default="retry_request.json")
    a=s.add_parser("assemble"); a.add_argument("timeline"); a.add_argument("--concat-file",default="concat.txt")
    x=p.parse_args()
    if x.cmd=="validate":
        e=validate_shot(x.path) if x.kind=="shot" else validate_file(x.path,x.kind); out({"valid":not e,"errors":e}); return 0 if not e else 2
    if x.cmd=="select-refs":
        q=resolve_references(x.shot,x.assets,x.characters_dir,x.locations_dir); dump_json(q,x.output); out({"output":x.output,"images":len(q["images"]),"audios":len(q["audios"])}); return 0
    if x.cmd=="build-prompt":
        q=build_prompt(x.shot,x.refs); Path(x.output).write_text(q,encoding="utf-8"); out({"output":x.output}); return 0
    if x.cmd=="build-request":
        q=Path(x.prompt).read_text(encoding="utf-8") if x.prompt else build_prompt(x.shot,x.refs); req=build_generation_request(x.shot,x.refs,x.project_id,x.attempt,q); dump_json(req,x.output); out({"output":x.output,"seed":req["seed"],"attempt":req["attempt"]}); return 0
    if x.cmd=="build-api-workflow":
        req=load_json(x.request); req["references"]=stage_references(req["references"],x.input_dir,req["project_id"],req["shot_id"]); q=build_api_workflow(req); dump_json(q,x.output); out({"output":x.output,"nodes":len(q)}); return 0
    if x.cmd=="comfyui-check":
        cfg=load_yaml(x.runtime_config); c=ComfyUIClient(cfg["server_url"]); missing=c.check_nodes(REQUIRED_NODE_CLASSES); out({"server_url":cfg["server_url"],"ok":not missing,"missing_nodes":missing}); return 0 if not missing else 3
    if x.cmd=="generate":
        q=generate_with_comfyui(load_json(x.request),x.runtime_config,load_json(x.workflow),x.result); out(q); return 0
    if x.cmd=="record-qc":
        q=record_qc_result(x.qc,x.result); out(q); return 0 if q.get("valid") else 2
    if x.cmd=="retry":
        gated=build_gated_retry(load_json(x.request),load_json(x.result))
        if not gated["valid"]: out(gated); return 2
        q=gated["request"]; dump_json(q,x.output); out({"output":x.output,"state":"READY","attempt":q["attempt"],"seed":q["seed"]}); return 0
    if x.cmd=="assemble": q=build_concat_plan(x.timeline); render_concat_file(q,x.concat_file); out({"concat_file":x.concat_file,"clips":len(q["clips"])}); return 0
    return 1
