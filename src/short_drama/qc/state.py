from ..validation import validate_file
from ..io import load_yaml
def evaluate_qc(path):
    e=validate_file(path,"qc")
    if e:return {"valid":False,"errors":e}
    q=load_yaml(path);d=q["decision"]
    state="APPROVED" if d=="approve" else ("RETRY_REQUIRED" if d=="retry" and q["attempt"]<3 else "MANUAL_INTERVENTION")
    return {"valid":True,"shot_id":q["shot_id"],"attempt":q["attempt"],"decision":d,"next_state":state}
