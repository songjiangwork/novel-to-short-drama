from copy import deepcopy
from ..seeds import deterministic_seed
def build_retry(request):
    a=int(request["attempt"])
    if a>=3:return {"shot_id":request["shot_id"],"state":"MANUAL_INTERVENTION","reason":"max_attempts_reached"}
    n=a+1;r=deepcopy(request);r["attempt"]=n;r["seed"]=deterministic_seed(request["project_id"],request["shot_id"],n);r["output_prefix"]=f"{request['shot_id']}_attempt_{n:03d}";r["retry_policy"]="seed_only";return r
