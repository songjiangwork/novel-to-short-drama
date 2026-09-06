import hashlib
def deterministic_seed(project_id,shot_id,attempt):
    if attempt<1: raise ValueError("attempt must be >= 1")
    d=hashlib.sha256(f"{project_id}|{shot_id}|{attempt}".encode()).digest()
    return int.from_bytes(d[:8],"big") & ((1<<63)-1)
