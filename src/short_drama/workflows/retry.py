from copy import deepcopy

from ..seeds import deterministic_seed


def build_retry(request):
    attempt = int(request["attempt"])
    if attempt >= 3:
        return {
            "shot_id": request["shot_id"],
            "state": "MANUAL_INTERVENTION",
            "reason": "max_attempts_reached",
        }

    next_attempt = attempt + 1
    retry = deepcopy(request)
    retry["attempt"] = next_attempt
    retry["seed"] = deterministic_seed(
        request["project_id"], request["shot_id"], next_attempt
    )
    retry["output_prefix"] = f"{request['shot_id']}_attempt_{next_attempt:03d}"
    retry["retry_policy"] = "seed_only"
    return retry


def build_gated_retry(request, result):
    errors = []
    if result.get("project_id") != request.get("project_id"):
        errors.append(
            f"project_id mismatch: request={request.get('project_id')} result={result.get('project_id')}"
        )
    if result.get("shot_id") != request.get("shot_id"):
        errors.append(
            f"shot_id mismatch: request={request.get('shot_id')} result={result.get('shot_id')}"
        )
    if result.get("attempt") != request.get("attempt"):
        errors.append(
            f"attempt mismatch: request={request.get('attempt')} result={result.get('attempt')}"
        )
    if result.get("state") != "RETRY_REQUIRED":
        errors.append(
            f"retry requires result state RETRY_REQUIRED; got {result.get('state')}"
        )
    if errors:
        return {"valid": False, "errors": errors}

    retry = build_retry(request)
    if retry.get("state") == "MANUAL_INTERVENTION":
        return {"valid": False, "errors": ["max attempts reached; retry is not permitted"]}

    return {"valid": True, "request": retry}
