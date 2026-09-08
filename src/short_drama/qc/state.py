from ..io import dump_json, load_json, load_yaml
from ..validation import validate_file


def evaluate_qc(path):
    errors = validate_file(path, "qc")
    if errors:
        return {"valid": False, "errors": errors}

    qc = load_yaml(path)
    decision = qc["decision"]
    failed_checks = sorted(
        name for name, value in qc["checks"].items() if value == "fail"
    )

    semantic_errors = []
    if decision == "approve" and failed_checks:
        semantic_errors.append(
            "decision 'approve' is inconsistent with failed checks: "
            + ", ".join(failed_checks)
        )
    if decision in {"retry", "reject"} and not failed_checks:
        semantic_errors.append(
            f"decision '{decision}' requires at least one failed check"
        )
    if semantic_errors:
        return {"valid": False, "errors": semantic_errors}

    if decision == "approve":
        state = "APPROVED"
    elif decision == "retry" and qc["attempt"] < 3:
        state = "RETRY_REQUIRED"
    else:
        state = "MANUAL_INTERVENTION"

    return {
        "valid": True,
        "project_id": qc["project_id"],
        "shot_id": qc["shot_id"],
        "attempt": qc["attempt"],
        "decision": decision,
        "failed_checks": failed_checks,
        "failure_reasons": qc.get("failure_reasons", []),
        "next_state": state,
    }


def record_qc_result(qc_path, result_path):
    evaluation = evaluate_qc(qc_path)
    if not evaluation["valid"]:
        return evaluation

    result = load_json(result_path)
    errors = []
    if result.get("project_id") != evaluation["project_id"]:
        errors.append(
            f"project_id mismatch: result={result.get('project_id')} qc={evaluation['project_id']}"
        )
    if result.get("shot_id") != evaluation["shot_id"]:
        errors.append(
            f"shot_id mismatch: result={result.get('shot_id')} qc={evaluation['shot_id']}"
        )
    if result.get("attempt") != evaluation["attempt"]:
        errors.append(
            f"attempt mismatch: result={result.get('attempt')} qc={evaluation['attempt']}"
        )
    if result.get("state") != "QC_PENDING":
        errors.append(
            f"result state must be QC_PENDING before recording QC; got {result.get('state')}"
        )
    if errors:
        return {"valid": False, "errors": errors}

    qc = load_yaml(qc_path)
    result["state"] = evaluation["next_state"]
    result["qc"] = {
        "decision": qc["decision"],
        "checks": qc["checks"],
        "failure_reasons": qc.get("failure_reasons", []),
        "notes": qc.get("notes"),
    }
    dump_json(result, result_path)

    return {
        **evaluation,
        "result": str(result_path),
        "state": result["state"],
    }
