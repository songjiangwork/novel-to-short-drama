#!/usr/bin/env python3
"""B1.3 Alice tokenizer & runtime-budget preflight (read-only; PR-A only).

Replaces the ``Budget UNKNOWN`` assessment with verifiable per-window capacity
evidence *without* invoking semantic generation.  It discovers the served
llama.cpp model identity and **verified** per-slot context via read-only
endpoints (``/v1/models``, ``/props``, ``/slots``), renders each production
request with the server's *authoritative* chat template (``POST /apply-template``),
counts the rendered prompt through ``POST /tokenize`` (``parse_special`` so
template special tokens are honored), builds the production B1.3 window semantic
preparations for the exact pinned B1.2 strategy (initial + cache-disabled
regeneration), and reports, per window, the production repository-estimator
budget, the tokenizer-measured prompt capacity, and a final operator verdict.

Fail-closed: the overall verdict is ``PASS`` only when capacity is fully
*trusted* -- the served model matches ``request_model`` (exact identity, no
first-entry shortcut), the discovery client is the same server as
``RuntimeConfig.base_url``, and per-slot ``n_ctx`` is verified (valid positive
integers, unique identities when present, count consistent with declared
``total_slots``, min across slots).  An operator ``model_context_tokens``
override may only *lower* the verified capacity and can never establish a
trusted PASS on its own.  Any capability that cannot be verified is ``UNKNOWN``
-- never a fabricated ``PASS``.  The preflight never calls a completion endpoint,
never writes production artifacts or mutates CURRENT pointers, and never logs a
raw prompt.

Exit codes: 0=PASS, 2=FAIL, 3=UNKNOWN, 1=unexpected diagnostic execution error.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from short_drama.adaptation import (
    B13WindowSemanticBudget, B13WindowSemanticBudgetError,
    build_b13_window_semantic_preparation, prepare_b13_disposition_windows,
)
from short_drama.adaptation.strategy_execution import B12StrategyArtifact
from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.llm import GenerationExecutionOptions, RuntimeConfig
from short_drama.paths import REPO_ROOT

# Exact B1.2 strategy authority for the real Alice run (pinned, not CURRENT).
B12_STRATEGY_ARTIFACT_TYPE = "adaptation_strategy"
B12_STRATEGY_ARTIFACT_ID = "a3e-real-novel.alice-b12-audit.b12.strategy"
B12_STRATEGY_REVISION = 1
B12_STRATEGY_CONTENT_HASH = "16c77dd60106ac60846c24660a4eee588caf2b9c48d44c96090d7a0af51918f5"
ADAPTATION_TARGET_ID = "alice-b12-audit"

# Observed runtime defaults (declarations only; never the authority itself).
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_REQUEST_MODEL = "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:IQ4_XS"
DEFAULT_PROVIDER_FAMILY = "qwen"
DEFAULT_TRANSPORT_ID = "llm-local"
DEFAULT_STORY_ROOT = REPO_ROOT / "runs" / "a4e_real_novel" / "a3e-real-novel" / "story"

# Exit codes (documented): PASS/FAIL/UNKNOWN/unexpected error.  A downstream runner
# must never interpret UNKNOWN (3) as authorization to generate.
EXIT_PASS, EXIT_FAIL, EXIT_UNKNOWN, EXIT_ERROR = 0, 2, 3, 1
_EXIT_BY_VERDICT = {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL, "UNKNOWN": EXIT_UNKNOWN}


def _exit_code(report) -> int:
    """Map a preflight report to its documented process exit code."""
    return _EXIT_BY_VERDICT.get((report or {}).get("overall_verdict"), EXIT_ERROR)

_GENERATION_PATHS = frozenset({
    "v1/chat/completions", "chat/completions", "v1/completions", "completions",
    "completion", "responses", "generate",
})


class LlamaHTTPError(Exception):
    """A read-only llama.cpp request failed (bounded, secret-safe detail)."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def _server_root(base_url: str) -> str:
    """Reduce a ``.../v1`` LLM base URL to the llama.cpp server root.

    The read-only discovery endpoints (``/props``, ``/slots``, ``/tokenize``,
    ``/apply-template``) and ``/v1/models`` are addressed from the server root,
    not the ``/v1`` chat-completions prefix.
    """
    parts = urlsplit(base_url)
    return f"{parts.scheme}://{parts.netloc}"


def _bounded_detail(body: bytes) -> str:
    return " ".join(body.decode("utf-8", errors="replace").split())[:200]


def _urllib_request(method, url, body, timeout, max_bytes) -> tuple[int, bytes]:
    headers = {"Content-Type": "application/json"} if body is not None else {}
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(max_bytes)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(max_bytes)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise LlamaHTTPError(f"timeout for {url}", 0) from exc
        raise LlamaHTTPError(f"connection failure for {url}: {exc.reason!r}", 0) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise LlamaHTTPError(f"timeout for {url}", 0) from exc


@dataclass(frozen=True)
class LlamaClient:
    """Bounded, read-only llama.cpp HTTP client over an injectable transport."""

    base_url: str
    transport: Callable[[str, str, bytes | None, float, int], tuple[int, bytes]] = _urllib_request
    timeout_seconds: float = 10.0
    max_response_bytes: int = 4_000_000
    requested_paths: list[str] = field(default_factory=list, repr=False)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> Any:
        url = self.base_url.rstrip("/") + path
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None
        self.requested_paths.append(path)
        status, raw = self.transport(method, url, body, self.timeout_seconds, self.max_response_bytes)
        if not 200 <= status < 300:
            raise LlamaHTTPError(f"HTTP {status} for {path}: {_bounded_detail(raw)}", status)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise LlamaHTTPError(f"malformed JSON for {path}: {exc}", status) from exc

    def get(self, path: str) -> Any:
        return self._request("GET", path, None)

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request("POST", path, payload)

    def apply_template(self, messages: Any, *, add_generation_prompt: bool = True,
                       parse_special: bool = True) -> str:
        """Render ``messages`` with the server's *real* chat template (non-generating)."""
        data = self.post("/apply-template", {
            "messages": list(messages),
            "add_generation_prompt": add_generation_prompt,
            "parse_special": parse_special,
        })
        if not isinstance(data, dict) or not isinstance(data.get("prompt"), str):
            raise LlamaHTTPError("apply-template response did not contain a 'prompt' string", 200)
        return data["prompt"]

    def tokenize(self, content: str, *, parse_special: bool = True) -> list:
        """Tokenize ``content`` on the server; ``parse_special`` honors special tokens."""
        data = self.post("/tokenize", {"content": content, "parse_special": parse_special})
        if not isinstance(data, dict) or not isinstance(data.get("tokens"), list):
            raise LlamaHTTPError("tokenize response did not contain a 'tokens' list", 200)
        return data["tokens"]

    @property
    def generation_calls(self) -> list[str]:
        return [p for p in self.requested_paths if p.lstrip("/").rstrip("/") in _GENERATION_PATHS]


@dataclass(frozen=True)
class RuntimeCapacity:
    """Fail-closed runtime capacity evidence (only verified values are trusted)."""

    client_base_url: str
    served_models: tuple[str, ...]
    model_identity_established: bool
    model_matched: bool
    declared_total_slots: int | None
    observed_slot_count: int | None
    per_slot_n_ctx: tuple[int, ...]
    slots_trusted: bool
    effective_n_ctx: int | None
    metadata_n_ctx: int | None
    tokenizer_supported: bool
    notes: tuple[str, ...] = ()


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else None


def _extract_model_ids(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return []
    ids: list[str] = []
    for item in data.get("models") or []:
        if isinstance(item, dict):
            ids.extend(v for v in (item.get("model"), item.get("name"), item.get("id"))
                       if isinstance(v, str) and v)
    for item in data.get("data") or []:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
            ids.append(item["id"])
    return ids


def _extract_models_meta_nctx(data: Any) -> int | None:
    if not isinstance(data, dict):
        return None
    for item in data.get("data") or []:
        if isinstance(item, dict):
            n = _positive_int((item.get("meta") or {}).get("n_ctx"))
            if n is not None:
                return n
    return None


def _validate_slots(slots: Any, declared_total_slots: int | None):
    """Return ``(per_slot_n_ctx_tuple, ok, reason)`` for *trusted* per-slot evidence.

    Requires every slot to carry a valid positive ``n_ctx``; rejects malformed
    or duplicate slot identities (when present) and a slot count that conflicts
    with the declared ``total_slots``.  Never infers capacity by division.
    """
    if not isinstance(slots, list) or not slots:
        return (), False, "slot list is empty or not an array"
    nctxs: list[int] = []
    seen_ids: set[int] = set()
    for slot in slots:
        if not isinstance(slot, dict):
            return (), False, "a slot entry is not an object"
        n = _positive_int(slot.get("n_ctx"))
        if n is None:
            return (), False, "a slot has a missing or non-positive n_ctx"
        nctxs.append(n)
        if "id" in slot:
            sid = slot["id"]
            if not (isinstance(sid, int) and not isinstance(sid, bool)):
                return (), False, "a slot identity is malformed"
            if sid in seen_ids:
                return (), False, "duplicate slot identity"
            seen_ids.add(sid)
    if declared_total_slots is not None and len(nctxs) != declared_total_slots:
        return (), False, f"slot count {len(nctxs)} != declared total_slots {declared_total_slots}"
    return tuple(nctxs), True, None


def discover_runtime(client: LlamaClient, request_model: str) -> RuntimeCapacity:
    """Discover capacity via read-only endpoints; unverified evidence stays untrusted."""
    notes: list[str] = []
    # /v1/models: served identities (exact match) + untrusted metadata n_ctx.
    served_models: tuple[str, ...] = ()
    model_identity_established = False
    models_meta_n_ctx: int | None = None
    try:
        data = client.get("/v1/models")
        ids = _extract_model_ids(data)
        models_meta_n_ctx = _extract_models_meta_nctx(data)
        if ids:
            served_models = tuple(dict.fromkeys(ids))
            model_identity_established = True
            if request_model not in served_models:
                notes.append("request_model is not among the served model identities")
        else:
            notes.append("/v1/models returned no usable model identifiers")
    except LlamaHTTPError as exc:
        notes.append(f"/v1/models unavailable: {exc}")
    model_matched = model_identity_established and request_model in served_models
    # /props: declared total_slots (for the slot count check) + untrusted default n_ctx.
    declared_total_slots: int | None = None
    props_n_ctx: int | None = None
    try:
        props = client.get("/props")
        if isinstance(props, dict):
            declared_total_slots = _positive_int(props.get("total_slots"))
            props_n_ctx = _positive_int((props.get("default_generation_settings") or {}).get("n_ctx"))
    except LlamaHTTPError as exc:
        notes.append(f"/props unavailable: {exc}")
    metadata_n_ctx = models_meta_n_ctx if models_meta_n_ctx is not None else props_n_ctx
    # /slots: the *only* source of trusted per-slot capacity.
    per_slot: tuple[int, ...] = ()
    slots_trusted = False
    effective: int | None = None
    observed_slot_count: int | None = None
    try:
        slots = client.get("/slots")
        if isinstance(slots, list):
            observed_slot_count = len(slots)
        per_slot, slots_trusted, reason = _validate_slots(slots, declared_total_slots)
        if reason:
            notes.append(f"/slots: {reason}")
        if slots_trusted:
            effective = min(per_slot)
    except LlamaHTTPError as exc:
        notes.append(f"/slots unavailable: {exc}")
    # Tokenizer support: the authoritative path needs /apply-template.
    tokenizer_supported = False
    try:
        prompt = client.apply_template([{"role": "user", "content": "x"}], add_generation_prompt=False)
        tokenizer_supported = bool(prompt and prompt.strip())
        if not tokenizer_supported:
            notes.append("/apply-template returned an empty prompt")
    except LlamaHTTPError as exc:
        notes.append(f"/apply-template unavailable: {exc}")
    return RuntimeCapacity(
        client_base_url=client.base_url, served_models=served_models,
        model_identity_established=model_identity_established, model_matched=model_matched,
        declared_total_slots=declared_total_slots, observed_slot_count=observed_slot_count,
        per_slot_n_ctx=per_slot, slots_trusted=slots_trusted, effective_n_ctx=effective,
        metadata_n_ctx=metadata_n_ctx, tokenizer_supported=tokenizer_supported, notes=tuple(notes),
    )


def count_prompt_tokens(client: LlamaClient, messages: Any) -> tuple[int | None, str | None]:
    """Count model prompt tokens via the server's authoritative endpoints.

    Renders the production messages through the served chat template
    (``POST /apply-template``) and counts the rendered prompt with ``POST
    /tokenize`` (``parse_special`` so template special tokens are honored).
    Returns ``(count, None)`` on success or ``(None, reason)`` when evidence is
    unavailable or malformed.  The raw prompt is never logged.
    """
    try:
        prompt = client.apply_template(messages, add_generation_prompt=True)
    except LlamaHTTPError as exc:
        return None, f"apply-template unavailable (HTTP {exc.status})"
    if not prompt or not prompt.strip():
        return None, "apply-template returned an empty prompt"
    try:
        tokens = client.tokenize(prompt)
    except LlamaHTTPError as exc:
        # HTTP failure/connection vs a 2xx body that lacks a valid token list.
        if exc.status == 0 or exc.status >= 400:
            return None, f"tokenizer endpoint unavailable (HTTP {exc.status})"
        return None, f"tokenizer response is malformed (HTTP {exc.status})"
    if not tokens:
        return None, "tokenizer returned an empty token list for a nonempty prompt"
    if any(not (isinstance(t, int) and not isinstance(t, bool)) for t in tokens):
        return None, "tokenizer returned a token list with invalid element types"
    return len(tokens), None


def _estimator_ok(semantic: Any, budget: B13WindowSemanticBudget) -> bool:
    try:
        semantic.enforce_budget(budget)
        return True
    except B13WindowSemanticBudgetError:
        return False


def assess_window(*, packet, initial, regen, model_context_tokens, input_ceiling_tokens,
                  tokenizer_prompt_tokens, tokenizer_reason) -> dict[str, Any]:
    """Build one bounded per-window preflight report row.

    The input ceiling is a hard limit even when ``n_ctx`` is not verified; total
    context verification requires a verified ``n_ctx`` (otherwise UNKNOWN).
    """
    window = packet.to_dict()["window"]
    reserved = initial.measurement.reserved_output_tokens
    est = initial.measurement
    regen_est = regen.measurement
    reasons: list[str] = []
    if model_context_tokens is None:
        over = (est.provider_request_estimated_tokens > input_ceiling_tokens
                or regen_est.provider_request_estimated_tokens > input_ceiling_tokens)
        estimator = "FAIL" if over else "UNKNOWN"
        reasons.append(f"estimator: initial {est.provider_request_estimated_tokens} / "
                       f"regen {regen_est.provider_request_estimated_tokens} est tokens "
                       f"(input ceiling {input_ceiling_tokens}, n_ctx not verified)")
    else:
        budget = B13WindowSemanticBudget(
            initial.runtime_config.request_model, model_context_tokens, input_ceiling_tokens)
        estimator = "PASS" if (_estimator_ok(initial, budget) and _estimator_ok(regen, budget)) else "FAIL"
        if estimator != "PASS":
            reasons.append(f"estimator: initial {est.provider_request_estimated_tokens} / "
                           f"regen {regen_est.provider_request_estimated_tokens} est tokens "
                           f"(ceiling {input_ceiling_tokens}, n_ctx {model_context_tokens})")
    if tokenizer_prompt_tokens is None:
        tokenizer = "UNKNOWN"
    elif model_context_tokens is None:
        tokenizer = "FAIL" if tokenizer_prompt_tokens > input_ceiling_tokens else "UNKNOWN"
        if tokenizer != "PASS":
            reasons.append(f"tokenizer: prompt {tokenizer_prompt_tokens} tokens "
                           f"(input ceiling {input_ceiling_tokens}, n_ctx not verified)")
    else:
        fits = (tokenizer_prompt_tokens <= input_ceiling_tokens
                and tokenizer_prompt_tokens + reserved <= model_context_tokens)
        tokenizer = "PASS" if fits else "FAIL"
        if not fits:
            reasons.append(f"tokenizer: prompt {tokenizer_prompt_tokens} + reserve {reserved} "
                           f"vs ceiling {input_ceiling_tokens} / n_ctx {model_context_tokens}")
    if tokenizer_reason:
        reasons.append(f"tokenizer: {tokenizer_reason}")
    verdict = ("FAIL" if ("FAIL" in (estimator, tokenizer))
               else "UNKNOWN" if "UNKNOWN" in (estimator, tokenizer)
               else "PASS")
    return {
        "window_id": packet.window_id, "window_ordinal": packet.window_ordinal,
        "owned_event_count": len(window["owned_event_refs"]),
        "context_event_count": len(window["context_event_refs"]),
        "packet_bytes": est.packet_bytes,
        "packet_estimated_tokens": est.packet_estimated_tokens,
        "initial_provider_request_bytes": est.provider_request_bytes,
        "initial_provider_request_estimated_tokens": est.provider_request_estimated_tokens,
        "regen_provider_request_bytes": regen_est.provider_request_bytes,
        "regen_provider_request_estimated_tokens": regen_est.provider_request_estimated_tokens,
        "reserved_output_tokens": reserved,
        "tokenizer_prompt_tokens": tokenizer_prompt_tokens,
        "request_identity": initial.request_identity,
        "estimator_budget": estimator, "tokenizer_budget": tokenizer, "verdict": verdict,
        "reasons": reasons,
    }


def assess_preparation(*, preparation, runtime_config, client, model_context_tokens,
                       input_ceiling_tokens):
    """Assess every prepared window (initial + cache-disabled regeneration)."""
    rows = []
    for packet in preparation.packets:
        initial = build_b13_window_semantic_preparation(packet, runtime_config=runtime_config)
        regen = build_b13_window_semantic_preparation(
            packet, runtime_config=runtime_config,
            execution_options=GenerationExecutionOptions(prompt_context_reuse="disabled"))
        if initial.request_identity != regen.request_identity:
            raise ValueError("regeneration changed the semantic request identity")
        # The server-rendered prompt depends only on the request messages,
        # identical for initial and regeneration (only ``cache_prompt`` differs
        # in the HTTP body), so one measurement covers both request shapes.
        tokens, reason = count_prompt_tokens(client, initial.request.messages)
        rows.append(assess_window(
            packet=packet, initial=initial, regen=regen, model_context_tokens=model_context_tokens,
            input_ceiling_tokens=input_ceiling_tokens,
            tokenizer_prompt_tokens=tokens, tokenizer_reason=reason))
    return rows


ALICE_STRATEGY_REF = ArtifactRef(
    B12_STRATEGY_ARTIFACT_TYPE, B12_STRATEGY_ARTIFACT_ID,
    B12_STRATEGY_REVISION, B12_STRATEGY_CONTENT_HASH,
)


def run_preflight(*, store, pointers, source_snapshot, runtime_config, client,
                  input_ceiling_tokens, model_context_tokens=None,
                  strategy_ref=None, adaptation_target_id=ADAPTATION_TARGET_ID):
    """Run the complete read-only preflight and return a bounded report dict.

    ``strategy_ref`` defaults to the pinned Alice B1.2 strategy (explicit, never
    the CURRENT fallback); tests may inject a fixture strategy ref.  The overall
    verdict is capped at ``UNKNOWN`` unless capacity is fully trusted.
    """
    preparation = prepare_b13_disposition_windows(
        store=store, pointers=pointers, source_snapshot=source_snapshot,
        adaptation_target_id=adaptation_target_id,
        strategy_ref=strategy_ref if strategy_ref is not None else ALICE_STRATEGY_REF)
    capacity = discover_runtime(client, runtime_config.request_model)
    server_root = _server_root(runtime_config.base_url)
    server_ok = _server_root(client.base_url) == server_root
    notes = list(capacity.notes)
    if not server_ok:
        notes.append(f"discovery client base_url {client.base_url} != runtime server root {server_root}")
    discovered = capacity.effective_n_ctx
    # Resolve the effective context limit; an override may only lower verified capacity.
    if model_context_tokens is None:
        effective, n_ctx_source = discovered, ("discovered_per_slot_min" if discovered is not None else "none")
    elif discovered is None:
        effective, n_ctx_source = model_context_tokens, "operator_override_discovery_unverified"
        notes.append("operator override used but discovered capacity is unverified; a trusted PASS is not possible")
    elif model_context_tokens > discovered:
        effective, n_ctx_source = None, "operator_override_rejected_exceeds_discovered"
        notes.append(f"operator override {model_context_tokens} exceeds discovered per-slot n_ctx "
                     f"{discovered}; refusing to raise verified capacity (fail closed)")
    else:
        effective, n_ctx_source = model_context_tokens, "operator_override_capped_at_discovered"
    capacity_trusted = (server_ok and capacity.model_identity_established and capacity.model_matched
                        and capacity.slots_trusted and discovered is not None and effective is not None)
    rows = assess_preparation(
        preparation=preparation, runtime_config=runtime_config, client=client,
        model_context_tokens=effective, input_ceiling_tokens=input_ceiling_tokens)
    if not capacity_trusted:
        rows = [{**row, "verdict": ("UNKNOWN" if row["verdict"] == "PASS" else row["verdict"])}
                for row in rows]
    verdicts = [row["verdict"] for row in rows]
    overall = "FAIL" if "FAIL" in verdicts else "UNKNOWN" if "UNKNOWN" in verdicts else "PASS"
    return {
        "schema_version": 1, "stage": "B1.3-B2A preflight", "read_only": True,
        "runtime": {"base_url": runtime_config.base_url,
                    "request_model": runtime_config.request_model,
                    "provider_family": runtime_config.provider_family,
                    "transport_id": runtime_config.transport_id},
        "capacity": {"client_base_url": client.base_url, "runtime_server_root": server_root,
                     "server_ok": server_ok, "served_models": list(capacity.served_models),
                     "model_identity_established": capacity.model_identity_established,
                     "model_matched": capacity.model_matched,
                     "declared_total_slots": capacity.declared_total_slots,
                     "observed_slot_count": capacity.observed_slot_count,
                     "per_slot_n_ctx": list(capacity.per_slot_n_ctx),
                     "slots_trusted": capacity.slots_trusted,
                     "discovered_effective_n_ctx": discovered, "metadata_n_ctx": capacity.metadata_n_ctx,
                     "tokenizer_supported": capacity.tokenizer_supported, "notes": notes},
        "policy": {"model_context_tokens": effective, "model_context_tokens_source": n_ctx_source,
                   "input_ceiling_tokens": input_ceiling_tokens,
                   "reserved_output_tokens_source": "pinned b13-window-disposition-llm-v1 profile",
                   "capacity_trusted": capacity_trusted},
        "windows_total": len(rows), "windows": rows, "overall_verdict": overall,
        "generation_calls": client.generation_calls,
    }


def load_alice_authority(story_root):
    """Open the Alice run read-only and recover the pinned source snapshot."""
    story_root = Path(story_root)
    store = FileArtifactStore(story_root / "artifacts")
    pointers = FilePointerStore(story_root / "pointers", store)
    envelope = store.get_ref(ArtifactRef(
        B12_STRATEGY_ARTIFACT_TYPE, B12_STRATEGY_ARTIFACT_ID,
        B12_STRATEGY_REVISION, B12_STRATEGY_CONTENT_HASH))
    return store, pointers, B12StrategyArtifact.from_dict(envelope.payload).source_snapshot


def _print_summary(report, args) -> None:
    cap, policy = report["capacity"], report["policy"]
    print("B1.3 Alice tokenizer & budget preflight (read-only; PR-A)")
    print(f"  model_matched={cap['model_matched']} server_ok={cap['server_ok']} "
          f"slots_trusted={cap['slots_trusted']} capacity_trusted={policy['capacity_trusted']}")
    print(f"  served={cap['served_models']} per_slot_n_ctx={cap['per_slot_n_ctx']} "
          f"discovered_effective={cap['discovered_effective_n_ctx']} metadata={cap['metadata_n_ctx']}")
    print(f"  n_ctx={policy['model_context_tokens']} (source={policy['model_context_tokens_source']}) "
          f"input_ceiling={policy['input_ceiling_tokens']} windows={report['windows_total']} "
          f"overall={report['overall_verdict']}")
    for row in report["windows"]:
        print(f"  {row['window_id']} ord={row['window_ordinal']} owned={row['owned_event_count']} "
              f"est={row['initial_provider_request_estimated_tokens']}/{row['regen_provider_request_estimated_tokens']} "
              f"tok={row['tokenizer_prompt_tokens']} estimator={row['estimator_budget']} "
              f"tokenizer={row['tokenizer_budget']} verdict={row['verdict']}")
        for reason in row["reasons"]:
            print(f"      - {reason}")
    print(f"  generation_calls={report['generation_calls']} report={args.report}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="B1.3 Alice tokenizer & budget preflight (read-only)")
    parser.add_argument("--story-root", type=Path, default=DEFAULT_STORY_ROOT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--request-model", default=DEFAULT_REQUEST_MODEL)
    parser.add_argument("--provider-family", default=DEFAULT_PROVIDER_FAMILY)
    parser.add_argument("--transport-id", default=DEFAULT_TRANSPORT_ID)
    parser.add_argument("--input-ceiling-tokens", type=int, required=True,
                        help="operator-supplied input ceiling (never auto-defaulted)")
    parser.add_argument("--model-context-tokens", type=int, default=None,
                        help="explicit operator n_ctx override (may only lower verified capacity)")
    parser.add_argument("--report", type=Path,
                        default=Path("/tmp/b13_alice_token_budget_preflight_report.json"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-response-bytes", type=int, default=4_000_000)
    args = parser.parse_args(argv)
    try:
        store, pointers, source_snapshot = load_alice_authority(args.story_root)
        runtime_config = RuntimeConfig(
            schema_version=2, transport_id=args.transport_id, base_url=args.base_url,
            request_model=args.request_model, provider_family=args.provider_family,
            credential_environment_name=None, timeout_seconds=args.timeout)
        client = LlamaClient(base_url=_server_root(args.base_url), timeout_seconds=args.timeout,
                             max_response_bytes=args.max_response_bytes)
        report = run_preflight(
            store=store, pointers=pointers, source_snapshot=source_snapshot,
            runtime_config=runtime_config, client=client,
            input_ceiling_tokens=args.input_ceiling_tokens,
            model_context_tokens=args.model_context_tokens)
    except Exception as exc:  # noqa: BLE001 - unexpected diagnostic execution error
        print(f"preflight diagnostic error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"failed to write report: {exc}", file=sys.stderr)
        return EXIT_ERROR
    _print_summary(report, args)
    return _exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
