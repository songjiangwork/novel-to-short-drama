#!/usr/bin/env python3
"""B1.3 Alice tokenizer & runtime-budget preflight (read-only; PR-A only).

Replaces the ``Budget UNKNOWN`` assessment with verifiable per-window capacity
evidence *without* invoking semantic generation.  It discovers the served
llama.cpp model identity and effective per-slot context via read-only endpoints
(``/v1/models``, ``/props``, ``/slots``, ``/tokenize``), builds the production B1.3
window semantic preparations for the exact pinned B1.2 strategy (initial +
cache-disabled regeneration), and reports, per window, the production
repository-estimator budget, the tokenizer-measured prompt capacity (where
reliably supported), and a final operator preflight verdict.

Read-only and generation-free: it never calls a completion/chat-completion
endpoint, never writes production artifacts or mutates CURRENT pointers, and
refuses to fabricate a PASS -- any capacity evidence that cannot be established
is reported as ``UNKNOWN`` with a precise reason.  The bounded diagnostic report
(default ``/tmp``) carries only metadata, hashes, and counts.
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

_GENERATION_PATHS = frozenset({
    "v1/chat/completions", "chat/completions", "v1/completions", "completions",
    "completion", "responses", "generate",
})


class LlamaHTTPError(Exception):
    """A read-only llama.cpp request failed (bounded, secret-safe detail)."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class ChatTemplateRenderError(Exception):
    """The served chat template could not be rendered to a prompt string."""


def _server_root(base_url: str) -> str:
    """Reduce a ``.../v1`` LLM base URL to the llama.cpp server root.

    The read-only discovery endpoints (``/props``, ``/slots``, ``/tokenize``)
    and ``/v1/models`` are addressed from the server root, not the ``/v1``
    chat-completions prefix.
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

    @property
    def generation_calls(self) -> list[str]:
        return [p for p in self.requested_paths if p.lstrip("/").rstrip("/") in _GENERATION_PATHS]


@dataclass(frozen=True)
class RuntimeCapacity:
    request_model: str
    served_model: str | None
    model_matched: bool
    total_slots: int | None
    per_slot_n_ctx: tuple[int, ...]
    effective_n_ctx: int | None
    n_ctx_source: str | None
    chat_template: str | None
    tokenizer_supported: bool
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.model_matched and self.effective_n_ctx is not None


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else None


def _discover_models(client: LlamaClient, request_model: str, notes: list[str]):
    data = client.get("/v1/models")
    if not isinstance(data, dict):
        raise LlamaHTTPError("/v1/models did not return an object", 200)
    ids: list[str] = []
    for item in data.get("models") or []:
        if isinstance(item, dict):
            ids.extend(v for v in (item.get("model"), item.get("name")) if isinstance(v, str))
    for item in data.get("data") or []:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    if request_model not in ids:
        notes.append("served model identity does not match request_model")
    meta = next(
        (_positive_int((it.get("meta") or {}).get("n_ctx")) for it in (data.get("data") or [])
         if isinstance(it, dict) and _positive_int((it.get("meta") or {}).get("n_ctx"))),
        None,
    )
    return (ids[0] if ids else None), meta


def _discover_slots(client: LlamaClient) -> tuple[int, ...]:
    data = client.get("/slots")
    if not isinstance(data, list):
        raise LlamaHTTPError("/slots did not return an array", 200)
    values = tuple(_positive_int(it.get("n_ctx")) if isinstance(it, dict) else None for it in data)
    if any(v is None for v in values) or not values:
        raise LlamaHTTPError("/slots has no positive per-slot n_ctx", 200)
    return values


def _discover_props(client: LlamaClient):
    data = client.get("/props")
    if not isinstance(data, dict):
        raise LlamaHTTPError("/props did not return an object", 200)
    return (
        _positive_int(data.get("total_slots")),
        _positive_int((data.get("default_generation_settings") or {}).get("n_ctx")),
        data.get("chat_template") if isinstance(data.get("chat_template"), str) else None,
    )


def discover_runtime(client: LlamaClient, request_model: str) -> RuntimeCapacity:
    """Discover capacity; any endpoint that cannot be read degrades to UNKNOWN."""
    notes: list[str] = []
    served_model = meta_n_ctx = None
    try:
        served_model, meta_n_ctx = _discover_models(client, request_model, notes)
    except LlamaHTTPError as exc:
        notes.append(f"/v1/models unavailable: {exc}")
    per_slot: tuple[int, ...] = ()
    try:
        per_slot = _discover_slots(client)
    except LlamaHTTPError as exc:
        notes.append(f"/slots unavailable: {exc}")
    total_slots = default_n_ctx = None
    chat_template = None
    try:
        total_slots, default_n_ctx, chat_template = _discover_props(client)
    except LlamaHTTPError as exc:
        notes.append(f"/props unavailable: {exc}")
    # Effective per-slot context = min across slots (never total/slot_count).
    # Model-level evidence is only a fallback when slots are unavailable, and the
    # source is always recorded so a value is never silently assumed.
    effective, source = (min(per_slot), "slots") if per_slot else (
        (meta_n_ctx, "models") if meta_n_ctx is not None else
        (default_n_ctx, "props") if default_n_ctx is not None else (None, None)
    )
    tokenizer_supported = False
    try:
        probe = client.post("/tokenize", {"content": "x"})
        tokenizer_supported = isinstance(probe, dict) and isinstance(probe.get("tokens"), list)
        if not tokenizer_supported:
            notes.append("/tokenize response missing a 'tokens' list")
    except LlamaHTTPError as exc:
        notes.append(f"/tokenize unavailable: {exc}")
    return RuntimeCapacity(
        request_model=request_model, served_model=served_model,
        model_matched=served_model is not None and request_model in (served_model,),
        total_slots=total_slots, per_slot_n_ctx=per_slot, effective_n_ctx=effective,
        n_ctx_source=source, chat_template=chat_template,
        tokenizer_supported=tokenizer_supported, notes=tuple(notes),
    )


def default_chat_template_renderer(chat_template: str, messages: Any) -> str:
    """Render the served llama.cpp chat template over ``messages``.

    Uses ``jinja2`` when importable (it is not a declared project dependency),
    registering the llama.cpp ``namespace``/``raise_exception`` extensions;
    raises :class:`ChatTemplateRenderError` when rendering cannot be trusted.
    """
    try:
        import jinja2
    except ImportError as exc:
        raise ChatTemplateRenderError("jinja2 is not available") from exc
    try:
        env = jinja2.Environment(undefined=jinja2.ChainableUndefined)
        # llama.cpp's ``namespace()`` maps to Jinja2's mutable Namespace object
        # (required for ``set x.attr = ...`` inside the template).
        env.globals["namespace"] = jinja2.runtime.Namespace
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(
            ChatTemplateRenderError(f"raise_exception: {msg}"))
        return str(env.from_string(chat_template).render(
            messages=list(messages), add_generation_prompt=True))
    except ChatTemplateRenderError:
        raise
    except Exception as exc:  # noqa: BLE001 - any engine failure is "untrusted"
        raise ChatTemplateRenderError(f"{type(exc).__name__}") from exc


def count_prompt_tokens(client, messages, chat_template, *, renderer=None):
    """Count model prompt tokens via the real tokenizer.

    Returns ``(count, None)`` on success, or ``(None, reason)`` when reliable
    contextual counting cannot be established (unsupported/malformed).
    """
    if not chat_template:
        return None, "server did not expose a chat template"
    try:
        rendered = (renderer or default_chat_template_renderer)(chat_template, messages)
    except ChatTemplateRenderError as exc:
        return None, f"chat template render unavailable: {exc}"
    if not rendered or not rendered.strip():
        return None, "rendered chat prompt is empty"
    try:
        response = client.post("/tokenize", {"content": rendered})
    except LlamaHTTPError as exc:
        return None, f"tokenizer endpoint unavailable (HTTP {exc.status})"
    if not isinstance(response, dict) or not isinstance(response.get("tokens"), list):
        return None, "tokenizer response is malformed (no 'tokens' list)"
    return len(response["tokens"]), None


def _estimator_ok(semantic: Any, budget: B13WindowSemanticBudget) -> bool:
    try:
        semantic.enforce_budget(budget)
        return True
    except B13WindowSemanticBudgetError:
        return False


def assess_window(*, packet, initial, regen, model_context_tokens, input_ceiling_tokens,
                  tokenizer_prompt_tokens, tokenizer_reason) -> dict[str, Any]:
    """Build one bounded per-window preflight report row."""
    window = packet.to_dict()["window"]
    reserved = initial.measurement.reserved_output_tokens
    reasons: list[str] = []
    if model_context_tokens is None:
        estimator, est_reason = "UNKNOWN", "effective per-slot n_ctx unknown (context limit unverifiable)"
    else:
        budget = B13WindowSemanticBudget(
            initial.runtime_config.request_model, model_context_tokens, input_ceiling_tokens)
        estimator = "PASS" if (_estimator_ok(initial, budget) and _estimator_ok(regen, budget)) else "FAIL"
    if estimator != "PASS":
        reasons.append(f"estimator: initial {initial.measurement.provider_request_estimated_tokens} / "
                       f"regen {regen.measurement.provider_request_estimated_tokens} est tokens "
                       f"(ceiling {input_ceiling_tokens}, n_ctx {model_context_tokens})")
    if tokenizer_prompt_tokens is None or model_context_tokens is None:
        tokenizer = "UNKNOWN"
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
        "packet_bytes": initial.measurement.packet_bytes,
        "packet_estimated_tokens": initial.measurement.packet_estimated_tokens,
        "initial_provider_request_bytes": initial.measurement.provider_request_bytes,
        "initial_provider_request_estimated_tokens": initial.measurement.provider_request_estimated_tokens,
        "regen_provider_request_bytes": regen.measurement.provider_request_bytes,
        "regen_provider_request_estimated_tokens": regen.measurement.provider_request_estimated_tokens,
        "reserved_output_tokens": reserved,
        "tokenizer_prompt_tokens": tokenizer_prompt_tokens,
        "request_identity": initial.request_identity,
        "estimator_budget": estimator, "tokenizer_budget": tokenizer, "verdict": verdict,
        "reasons": reasons,
    }


def assess_preparation(*, preparation, runtime_config, capacity, client,
                       model_context_tokens, input_ceiling_tokens, renderer=None):
    """Assess every prepared window (initial + cache-disabled regeneration)."""
    rows = []
    for packet in preparation.packets:
        initial = build_b13_window_semantic_preparation(packet, runtime_config=runtime_config)
        regen = build_b13_window_semantic_preparation(
            packet, runtime_config=runtime_config,
            execution_options=GenerationExecutionOptions(prompt_context_reuse="disabled"))
        if initial.request_identity != regen.request_identity:
            raise ValueError("regeneration changed the semantic request identity")
        # The chat-templated prompt depends only on request messages, identical
        # for initial and regeneration (only ``cache_prompt`` differs in the
        # HTTP body), so one tokenizer measurement covers both request shapes.
        tokens, reason = count_prompt_tokens(
            client, initial.request.messages, capacity.chat_template, renderer=renderer)
        rows.append(assess_window(
            packet=packet, initial=initial, regen=regen,
            model_context_tokens=model_context_tokens,
            input_ceiling_tokens=input_ceiling_tokens,
            tokenizer_prompt_tokens=tokens, tokenizer_reason=reason))
    return rows


ALICE_STRATEGY_REF = ArtifactRef(
    B12_STRATEGY_ARTIFACT_TYPE, B12_STRATEGY_ARTIFACT_ID,
    B12_STRATEGY_REVISION, B12_STRATEGY_CONTENT_HASH,
)


def run_preflight(*, store, pointers, source_snapshot, runtime_config, client,
                  input_ceiling_tokens, model_context_tokens=None, renderer=None,
                  strategy_ref=None, adaptation_target_id=ADAPTATION_TARGET_ID):
    """Run the complete read-only preflight and return a bounded report dict.

    ``strategy_ref`` defaults to the pinned Alice B1.2 strategy (explicit, never
    the CURRENT fallback); tests may inject a fixture strategy ref.
    """
    preparation = prepare_b13_disposition_windows(
        store=store, pointers=pointers, source_snapshot=source_snapshot,
        adaptation_target_id=adaptation_target_id,
        strategy_ref=strategy_ref if strategy_ref is not None else ALICE_STRATEGY_REF)
    capacity = discover_runtime(client, runtime_config.request_model)
    effective = model_context_tokens if model_context_tokens is not None else capacity.effective_n_ctx
    rows = assess_preparation(
        preparation=preparation, runtime_config=runtime_config, capacity=capacity,
        client=client, model_context_tokens=effective,
        input_ceiling_tokens=input_ceiling_tokens, renderer=renderer)
    verdicts = [row["verdict"] for row in rows]
    overall = "FAIL" if "FAIL" in verdicts else "UNKNOWN" if "UNKNOWN" in verdicts else "PASS"
    return {
        "schema_version": 1, "stage": "B1.3-B2A preflight", "read_only": True,
        "runtime": {"base_url": runtime_config.base_url,
                    "request_model": runtime_config.request_model,
                    "provider_family": runtime_config.provider_family,
                    "transport_id": runtime_config.transport_id},
        "capacity": {"served_model": capacity.served_model,
                     "model_matched": capacity.model_matched,
                     "total_slots": capacity.total_slots,
                     "per_slot_n_ctx": list(capacity.per_slot_n_ctx),
                     "effective_n_ctx": capacity.effective_n_ctx,
                     "n_ctx_source": capacity.n_ctx_source,
                     "chat_template_present": capacity.chat_template is not None,
                     "tokenizer_supported": capacity.tokenizer_supported,
                     "notes": list(capacity.notes)},
        "policy": {"model_context_tokens": effective,
                   "model_context_tokens_source": (
                       "operator_override" if model_context_tokens is not None
                       else (capacity.n_ctx_source or "unknown")),
                   "input_ceiling_tokens": input_ceiling_tokens,
                   "reserved_output_tokens_source": "pinned b13-window-disposition-llm-v1 profile"},
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
    print(f"  model_matched={cap['model_matched']} served={cap['served_model']} "
          f"per_slot_n_ctx={cap['per_slot_n_ctx']} effective={cap['effective_n_ctx']} (source={cap['n_ctx_source']})")
    print(f"  tokenizer_supported={cap['tokenizer_supported']} n_ctx={policy['model_context_tokens']} "
          f"input_ceiling={policy['input_ceiling_tokens']} windows={report['windows_total']} overall={report['overall_verdict']}")
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
                        help="explicit operator n_ctx override (default: discovered)")
    parser.add_argument("--report", type=Path,
                        default=Path("/tmp/b13_alice_token_budget_preflight_report.json"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-response-bytes", type=int, default=4_000_000)
    args = parser.parse_args(argv)
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
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(report, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
