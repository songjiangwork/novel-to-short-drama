"""A-I3 local-Qwen structured-output smoke.

Verifies the full A-I3 infrastructure path against a real local
OpenAI-compatible server (llama.cpp / Qwen):

    runtime config
      -> OpenAICompatibleLLMClient
      -> local Qwen
      -> structured response
      -> strict local JSON + JSON Schema validation
      -> invocation provenance

It uses a tiny synthetic prompt and an artificial JSON Schema. It does NOT test
novel/story understanding (that is A-I4). It is intentionally not part of the
unit test suite: unit tests use a deterministic fake transport and never touch
a live server.

Usage:
    python scripts/llm_smoke.py \
        --runtime-config profiles/llm_local.yaml \
        --profile profiles/story_extraction_llm_v1.yaml \
        [--model <exact-model-name>]

Exit code 0 on success, 2 on failure.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from short_drama.llm import (
    LLMError,
    OpenAICompatibleLLMClient,
    OutputSchema,
    PromptSpec,
    load_runtime_config,
    load_semantic_profile,
    render_prompt,
)


def _build_synthetic_rendered() -> object:
    spec = PromptSpec.create(
        prompt_id="a3.infra-smoke",
        version=1,
        system_template=(
            "You are a precise extraction assistant. Respond only with valid "
            "JSON that matches the required schema."
        ),
        user_template=(
            "Extract the single integer described in the text below and return "
            "it. Text: \"{{text}}\". Respond with a JSON object having exactly "
            "one field, \"answer\", whose value is that integer."
        ),
        required_variables=["text"],
    )
    return render_prompt(spec, {"text": "The important number is 42."})


def _build_artificial_schema() -> OutputSchema:
    return OutputSchema.create(
        schema_id="a3.infra-smoke-schema",
        schema_version=1,
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {"answer": {"type": "integer", "minimum": 0, "maximum": 1000}},
        },
    )


def run_smoke(
    *,
    runtime_config_path: str,
    profile_path: str,
    model_override: str | None,
) -> int:
    runtime_config = load_runtime_config(runtime_config_path)
    semantic_profile = load_semantic_profile(profile_path)
    # The concrete backend model is a runtime/routing identity carried by the
    # RuntimeConfig (request_model), not by the semantic profile, so a --model
    # override retargets the runtime config, not the semantic identity.
    if model_override is not None:
        runtime_config = dataclasses.replace(
            runtime_config, request_model=model_override
        )

    rendered = _build_synthetic_rendered()
    output_schema = _build_artificial_schema()

    client = OpenAICompatibleLLMClient(runtime_config)

    # Inspect the provider request body to confirm the semantic reasoning
    # settings map to an EXPLICIT provider value (reasoning_effort), so the
    # server startup default never silently decides reasoning behavior.
    request_body = client.build_request_body(
        rendered, output_schema, semantic_profile
    )
    reasoning_effort = request_body.get("reasoning_effort")
    expected_effort = semantic_profile.reasoning.request_effort
    if reasoning_effort != expected_effort:
        print(
            f"SMOKE FAILED: request reasoning_effort {reasoning_effort!r} does "
            f"not match profile {expected_effort!r}",
            file=sys.stderr,
        )
        return 2

    print("== A-I3 structured-output smoke ==")
    print(f"transport_id:        {runtime_config.transport_id}")
    print(f"base_url:            {runtime_config.base_url}")
    print(f"credential_env:      {runtime_config.credential_environment_name}")
    print(f"profile_id:          {semantic_profile.profile_id}")
    print(f"provider_family:     {runtime_config.provider_family}")
    print(f"model:               {runtime_config.request_model}")
    print(f"structured_output:   {semantic_profile.structured_output_mode}")
    print(f"reasoning_effort:    {reasoning_effort}")
    print()

    result = client.generate_structured(rendered, output_schema, semantic_profile)

    request_hash = result.provenance.request_hash
    print(f"attempts:            {result.attempts}")
    print(f"parsed_json:         {json.dumps(result.parsed_json, ensure_ascii=False)}")
    print(f"request_hash:        {request_hash}")
    print("provenance:")
    print(json.dumps(result.provenance.to_dict(), ensure_ascii=False, indent=2))

    # Explicit runtime failure handling (not `assert`, which is stripped under
    # `python -O`) so the documented failure exit behavior always holds.
    if result.parsed_json != {"answer": 42}:
        print(
            f"SMOKE FAILED: expected the model to return {{'answer': 42}}; "
            f"got {result.parsed_json!r}",
            file=sys.stderr,
        )
        return 2
    print("SMOKE OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A-I3 local-Qwen structured-output smoke"
    )
    parser.add_argument("--runtime-config", required=True, help="runtime transport config YAML")
    parser.add_argument("--profile", required=True, help="semantic LLM profile YAML")
    parser.add_argument(
        "--model",
        default=None,
        help="override the profile's model with the exact server model name",
    )
    args = parser.parse_args()
    try:
        return run_smoke(
            runtime_config_path=args.runtime_config,
            profile_path=args.profile,
            model_override=args.model,
        )
    except LLMError as exc:
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
