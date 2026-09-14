from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from short_drama.artifacts import content_hash
from short_drama.io import load_yaml

from .errors import LLMPromptError
from .models import (
    SHA256_RE,
    PromptSpec,
    RenderedPrompt,
    compute_prompt_content_hash,
    compute_rendered_prompt_hash,
    require_storage_id,
    substitute_template,
)

PROMPT_DIR_LAYOUT_VERSION = 1
_PROMPT_YAML_FIELDS = {
    "schema_version",
    "prompt_id",
    "version",
    "required_variables",
    "content_hash",
}


class PromptRegistry:
    """Lightweight deterministic versioned prompt registry.

    Storage layout (no database, no remote service, no template engine):

        <base_dir>/<prompt_id>/v<version>/prompt.yaml
        <base_dir>/<prompt_id>/v<version>/system.txt
        <base_dir>/<prompt_id>/v<version>/user.txt

    A given ``prompt_id``/``version`` directory is immutable: changing the
    template text or metadata under the same version is detected via the
    content hash and fails closed rather than being silently accepted.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def load(self, prompt_id: str, *, version: int) -> PromptSpec:
        # Validate the prompt_id against the safe storage-ID contract BEFORE
        # using it to construct filesystem paths (no traversal / unsafe names).
        require_storage_id(prompt_id, "prompt_id", LLMPromptError)
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise LLMPromptError("version must be an integer >= 1")

        version_dir = self._base_dir / prompt_id / f"v{version}"
        yaml_path = version_dir / "prompt.yaml"
        system_path = version_dir / "system.txt"
        user_path = version_dir / "user.txt"

        if not version_dir.is_dir():
            raise LLMPromptError(f"prompt version not found: {version_dir}")
        for required in (yaml_path, system_path, user_path):
            if not required.is_file():
                raise LLMPromptError(f"prompt file not found: {required}")

        try:
            metadata = load_yaml(yaml_path)
        except Exception as exc:  # noqa: BLE001 - report any load failure
            raise LLMPromptError(f"failed to load prompt.yaml: {exc}") from exc
        if not isinstance(metadata, dict):
            raise LLMPromptError("prompt.yaml must contain an object")
        self._validate_metadata(metadata, prompt_id=prompt_id, version=version)

        try:
            system_template = system_path.read_text(encoding="utf-8")
            user_template = user_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise LLMPromptError(f"failed to read prompt templates: {exc}") from exc

        # Enforce same-version immutability: the pinned content hash in
        # prompt.yaml must exactly match the hash computed from the semantic
        # prompt material. Changing system.txt, user.txt, or semantic metadata
        # (e.g. required_variables) under the same prompt_id + version without
        # re-versioning fails closed rather than being silently accepted.
        actual_hash = compute_prompt_content_hash(
            prompt_id=prompt_id,
            version=version,
            system_template=system_template,
            user_template=user_template,
            required_variables=metadata["required_variables"],
        )
        if actual_hash != metadata["content_hash"]:
            raise LLMPromptError(
                "prompt.yaml content_hash does not match the computed prompt "
                "content hash; a prompt version was modified without "
                "re-versioning (prompt identity is "
                "prompt_id + version + content_hash)"
            )

        return PromptSpec.create(
            prompt_id=prompt_id,
            version=version,
            system_template=system_template,
            user_template=user_template,
            required_variables=metadata["required_variables"],
        )

    def _validate_metadata(
        self, metadata: dict[str, Any], *, prompt_id: str, version: int
    ) -> None:
        if set(metadata) != _PROMPT_YAML_FIELDS:
            raise LLMPromptError(
                "prompt.yaml must contain exactly: "
                + ", ".join(sorted(_PROMPT_YAML_FIELDS))
            )
        if metadata["schema_version"] != 1:
            raise LLMPromptError("prompt.yaml schema_version must be 1")
        if metadata["prompt_id"] != prompt_id:
            raise LLMPromptError(
                "prompt.yaml prompt_id does not match the prompt directory"
            )
        if metadata["version"] != version:
            raise LLMPromptError(
                "prompt.yaml version does not match the prompt directory"
            )
        required = metadata["required_variables"]
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise LLMPromptError("required_variables must be a list of strings")
        pinned_hash = metadata["content_hash"]
        if not isinstance(pinned_hash, str) or SHA256_RE.fullmatch(pinned_hash) is None:
            raise LLMPromptError(
                "content_hash must be a lowercase 64-char SHA-256 hex digest"
            )


def render_prompt(
    spec: PromptSpec, variables: Mapping[str, str]
) -> RenderedPrompt:
    """Deterministically render a PromptSpec with an exact variable mapping.

    Fails closed if the provided variables do not exactly match the declared
    ``required_variables`` (missing or unexpected), or if a variable value is
    not a string.
    """

    if not isinstance(spec, PromptSpec):
        raise LLMPromptError("spec must be a PromptSpec")
    provided = dict(variables)
    required = set(spec.required_variables)
    if set(provided) != required:
        missing = sorted(required - set(provided))
        unexpected = sorted(set(provided) - required)
        raise LLMPromptError(
            "render variables must exactly match required_variables; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, value in provided.items():
        if not isinstance(value, str):
            raise LLMPromptError(f"variable {name!r} must be a string value")

    system_text = substitute_template(spec.system_template, provided)
    user_text = substitute_template(spec.user_template, provided)
    variables_hash = content_hash(
        {"variables": {name: provided[name] for name in sorted(provided)}}
    )
    rendered_hash = compute_rendered_prompt_hash(
        prompt_id=spec.prompt_id,
        prompt_version=spec.version,
        prompt_content_hash=spec.content_hash,
        variables_hash=variables_hash,
        system_text=system_text,
        user_text=user_text,
    )
    return RenderedPrompt(
        prompt_id=spec.prompt_id,
        prompt_version=spec.version,
        prompt_content_hash=spec.content_hash,
        variables_hash=variables_hash,
        system_text=system_text,
        user_text=user_text,
        rendered_prompt_hash=rendered_hash,
    )
