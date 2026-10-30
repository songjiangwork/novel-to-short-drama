"""Only synthetic bytes; successful parsing is not runtime evidence verification."""
import copy
import hashlib
import json
import builtins
import os
import socket
from dataclasses import FrozenInstanceError

import pytest

from short_drama.artifacts import canonical_json_bytes
from short_drama.adaptation.disposition_recovery_runtime_evidence_codec import (
    RuntimeEvidenceCodecError, check_runtime_evidence_bytes,
)

PROVIDER = "b13.provider-config.v1"
MODEL = "b13.model-config.v1"
TOKENIZER = "b13.tokenizer-config.v1"
H = "a" * 64
VECTOR = (b'{"codec":"b13.provider-config.v1","id":"transport.synthetic",'
          b'"provider_family":"synthetic","request_model":"synthetic-model",'
          b'"schema_version":1,"timeout_decimal_seconds":"30",'
          b'"transport_binding_ref":"synthetic-binding"}')


def samples():
    return {
        PROVIDER: {"schema_version": 1, "codec": PROVIDER, "id": "transport.synthetic",
                   "provider_family": "synthetic", "request_model": "synthetic-model",
                   "timeout_decimal_seconds": "30", "transport_binding_ref": "synthetic-binding"},
        MODEL: {"schema_version": 1, "codec": MODEL, "id": "synthetic-model",
                "deployment_ref": "fake-deployment", "model_revision_ref": "fake-revision",
                "context_limit_tokens": 4096, "discovery_method": "fake"},
        TOKENIZER: {"schema_version": 1, "codec": TOKENIZER, "id": "fake-tokenizer",
                    "implementation_ref": "fake-implementation", "version": "v1",
                    "vocabulary_sha256": H, "special_tokens_sha256": H,
                    "chat_template_sha256": H, "discovery_method": "fake"},
    }


def check(obj, expected=None):
    return check_runtime_evidence_bytes(raw=canonical_json_bytes(obj), expected_codec=expected or obj["codec"])


@pytest.mark.parametrize("codec", [PROVIDER, MODEL, TOKENIZER])
def test_synthetic_canonical_bytes(codec):
    obj = samples()[codec]
    raw = canonical_json_bytes(obj)
    result = check_runtime_evidence_bytes(raw=raw, expected_codec=codec)
    assert result.codec == codec
    assert result.evidence_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.__slots__ == ("codec", "evidence_sha256")
    assert not callable(result)
    with pytest.raises(FrozenInstanceError):
        result.codec = "changed"


def test_published_provider_vector():
    assert canonical_json_bytes(samples()[PROVIDER]) == VECTOR
    assert check_runtime_evidence_bytes(raw=VECTOR, expected_codec=PROVIDER).evidence_sha256 == (
        "02b11b2ad82bafb54440283064fa884e64084208c83b0e9d01126bb622186558")


def test_noncanonical_key_order():
    raw = json.dumps(samples()[PROVIDER], separators=(",", ":")).encode()
    assert raw != VECTOR
    with pytest.raises(RuntimeEvidenceCodecError):
        check_runtime_evidence_bytes(raw=raw, expected_codec=PROVIDER)


def test_transport_binding_changes_digest_even_with_same_model():
    a = samples()[PROVIDER]
    b = {**a, "transport_binding_ref": "another-synthetic-binding"}
    assert a["request_model"] == b["request_model"]
    assert canonical_json_bytes(a) != canonical_json_bytes(b)
    assert check(a).evidence_sha256 != check(b).evidence_sha256


@pytest.mark.parametrize("codec", [PROVIDER, MODEL, TOKENIZER])
def test_every_field_is_required_and_closed(codec):
    original = samples()[codec]
    for field in original:
        missing = copy.deepcopy(original)
        del missing[field]
        renamed = {**missing, "renamed_field": original[field]}
        for obj in (missing, renamed, {**original, "unknown_field": "fake"}):
            with pytest.raises(RuntimeEvidenceCodecError, match="^invalid runtime evidence$"):
                check_runtime_evidence_bytes(raw=canonical_json_bytes(obj), expected_codec=codec)


@pytest.mark.parametrize("codec", [PROVIDER, MODEL, TOKENIZER])
@pytest.mark.parametrize("version", [True, False, 0, 2, "1", None, [], {}])
def test_bad_version(codec, version):
    obj = samples()[codec]
    obj["schema_version"] = version
    with pytest.raises(RuntimeEvidenceCodecError):
        check(obj)


@pytest.mark.parametrize("codec", [PROVIDER, MODEL, TOKENIZER])
def test_codec_confusion(codec):
    raw = canonical_json_bytes(samples()[codec])
    for expected in (PROVIDER, MODEL, TOKENIZER):
        if expected != codec:
            with pytest.raises(RuntimeEvidenceCodecError):
                check_runtime_evidence_bytes(raw=raw, expected_codec=expected)
    obj = samples()[codec]
    obj["codec"] = "b13.measurement.v1"
    with pytest.raises(RuntimeEvidenceCodecError):
        check_runtime_evidence_bytes(raw=canonical_json_bytes(obj), expected_codec=codec)


@pytest.mark.parametrize("expected", ["", "b13.measurement.v1", "b13.provider-config.v2", None, [], 1])
def test_unknown_expected_codec(expected):
    with pytest.raises(RuntimeEvidenceCodecError, match="^invalid runtime evidence$"):
        check_runtime_evidence_bytes(raw=VECTOR, expected_codec=expected)


@pytest.mark.parametrize("field", ["id", "provider_family", "request_model", "transport_binding_ref"])
@pytest.mark.parametrize("value", ["", "a" * 129, "-bad", "bad space", "é", "a\n", None, 1, True, [], {}])
def test_invalid_provider_ids(field, value):
    obj = samples()[PROVIDER]
    obj[field] = value
    with pytest.raises(RuntimeEvidenceCodecError):
        check(obj)


@pytest.mark.parametrize("codec,field", [(MODEL, "id"), (MODEL, "deployment_ref"),
    (MODEL, "model_revision_ref"), (MODEL, "discovery_method"), (TOKENIZER, "id"),
    (TOKENIZER, "implementation_ref"), (TOKENIZER, "version"), (TOKENIZER, "discovery_method")])
def test_other_id_fields(codec, field):
    obj = samples()[codec]
    obj[field] = "bad id"
    with pytest.raises(RuntimeEvidenceCodecError):
        check(obj)


@pytest.mark.parametrize("field", ["vocabulary_sha256", "special_tokens_sha256", "chat_template_sha256"])
@pytest.mark.parametrize("value", ["A" * 64, "g" * 64, "a" * 63, "a" * 65, None, 1, [], {}])
def test_invalid_hashes(field, value):
    obj = samples()[TOKENIZER]
    obj[field] = value
    with pytest.raises(RuntimeEvidenceCodecError):
        check(obj)


@pytest.mark.parametrize("value", [0, -1, 9007199254740992, True, False, 1.5, "1", None, [], {}])
def test_invalid_context_limit(value):
    obj = samples()[MODEL]
    obj["context_limit_tokens"] = value
    # Encode independently so that out-of-range values reach the checker.
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(RuntimeEvidenceCodecError):
        check_runtime_evidence_bytes(raw=raw, expected_codec=MODEL)


@pytest.mark.parametrize("value", [1, 9007199254740991])
def test_context_limit_boundaries(value):
    obj = samples()[MODEL]
    obj["context_limit_tokens"] = value
    check(obj)


@pytest.mark.parametrize("value", ["", "x" * 257, "é" * 129, "x\n", "x\x80", None, 1, [], {}])
def test_invalid_timeout_string(value):
    obj = samples()[PROVIDER]
    obj["timeout_decimal_seconds"] = value
    with pytest.raises(RuntimeEvidenceCodecError):
        check(obj)


def test_timeout_is_only_a_string_not_numeric_semantics():
    for value in ("not-a-number", "0", "é" * 128):
        obj = samples()[PROVIDER]
        obj["timeout_decimal_seconds"] = value
        check(obj)


def test_id_length_boundaries():
    obj = samples()[PROVIDER]
    for value in ("a", "a" * 128):
        obj["id"] = value
        check(obj)


def test_no_external_discovery_or_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected external access")

    class NoEnvironment:
        def __getitem__(self, key):
            forbidden()

        def get(self, *args):
            forbidden()

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(socket, "socket", forbidden)
        patch.setattr(os, "getenv", forbidden)
        patch.setattr(os, "environ", NoEnvironment())
        check_runtime_evidence_bytes(raw=VECTOR, expected_codec=PROVIDER)


@pytest.mark.parametrize("raw", [
    b'', b' ' * 65537, b'\xff', b'\xef\xbb\xbf{}', b'{"x":"\\ud800"}',
    VECTOR + b'\n', VECTOR + b' ', b' ' + VECTOR,
    VECTOR.replace(b'"schema_version":1', b'"schema_version":1.0'),
    VECTOR.replace(b'"id":"transport.synthetic"', b'"id":"transport.synthetic","id":"secret-fake"'),
    VECTOR.replace(b'"id":"transport.synthetic"', b'"id":[{"nested":"value"}]'),
])
def test_invalid_raw_and_safe_message(raw):
    with pytest.raises(RuntimeEvidenceCodecError, match="^invalid runtime evidence$") as exc:
        check_runtime_evidence_bytes(raw=raw, expected_codec=PROVIDER)
    assert "secret-fake" not in str(exc.value)
    assert exc.value.__suppress_context__ or exc.value.__context__ is None


@pytest.mark.parametrize("raw", [None, "{}", bytearray(VECTOR), memoryview(VECTOR)])
def test_raw_requires_exact_bytes(raw):
    with pytest.raises(RuntimeEvidenceCodecError):
        check_runtime_evidence_bytes(raw=raw, expected_codec=PROVIDER)
