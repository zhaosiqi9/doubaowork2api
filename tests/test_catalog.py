from __future__ import annotations

from doubao_work2api.catalog import (
    MODEL_CATALOG,
    find_model,
    normalize_model,
    resolve_upstream_key,
)
from doubao_work2api.errors import UnknownModelError


def test_catalog_public_ids_are_unique() -> None:
    ids = [item.id for item in MODEL_CATALOG]
    assert len(ids) == len(set(ids)) == 8


def test_aliases_resolve_to_public_model() -> None:
    assert normalize_model("DeepSeek-V4-1-Flash") == "deepseek-v4.1-flash"
    assert normalize_model("kimi-k3.1-pro") == "person-kimi-k3-1-pro-261111"
    assert normalize_model("glm-5.3-flash") == "person-glm-5-3-flash-261111"


def test_upstream_key_follows_requested_model() -> None:
    assert resolve_upstream_key(
        "person-kimi-k3-1-pro-261111"
    ) == "person-kimi-k3-1-pro-261111"
    assert (
        resolve_upstream_key("deepseek-v4-pro")
        == "person-deepseek-v4-pro-ga-260813"
    )


def test_unknown_model_is_rejected() -> None:
    try:
        resolve_upstream_key("unknown-model")
    except UnknownModelError as exc:
        assert exc.status_code == 400
        assert exc.code == "unknown_model"
    else:
        raise AssertionError("unknown model must not fall back to another model")


def test_catalog_contains_observed_model_keys() -> None:
    assert find_model("seed-lite-7b") is not None
    assert find_model("person-kimi-k3-261111") is not None
    assert find_model("person-glm-5-3-261111") is not None
