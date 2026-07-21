from __future__ import annotations

import importlib

import agent.credential_pool as credential_pool


def _reload_compat() -> None:
    module = importlib.import_module("sitecustomize")
    importlib.reload(module)


def test_sitecustomize_installs_missing_provider_matcher(monkeypatch):
    monkeypatch.delattr(
        credential_pool,
        "credential_pool_matches_provider",
        raising=False,
    )

    _reload_compat()

    matcher = credential_pool.credential_pool_matches_provider
    assert matcher("deepseek", "deepseek")
    assert not matcher("openai-codex", "deepseek")
    assert not matcher("", "deepseek")


def test_existing_provider_matcher_is_not_overwritten(monkeypatch):
    sentinel = lambda *_args, **_kwargs: "sentinel"  # noqa: E731
    monkeypatch.setattr(
        credential_pool,
        "credential_pool_matches_provider",
        sentinel,
        raising=False,
    )

    _reload_compat()

    assert credential_pool.credential_pool_matches_provider is sentinel
