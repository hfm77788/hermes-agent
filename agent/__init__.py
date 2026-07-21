"""Agent internals -- extracted modules from run_agent.py.

These modules contain pure utility functions and self-contained classes
that were previously embedded in the 3,600-line run_agent.py. Extracting
them makes run_agent.py focused on the AIAgent orchestrator class.
"""

from __future__ import annotations

from typing import Any, Optional

from . import jiter_preload as _jiter_preload  # noqa: F401


def _install_credential_pool_provider_matcher() -> None:
    """Restore the provider-boundary helper missing from partial syncs.

    The current runtime callers import this helper from ``agent.credential_pool``.
    Keep the compatibility local to the ``agent`` package and install it only
    when the canonical module does not already provide an implementation.

    Minimal wheel consumers such as ``from agent import i18n`` intentionally
    install without the full runtime dependency set. In that environment the
    credential-pool import is skipped, preserving the package's existing
    lightweight-import contract.
    """

    try:
        from . import credential_pool
    except ImportError:
        return

    if callable(getattr(credential_pool, "credential_pool_matches_provider", None)):
        return

    def credential_pool_matches_provider(
        pool_or_provider: Any,
        provider: Optional[str],
        *,
        base_url: Optional[str] = None,
    ) -> bool:
        raw_pool_provider = getattr(pool_or_provider, "provider", None)
        if raw_pool_provider is None:
            if isinstance(pool_or_provider, str):
                raw_pool_provider = pool_or_provider
            else:
                # Legacy lightweight adapters may not expose provider scope.
                return True

        pool_provider = str(raw_pool_provider or "").strip().lower()
        provider_norm = str(provider or "").strip().lower()
        if not pool_provider or not provider_norm:
            return False
        if pool_provider == provider_norm:
            return True
        if provider_norm != "custom" or not pool_provider.startswith(
            credential_pool.CUSTOM_POOL_PREFIX
        ):
            return False

        try:
            matched_pool = credential_pool.get_custom_provider_pool_key(base_url or "")
        except Exception:
            return False
        return str(matched_pool or "").strip().lower() == pool_provider

    credential_pool.credential_pool_matches_provider = credential_pool_matches_provider


_install_credential_pool_provider_matcher()
