"""Bootstrap compatibility for a partially synchronized credential-pool runtime.

The gateway imports ``credential_pool_matches_provider`` from
``agent.credential_pool``.  Some deployments received the callers before the
helper itself, which makes the gateway crash during module import.  Python loads
``sitecustomize`` during interpreter startup, so install the upstream-compatible
helper only when the real module does not already provide it.

This shim is deliberately narrow and self-disabling.  It can be removed once
``agent/credential_pool.py`` contains the canonical helper.
"""

from __future__ import annotations

from typing import Any, Optional


def _install_credential_pool_provider_matcher() -> None:
    try:
        import agent.credential_pool as credential_pool
    except Exception:
        # Never make interpreter startup fail because the optional compatibility
        # bridge could not import its target module.
        return

    if callable(getattr(credential_pool, "credential_pool_matches_provider", None)):
        return

    def credential_pool_matches_provider(
        pool_or_provider: Any,
        provider: Optional[str],
        *,
        base_url: Optional[str] = None,
    ) -> bool:
        """Return whether a credential pool belongs to the runtime provider."""

        raw_pool_provider = getattr(pool_or_provider, "provider", None)
        if raw_pool_provider is None:
            if isinstance(pool_or_provider, str):
                raw_pool_provider = pool_or_provider
            else:
                # Preserve compatibility with lightweight pool adapters that
                # predate provider scoping. Production pools expose provider.
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
