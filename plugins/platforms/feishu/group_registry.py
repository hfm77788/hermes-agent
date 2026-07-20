"""Group Registry Loader — single source of truth for memory_catalog resolution.

Reads group-registry.yaml, validates schema, and provides exact-lookup APIs:
  - resolve_sender(open_id, chat_id) → SenderIdentity
  - resolve_bank(catalog_key) → bank_id

Design principles:
  - Fail-closed: unknown sender, unknown catalog_key, duplicate mapping,
    corrupt YAML → structured error, never silent fallback.
  - Only registered target groups are affected; unregistered chats pass through.
  - No hardcoded bank names anywhere.
  - Safe reload: mtime-based cache invalidation (restart-free for config edits).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ── Structured error types ──────────────────────────────────────────────────

class RegistryError(Exception):
    """Base error for group registry failures."""
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"[{code}] {detail}")


class RegistryLoadError(RegistryError):
    """YAML file missing, unreadable, or structurally invalid."""
    pass


class CatalogKeyError(RegistryError):
    """Unknown or missing catalog key."""
    pass


class SenderResolutionError(RegistryError):
    """Sender cannot be resolved in the given chat."""
    pass


class DuplicateMappingError(RegistryError):
    """Duplicate open_id or catalog key detected."""
    pass


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CatalogEntry:
    key: str
    bank_id: str
    role: str
    status: str  # current | readonly | quarantined | deprecated
    write_scope: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SenderIdentity:
    open_id: str
    role: str  # boss | employee
    display_name: str
    catalog_key: str  # → memory_catalog key
    profile_model: str
    chat_id: str


@dataclass
class RegistrySnapshot:
    """Immutable validated snapshot of the registry."""
    catalog: Dict[str, CatalogEntry] = field(default_factory=dict)
    # chat_id → {open_id → SenderIdentity}
    sender_index: Dict[str, Dict[str, SenderIdentity]] = field(default_factory=dict)
    registered_chat_ids: Set[str] = field(default_factory=set)
    raw: Dict[str, Any] = field(default_factory=dict)


# ── Loader ──────────────────────────────────────────────────────────────────

_REGISTRY_RELATIVE_PATH = (
    Path("projects") / "ma-secretary-interaction-system" / "group-registry.yaml"
)

_REQUIRED_WORKSPACE_FIELDS = [
    "chat_id", "chat_name", "employee_name", "employee_open_id",
    "boss_open_id", "boss_name", "privacy_boundary", "status",
]

_VALID_CATALOG_STATUSES = {"current", "readonly", "quarantined", "deprecated"}


class GroupRegistryLoader:
    """Thread-safe, mtime-cached registry loader with schema validation."""

    def __init__(self, registry_path: Optional[Path | str] = None):
        self._path = Path(registry_path) if registry_path else get_hermes_home() / _REGISTRY_RELATIVE_PATH
        self._cache: Optional[Tuple[float, RegistrySnapshot]] = None

    # ── Public API ──────────────────────────────────────────────────────

    def is_registered_chat(self, chat_id: str) -> bool:
        """Return True only if chat_id is a registered target group."""
        if not chat_id:
            return False
        snap = self._get_snapshot()
        return chat_id in snap.registered_chat_ids

    def resolve_sender(self, open_id: str, chat_id: str) -> SenderIdentity:
        """Exact-lookup sender identity. Fail-closed on unknown."""
        if not open_id or not chat_id:
            raise SenderResolutionError(
                "SENDER_MISSING_INPUT",
                f"open_id={open_id!r} chat_id={chat_id!r}",
            )
        snap = self._get_snapshot()
        chat_senders = snap.sender_index.get(chat_id)
        if chat_senders is None:
            raise SenderResolutionError(
                "CHAT_NOT_REGISTERED",
                f"chat_id={chat_id!r} is not a registered target group",
            )
        identity = chat_senders.get(open_id)
        if identity is None:
            raise SenderResolutionError(
                "SENDER_UNKNOWN",
                f"open_id={open_id!r} not found in chat_id={chat_id!r}",
            )
        return identity

    def resolve_bank(self, catalog_key: str) -> str:
        """Exact-lookup bank_id from memory_catalog. Fail-closed on unknown."""
        if not catalog_key:
            raise CatalogKeyError("CATALOG_KEY_EMPTY", "catalog_key is empty")
        snap = self._get_snapshot()
        entry = snap.catalog.get(catalog_key)
        if entry is None:
            raise CatalogKeyError(
                "CATALOG_KEY_UNKNOWN",
                f"catalog_key={catalog_key!r} not in memory_catalog; "
                f"available: {sorted(snap.catalog.keys())}",
            )
        return entry.bank_id

    def resolve_bank_for_sender(self, open_id: str, chat_id: str) -> str:
        """Convenience: resolve sender → catalog_key → bank_id."""
        identity = self.resolve_sender(open_id, chat_id)
        return self.resolve_bank(identity.catalog_key)

    def get_memory_catalog(self) -> Dict[str, Any]:
        """Return the full memory_catalog dict. Fail-closed if missing."""
        snap = self._get_snapshot()
        catalog = snap.raw.get("memory_catalog")
        if not isinstance(catalog, dict) or not catalog:
            raise RegistryLoadError(
                "MEMORY_CATALOG_MISSING",
                "memory_catalog section missing or empty in registry",
            )
        return catalog

    def get_catalog_entry(self, catalog_key: str) -> CatalogEntry:
        """Return full catalog entry. Fail-closed."""
        if not catalog_key:
            raise CatalogKeyError("CATALOG_KEY_EMPTY", "catalog_key is empty")
        snap = self._get_snapshot()
        entry = snap.catalog.get(catalog_key)
        if entry is None:
            raise CatalogKeyError(
                "CATALOG_KEY_UNKNOWN",
                f"catalog_key={catalog_key!r} not in memory_catalog",
            )
        return entry

    def get_workspace(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Return raw workspace dict for a registered chat, or None."""
        snap = self._get_snapshot()
        workspaces = snap.raw.get("workspaces")
        if not isinstance(workspaces, list):
            return None
        for ws in workspaces:
            if isinstance(ws, dict) and str(ws.get("chat_id") or "").strip() == chat_id:
                return ws
        return None

    def invalidate_cache(self) -> None:
        """Force reload on next access (for tests or explicit refresh)."""
        self._cache = None

    # ── Internal ────────────────────────────────────────────────────────

    def _get_snapshot(self) -> RegistrySnapshot:
        """Load + validate with mtime cache."""
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            raise RegistryLoadError(
                "REGISTRY_FILE_NOT_FOUND",
                f"Registry file not found: {self._path}",
            )

        if self._cache is not None and self._cache[0] == stat.st_mtime:
            return self._cache[1]

        snap = self._load_and_validate()
        self._cache = (stat.st_mtime, snap)
        return snap

    def _load_and_validate(self) -> RegistrySnapshot:
        """Parse YAML, validate schema, build indexes. Fail-closed on any error."""
        try:
            text = self._path.read_text(encoding="utf-8")
        except Exception as exc:
            raise RegistryLoadError("REGISTRY_READ_ERROR", str(exc))

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise RegistryLoadError("REGISTRY_YAML_PARSE_ERROR", str(exc))

        if not isinstance(data, dict):
            raise RegistryLoadError(
                "REGISTRY_NOT_DICT",
                f"Top-level YAML is {type(data).__name__}, expected dict",
            )

        # ── Validate memory_catalog ──
        raw_catalog = data.get("memory_catalog")
        if not isinstance(raw_catalog, dict) or not raw_catalog:
            raise RegistryLoadError(
                "CATALOG_MISSING",
                "memory_catalog is missing or empty",
            )

        catalog: Dict[str, CatalogEntry] = {}
        seen_bank_ids: Dict[str, str] = {}  # bank_id → first catalog_key

        for key, entry in raw_catalog.items():
            if not isinstance(entry, dict):
                raise RegistryLoadError(
                    "CATALOG_ENTRY_INVALID",
                    f"memory_catalog.{key} is not a dict",
                )
            bank_id = str(entry.get("bank_id") or "").strip()
            if not bank_id:
                raise RegistryLoadError(
                    "CATALOG_BANK_ID_MISSING",
                    f"memory_catalog.{key}.bank_id is empty",
                )
            status = str(entry.get("status") or "").strip()
            if status not in _VALID_CATALOG_STATUSES:
                raise RegistryLoadError(
                    "CATALOG_STATUS_INVALID",
                    f"memory_catalog.{key}.status={status!r} not in {_VALID_CATALOG_STATUSES}",
                )
            # Duplicate bank_id across different catalog keys → fail-closed
            if bank_id in seen_bank_ids:
                raise DuplicateMappingError(
                    "DUPLICATE_BANK_ID",
                    f"bank_id={bank_id!r} mapped to both "
                    f"{seen_bank_ids[bank_id]!r} and {key!r}",
                )
            seen_bank_ids[bank_id] = key
            write_scope = tuple(entry.get("write_scope") or ())
            catalog[key] = CatalogEntry(
                key=key,
                bank_id=bank_id,
                role=str(entry.get("role") or ""),
                status=status,
                write_scope=write_scope,
            )

        # ── Validate workspaces & build sender index ──
        raw_workspaces = data.get("workspaces")
        if not isinstance(raw_workspaces, list):
            raise RegistryLoadError(
                "WORKSPACES_MISSING",
                "workspaces is missing or not a list",
            )

        sender_index: Dict[str, Dict[str, SenderIdentity]] = {}
        registered_chat_ids: Set[str] = set()
        seen_chat_ids: Set[str] = set()

        for ws in raw_workspaces:
            if not isinstance(ws, dict):
                raise RegistryLoadError(
                    "WORKSPACE_NOT_DICT",
                    f"workspace entry is {type(ws).__name__}",
                )
            # Required fields
            missing = [f for f in _REQUIRED_WORKSPACE_FIELDS if not str(ws.get(f) or "").strip()]
            if missing:
                raise RegistryLoadError(
                    "WORKSPACE_MISSING_FIELDS",
                    f"workspace missing required fields: {missing}",
                )

            chat_id = str(ws["chat_id"]).strip()
            if chat_id in seen_chat_ids:
                raise DuplicateMappingError(
                    "DUPLICATE_CHAT_ID",
                    f"chat_id={chat_id!r} appears in multiple workspaces",
                )
            seen_chat_ids.add(chat_id)

            # Only index active workspaces
            status = str(ws.get("status") or "").strip()
            if status != "active":
                continue

            registered_chat_ids.add(chat_id)
            chat_senders: Dict[str, SenderIdentity] = {}

            # Boss
            boss_open_id = str(ws["boss_open_id"]).strip()
            boss_catalog_key = str(ws.get("boss_profile_catalog") or "").strip()
            if not boss_catalog_key:
                raise RegistryLoadError(
                    "BOSS_CATALOG_KEY_MISSING",
                    f"workspace {chat_id}: boss_profile_catalog is empty",
                )
            if boss_catalog_key not in catalog:
                raise CatalogKeyError(
                    "BOSS_CATALOG_KEY_UNKNOWN",
                    f"workspace {chat_id}: boss_profile_catalog={boss_catalog_key!r} "
                    f"not in memory_catalog",
                )
            boss_identity = SenderIdentity(
                open_id=boss_open_id,
                role="boss",
                display_name=str(ws.get("boss_name") or "老板").strip(),
                catalog_key=boss_catalog_key,
                profile_model=str(ws.get("boss_profile_model") or "").strip(),
                chat_id=chat_id,
            )

            # Employee
            emp_open_id = str(ws["employee_open_id"]).strip()
            emp_catalog_key = str(ws.get("employee_profile_catalog") or "").strip()
            if not emp_catalog_key:
                raise RegistryLoadError(
                    "EMPLOYEE_CATALOG_KEY_MISSING",
                    f"workspace {chat_id}: employee_profile_catalog is empty",
                )
            if emp_catalog_key not in catalog:
                raise CatalogKeyError(
                    "EMPLOYEE_CATALOG_KEY_UNKNOWN",
                    f"workspace {chat_id}: employee_profile_catalog={emp_catalog_key!r} "
                    f"not in memory_catalog",
                )
            emp_identity = SenderIdentity(
                open_id=emp_open_id,
                role="employee",
                display_name=str(ws.get("employee_name") or "").strip(),
                catalog_key=emp_catalog_key,
                profile_model=str(ws.get("employee_profile_model") or "").strip(),
                chat_id=chat_id,
            )

            # Duplicate open_id within same chat → fail-closed
            if boss_open_id == emp_open_id:
                raise DuplicateMappingError(
                    "DUPLICATE_OPEN_ID_IN_CHAT",
                    f"chat_id={chat_id}: boss and employee share open_id={boss_open_id!r}",
                )

            chat_senders[boss_open_id] = boss_identity
            chat_senders[emp_open_id] = emp_identity
            sender_index[chat_id] = chat_senders

        return RegistrySnapshot(
            catalog=catalog,
            sender_index=sender_index,
            registered_chat_ids=registered_chat_ids,
            raw=data,
        )


# ── Module-level singleton ──────────────────────────────────────────────────

_default_loader: Optional[GroupRegistryLoader] = None


def get_registry_loader() -> GroupRegistryLoader:
    """Get or create the module-level singleton loader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = GroupRegistryLoader()
    return _default_loader