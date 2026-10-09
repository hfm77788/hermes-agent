"""Bounded per-turn Skill retrieval.

The system prompt keeps the complete skill *name* catalog for discoverability while
this module selects a small description-bearing shortlist for the current user turn.
Retrieval is deliberately local and fail-open: exact/name and lexical channels are
always available; an optional NumPy LSA channel adds lightweight semantic recall
without a model call or network dependency.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import unicodedata
from collections import Counter, OrderedDict
from typing import Any, Iterable

_ASCII_WORD_RE = re.compile(r"[a-z0-9]+(?:[._+/-][a-z0-9]+)*")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_SPLIT_RE = re.compile(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+")
_CACHE_MAX = 8
_SEMANTIC_VOCAB_MAX = 256
_SEMANTIC_RANK_MAX = 24
_RRF_K = 40.0

_SEMANTIC_CACHE: "OrderedDict[str, dict[str, Any] | None]" = OrderedDict()
_LEXICAL_CACHE: "OrderedDict[str, tuple[tuple[set[str], set[str], set[str]], ...]]" = OrderedDict()
_SEMANTIC_CACHE_LOCK = threading.Lock()


def _normalize(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).casefold().strip()


def _compact(text: Any) -> str:
    return _SPLIT_RE.sub("", _normalize(text))


def _tokens(text: Any) -> set[str]:
    """Language-light tokens: ASCII words plus bounded CJK n-grams.

    CJK bigrams preserve short Chinese queries (for example ``表单``) without a
    tokenizer dependency. Whole short runs stay available for exact-ish matches.
    """
    norm = _normalize(text)
    compounds = set(_ASCII_WORD_RE.findall(norm))
    out = set(compounds)
    for token in compounds:
        if any(sep in token for sep in "-_/+."):
            out.update(part for part in re.split(r"[-_/+.]+", token) if part)
    for run in _CJK_RUN_RE.findall(norm):
        if len(run) <= 8:
            out.add(run)
        if len(run) == 1:
            out.add(run)
            continue
        out.update(run[i : i + 2] for i in range(len(run) - 1))
        if len(run) >= 3:
            out.update(run[i : i + 3] for i in range(len(run) - 2))
    return {token for token in out if token}


def _semantic_terms(entry: dict[str, Any]) -> str:
    raw = entry.get("semantic_terms") or []
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (list, tuple, set)):
        return " ".join(str(item) for item in raw if str(item).strip())
    return ""


def _document_text(entry: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(entry.get("name") or ""),
            str(entry.get("category") or ""),
            str(entry.get("description") or ""),
            _semantic_terms(entry),
        )
        if part
    )


def _catalog_fingerprint(catalog: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for entry in catalog:
        for key in ("name", "category", "description"):
            h.update(str(entry.get(key) or "").encode("utf-8", "replace"))
            h.update(b"\0")
        h.update(_semantic_terms(entry).encode("utf-8", "replace"))
        h.update(b"\xff")
    return h.hexdigest()


def _build_semantic_state(catalog: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Build a small latent-semantic index, or ``None`` when NumPy is absent.

    Only terms present in at least two documents enter LSA; one-off identifiers
    remain the job of exact/lexical recall. The matrix is intentionally bounded.
    """
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    if len(catalog) < 3:
        return None

    docs = [_tokens(_document_text(entry)) for entry in catalog]
    df = Counter(token for tokens in docs for token in tokens)
    n_docs = len(docs)
    eligible = [
        token for token, count in df.items()
        if 2 <= count <= max(2, int(n_docs * 0.85))
    ]
    if len(eligible) < 2:
        return None
    eligible.sort(key=lambda token: (-(math.log((n_docs + 1) / (df[token] + 1)) + 1.0) * math.sqrt(df[token]), token))
    vocab_terms = eligible[:_SEMANTIC_VOCAB_MAX]
    vocab = {token: idx for idx, token in enumerate(vocab_terms)}
    idf = np.array([math.log((n_docs + 1) / (df[token] + 1)) + 1.0 for token in vocab_terms], dtype=float)
    matrix = np.zeros((n_docs, len(vocab_terms)), dtype=float)
    for row, tokens in enumerate(docs):
        for token in tokens:
            col = vocab.get(token)
            if col is not None:
                matrix[row, col] = idf[col]
    try:
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    except Exception:
        return None
    rank = min(_SEMANTIC_RANK_MAX, len(s), max(1, len(vocab_terms) - 1), max(1, n_docs - 1))
    if rank < 1:
        return None
    doc_vectors = u[:, :rank] * s[:rank]
    norms = np.linalg.norm(doc_vectors, axis=1)
    norms[norms == 0] = 1.0
    doc_vectors = doc_vectors / norms[:, None]
    return {"np": np, "vocab": vocab, "idf": idf, "vt": vt[:rank, :], "docs": doc_vectors}


def _semantic_state(catalog: list[dict[str, Any]]) -> dict[str, Any] | None:
    key = _catalog_fingerprint(catalog)
    with _SEMANTIC_CACHE_LOCK:
        if key in _SEMANTIC_CACHE:
            state = _SEMANTIC_CACHE.pop(key)
            _SEMANTIC_CACHE[key] = state
            return state
    state = _build_semantic_state(catalog)
    with _SEMANTIC_CACHE_LOCK:
        _SEMANTIC_CACHE[key] = state
        while len(_SEMANTIC_CACHE) > _CACHE_MAX:
            _SEMANTIC_CACHE.popitem(last=False)
    return state


def _lexical_state(catalog: list[dict[str, Any]]) -> tuple[tuple[set[str], set[str], set[str]], ...]:
    key = _catalog_fingerprint(catalog)
    with _SEMANTIC_CACHE_LOCK:
        cached = _LEXICAL_CACHE.get(key)
        if cached is not None:
            _LEXICAL_CACHE.move_to_end(key)
            return cached
    state = tuple(
        (
            _tokens(entry.get("name")),
            _tokens(_semantic_terms(entry)),
            _tokens(_document_text(entry)),
        )
        for entry in catalog
    )
    with _SEMANTIC_CACHE_LOCK:
        _LEXICAL_CACHE[key] = state
        _LEXICAL_CACHE.move_to_end(key)
        while len(_LEXICAL_CACHE) > _CACHE_MAX:
            _LEXICAL_CACHE.popitem(last=False)
    return state


def clear_skill_retrieval_cache() -> None:
    with _SEMANTIC_CACHE_LOCK:
        _SEMANTIC_CACHE.clear()
        _LEXICAL_CACHE.clear()


def _semantic_scores(query: str, catalog: list[dict[str, Any]]) -> list[float]:
    state = _semantic_state(catalog)
    if not state:
        return [0.0] * len(catalog)
    np = state["np"]
    q = np.zeros(len(state["vocab"]), dtype=float)
    for token in _tokens(query):
        idx = state["vocab"].get(token)
        if idx is not None:
            q[idx] = state["idf"][idx]
    if not bool(np.any(q)):
        return [0.0] * len(catalog)
    latent = q @ state["vt"].T
    norm = float(np.linalg.norm(latent))
    if norm <= 0:
        return [0.0] * len(catalog)
    latent = latent / norm
    scores = state["docs"] @ latent
    return [max(0.0, float(score)) for score in scores]


def _exact_score(query: str, entry: dict[str, Any]) -> float:
    q_compact = _compact(query)
    name_raw = _normalize(entry.get("name"))
    name_compact = _compact(name_raw)
    if not q_compact or not name_compact:
        return 0.0
    if q_compact == name_compact:
        return 1.0
    q_raw = _normalize(query)
    # Natural-language mentions of a generic name (for example "GitHub PR review")
    # are not explicit invocations. Slug-shaped names and command-like forms are.
    explicit_command = f"/{name_raw}" in q_raw or f"skill:{name_raw}" in q_raw or f"skill {name_raw}" in q_raw
    slug_phrase = re.sub(r"[-_/.:]+", " ", name_raw).strip()
    slug_shaped = any(ch in name_raw for ch in "-_/:") and (name_raw in q_raw or slug_phrase in q_raw)
    if explicit_command or slug_shaped:
        return 0.95
    return 0.0


def _lexical_score_tokens(
    q_tokens: set[str], name_tokens: set[str], semantic_tokens: set[str], doc_tokens: set[str]
) -> float:
    if not q_tokens:
        return 0.0

    def cosine_overlap(target_tokens: set[str]) -> float:
        if not target_tokens:
            return 0.0
        matched = len(q_tokens & target_tokens)
        return matched / math.sqrt(len(q_tokens) * len(target_tokens)) if matched else 0.0

    overlap = cosine_overlap(doc_tokens)
    name_overlap = cosine_overlap(name_tokens)
    metadata_overlap = cosine_overlap(semantic_tokens)
    return min(1.0, overlap + 0.65 * name_overlap + 0.45 * metadata_overlap)


def _lexical_score(query: str, entry: dict[str, Any]) -> float:
    q_tokens = _tokens(query)
    return _lexical_score_tokens(
        q_tokens, _tokens(entry.get("name")), _tokens(_semantic_terms(entry)), _tokens(_document_text(entry))
    )


def _rank_map(scores: Iterable[float], minimum: float) -> dict[int, int]:
    ranked = sorted(
        ((idx, float(score)) for idx, score in enumerate(scores) if float(score) >= minimum),
        key=lambda item: (-item[1], item[0]),
    )
    return {idx: rank for rank, (idx, _score) in enumerate(ranked, start=1)}


def retrieve_skills(query: str, catalog: list[dict[str, Any]], *, top_k: int = 8) -> list[dict[str, Any]]:
    """Return a bounded exact + lexical + semantic fusion shortlist."""
    if not isinstance(query, str) or not query.strip() or not catalog or top_k <= 0:
        return []
    exact = [_exact_score(query, entry) for entry in catalog]
    q_tokens = _tokens(query)
    lexical = [
        _lexical_score_tokens(q_tokens, name_tokens, semantic_tokens, doc_tokens)
        for name_tokens, semantic_tokens, doc_tokens in _lexical_state(catalog)
    ]
    semantic = _semantic_scores(query, catalog)
    exact_ranks = _rank_map(exact, 0.8)
    lexical_ranks = _rank_map(lexical, 0.08)
    semantic_ranks = _rank_map(semantic, 0.12)

    results: list[dict[str, Any]] = []
    for idx, entry in enumerate(catalog):
        if idx not in exact_ranks and idx not in lexical_ranks and idx not in semantic_ranks:
            continue
        fused = 0.0
        if idx in exact_ranks:
            fused += 4.0 / (_RRF_K + exact_ranks[idx])
        if idx in lexical_ranks:
            fused += 4.0 / (_RRF_K + lexical_ranks[idx])
        if idx in semantic_ranks:
            fused += 0.5 / (_RRF_K + semantic_ranks[idx])
        results.append({
            **entry,
            "exact_score": exact[idx],
            "lexical_score": lexical[idx],
            "semantic_score": semantic[idx],
            "fusion_score": fused,
        })
    results.sort(
        key=lambda item: (
            -float(item["exact_score"]),
            -float(item["fusion_score"]),
            -float(item["lexical_score"]),
            -float(item["semantic_score"]),
            str(item.get("name") or ""),
        )
    )
    # Historical duplicate copies can expose the same canonical skill name from
    # two categories. Collapse them in retrieval unless the prompt explicitly
    # marked a personal/org name collision, where category disambiguation matters.
    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for item in results:
        name_key = _normalize(item.get("name"))
        collision = "name collision" in str(item.get("description") or "").casefold()
        if name_key in seen_names and not collision:
            continue
        seen_names.add(name_key)
        deduped.append(item)
        if len(deduped) >= min(int(top_k), 12):
            break
    return deduped


def build_skill_retrieval_context(query: str, catalog: list[dict[str, Any]], *, top_k: int = 8) -> str:
    """Render the ephemeral shortlist injected behind the current user message."""
    hits = retrieve_skills(query, catalog, top_k=top_k)
    if not hits:
        return ""
    lines = [
        "<relevant_skills>",
        "Retrieved candidates for this turn only; these are not loaded instructions. "
        "Load a relevant candidate with skill_view(name) before acting. Explicit user skill invocation always wins.",
    ]
    for item in hits:
        name = str(item.get("name") or "").strip()
        category = str(item.get("category") or "general").strip() or "general"
        description = str(item.get("description") or "").strip()
        lines.append(f"- {name} [{category}]: {description}" if description else f"- {name} [{category}]")
    lines.append("</relevant_skills>")
    return "\n".join(lines)
