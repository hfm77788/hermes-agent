from __future__ import annotations

from agent.skill_retrieval import build_skill_retrieval_context, retrieve_skills


def _entry(name: str, description: str, *, category: str = "general", semantic_terms=None):
    return {
        "name": name,
        "category": category,
        "description": description,
        "semantic_terms": list(semantic_terms or []),
    }


def test_explicit_skill_name_has_priority():
    catalog = [
        _entry("wps-data-workflow", "Build data query forms and structured WPS workflows"),
        _entry("generic-forms", "Forms surveys spreadsheets and data collection"),
    ]
    hits = retrieve_skills("use wps-data-workflow for this", catalog, top_k=2)
    assert hits[0]["name"] == "wps-data-workflow"
    assert hits[0]["exact_score"] >= 0.95


def test_short_cjk_query_uses_retrieval_metadata():
    catalog = [
        _entry(
            "wps-form-query",
            "Build governed WPS data collection and lookup workflows",
            semantic_terms=["WPS 表单", "问卷", "数据收集", "多维查询"],
        ),
        _entry("deploy-helper", "Deploy and verify a server runtime"),
    ]
    hits = retrieve_skills("表单", catalog, top_k=2)
    assert hits
    assert hits[0]["name"] == "wps-form-query"
    assert hits[0]["lexical_score"] > 0


def test_lsa_semantic_channel_recovers_related_candidate():
    catalog = [
        _entry("form-builder", "spreadsheet forms survey workflow data collection", category="office"),
        _entry("questionnaire-helper", "questionnaire survey workflow audience research", category="office"),
        _entry("deploy-helper", "docker server deployment runtime health", category="devops"),
        _entry("research-helper", "audience research evidence sources", category="research"),
    ]
    hits = retrieve_skills("audience questionnaire", catalog, top_k=4)
    by_name = {hit["name"]: hit for hit in hits}
    assert "research-helper" in by_name
    assert by_name["research-helper"]["semantic_score"] > 0


def test_rendered_context_is_bounded_and_requires_skill_view():
    catalog = [_entry(f"skill-{i}", f"shared workflow topic {i}") for i in range(20)]
    context = build_skill_retrieval_context("shared workflow", catalog, top_k=5)
    assert "<relevant_skills>" in context
    assert "skill_view(name)" in context
    assert sum(1 for line in context.splitlines() if line.startswith("- skill-")) <= 5


def test_generic_name_mention_does_not_steal_exact_priority():
    catalog = [
        _entry("github", "GitHub via gh CLI: PRs, issues, reviews, repos, auth."),
        _entry("github-code-review", "Review PR diffs and code changes safely"),
    ]
    hits = retrieve_skills("GitHub PR code review", catalog, top_k=2)
    assert hits[0]["name"] == "github-code-review"
    assert next(hit for hit in hits if hit["name"] == "github")["exact_score"] == 0


def test_non_collision_duplicate_names_are_collapsed():
    catalog = [
        _entry("wps-query-app", "WPS forms query workflow", category="productivity"),
        _entry("wps-query-app", "WPS forms query workflow", category="wps-query-app"),
        _entry("other", "unrelated runtime", category="devops"),
    ]
    hits = retrieve_skills("WPS forms query", catalog, top_k=5)
    assert [hit["name"] for hit in hits].count("wps-query-app") == 1


def test_lexical_fallback_when_semantic_index_unavailable(monkeypatch):
    import agent.skill_retrieval as sr

    catalog = [
        _entry("wps-query-app", "WPS forms query workflow"),
        _entry("deploy-helper", "server deployment runtime"),
    ]
    sr.clear_skill_retrieval_cache()
    monkeypatch.setattr(sr, "_build_semantic_state", lambda _catalog: None)
    hits = sr.retrieve_skills("WPS forms", catalog, top_k=2)
    assert hits
    assert hits[0]["name"] == "wps-query-app"
    assert hits[0]["semantic_score"] == 0


def test_resumed_turn_can_reconstruct_missing_catalog(monkeypatch):
    from types import SimpleNamespace
    from agent.turn_context import _skill_turn_retrieval
    import agent.system_prompt as system_prompt

    agent = SimpleNamespace(_skill_retrieval_catalog=[])
    catalog = [_entry("wps-query-app", "WPS forms query workflow")]

    def fake_skills_prompt(target):
        target._skill_retrieval_catalog = catalog
        return "names-only"

    monkeypatch.setattr(system_prompt, "_skills_prompt", fake_skills_prompt)
    context = _skill_turn_retrieval(agent, "WPS forms")
    assert "wps-query-app" in context


def test_compound_skill_name_emits_component_tokens():
    catalog = [_entry("pdf-tools", "Manipulate documents")]
    hits = retrieve_skills("use PDF", catalog, top_k=2)
    assert hits
    assert hits[0]["name"] == "pdf-tools"
    assert hits[0]["lexical_score"] > 0


def test_long_verbose_query_keeps_single_decisive_lexical_match():
    catalog = [_entry("doc-handler", "Convert PDF files")]
    query = (
        "Please help me prepare a careful workflow for this customer request with several constraints, "
        "review steps, naming rules, archival notes, and finally convert the attached PDF before delivery"
    )
    hits = retrieve_skills(query, catalog, top_k=2)
    assert hits
    assert hits[0]["name"] == "doc-handler"
    assert hits[0]["lexical_score"] >= 0.08
