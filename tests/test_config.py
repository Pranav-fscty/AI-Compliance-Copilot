import pytest

from src.config import load_config


def test_string_override_kept_raw(monkeypatch):
    monkeypatch.setenv("RAG_EMBED_QUERY_PREFIX", "search_query: ")  # YAML would parse this as a dict
    monkeypatch.setenv("RAG_TOP_K", "5")
    cfg = load_config()
    assert cfg["embed_query_prefix"] == "search_query: " and cfg["top_k"] == 5


def test_preset_overlays_and_env_wins(monkeypatch):
    assert load_config(preset="nomic")["top_k"] == 5
    monkeypatch.setenv("RAG_TOP_K", "3")
    assert load_config(preset="nomic")["top_k"] == 3


def test_typos_raise(monkeypatch):
    with pytest.raises(FileNotFoundError):
        load_config(preset="no_such_preset")
    monkeypatch.setenv("RAG_TOPK", "3")
    with pytest.raises(ValueError):
        load_config()


@pytest.mark.parametrize("preset", ["default", "minilm", "nomic", "small_llm", "no_reranker"])
def test_all_presets_load(preset):
    assert load_config(preset=preset)["collection"]


def test_reranker_reorders_and_cuts_to_k(monkeypatch):
    from langchain_core.documents import Document
    from src import rag_chain as rc

    docs = [Document(page_content=t) for t in ("a", "bb", "ccc")]
    monkeypatch.setitem(rc.CFG, "reranker", "fake")
    monkeypatch.setattr(rc, "get_db", lambda: type("DB", (), {"similarity_search": lambda self, q, k, filter: docs[:k]})())
    monkeypatch.setattr(rc, "get_reranker", lambda: type("R", (), {"predict": lambda self, pairs: [len(d) for _, d in pairs]})())
    assert [d.page_content for d in rc.retrieve("q", k=2)] == ["ccc", "bb"]  # fetched 3 (fetch_k), kept best 2
