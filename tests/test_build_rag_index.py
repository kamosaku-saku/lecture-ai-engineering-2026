"""実SQLiteと固定ベクトルで、索引作成から検索までを確認する。APIは呼ばない。"""

import json
from importlib import reload
from unittest.mock import Mock

import pytest

from app import config, retrieval
from scripts.build_rag_index import build_index


@pytest.fixture
def materials(tmp_path):
    source = tmp_path / "lectures.json"
    source.write_text(json.dumps([
        {"id": f"lec{n:02d}-main", "title": f"講義{n}", "content": [
            {"section": "章", "content": [
                {"id": f"lec{n:02d}-main-p001", "title": f"ページ{n}",
                 "page": 1, "content": f"本文{n}"}]}]}
        for n in (1, 2, 5)
    ], ensure_ascii=False), encoding="utf-8")
    return source, tmp_path / "index.sqlite3"


def test_built_index_can_be_searched(materials, monkeypatch):
    source, database = materials
    texts = []

    def embed(text, task):
        assert task == "RETRIEVAL_DOCUMENT"
        texts.append(text)
        return [1, 0] if "本文1" in text else [0, 1]

    result = build_index(source, database, embedder=embed)
    assert result["pages"] == 2  # 第5回の資料は検索対象に含めない
    assert all("章" in text and "ページ" in text for text in texts)
    monkeypatch.setattr(retrieval, "RAG_DB", database)
    monkeypatch.setattr(retrieval, "RAG_NEIGHBORS", False)
    monkeypatch.setattr(retrieval, "embed_text", lambda *_: [1, 0])
    hits = retrieval.search("質問")
    assert [h["page_id"] for h in hits] == ["lec01-main-p001", "lec02-main-p001"]
    assert hits[0]["content"] == "本文1"
    assert hits[0]["distance"] == pytest.approx(0)


def _neighbor_index(tmp_path, monkeypatch, selected_pages):
    source = tmp_path / "neighbor-lectures.json"
    documents = []
    for lecture, page_count in ((1, 4), (2, 3)):
        document_id = f"lec{lecture:02d}-main"
        documents.append({
            "id": document_id,
            "title": f"講義{lecture}",
            "content": [{
                "section": "章",
                "content": [
                    {
                        "id": f"{document_id}-p{page:03d}",
                        "title": f"ページ{page}",
                        "page": page,
                        "content": f"本文 {document_id}-p{page:03d}",
                    }
                    for page in range(1, page_count + 1)
                ],
            }],
        })
    source.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
    database = tmp_path / "neighbor-index.sqlite3"

    def embed(text, _task):
        return [1, 0] if any(page_id in text for page_id in selected_pages) else [0, 1]

    build_index(source, database, embedder=embed)
    monkeypatch.setattr(retrieval, "RAG_DB", database)
    monkeypatch.setattr(retrieval, "RAG_TOP_K", len(selected_pages))
    monkeypatch.setattr(retrieval, "embed_text", lambda *_: [1, 0])
    monkeypatch.setattr(retrieval, "RAG_NEIGHBORS", True)


def test_search_appends_neighbors_once_without_crossing_documents(tmp_path, monkeypatch):
    selected = ["lec01-main-p002", "lec01-main-p003", "lec02-main-p002"]
    _neighbor_index(tmp_path, monkeypatch, selected)

    results = retrieval.search("質問")

    assert [page["page_id"] for page in results] == [
        *selected,
        "lec01-main-p001",
        "lec01-main-p004",
        "lec02-main-p001",
        "lec02-main-p003",
    ]
    assert [page.get("retrieval_source") for page in results[:3]] == [None] * 3
    assert all(page["retrieval_source"] == "neighbor" for page in results[3:])
    assert results[-1]["content"] == "本文 lec02-main-p003"
    assert len({page["page_id"] for page in results}) == len(results)


def test_search_neighbors_stop_at_document_boundaries(tmp_path, monkeypatch):
    _neighbor_index(tmp_path, monkeypatch, ["lec01-main-p001", "lec01-main-p004"])

    results = retrieval.search("質問")

    assert [page["page_id"] for page in results] == [
        "lec01-main-p001",
        "lec01-main-p004",
        "lec01-main-p002",
        "lec01-main-p003",
    ]
    assert all(page["document_id"] == "lec01-main" for page in results)


def test_neighbors_are_disabled_by_default(tmp_path, monkeypatch):
    _neighbor_index(tmp_path, monkeypatch, ["lec01-main-p002"])
    monkeypatch.setattr(retrieval, "RAG_NEIGHBORS", False)

    results = retrieval.search("質問")

    assert [page["page_id"] for page in results] == ["lec01-main-p002"]
    assert "retrieval_source" not in results[0]


def test_graph_does_not_use_neighbor_pages_as_search_seeds(tmp_path, monkeypatch):
    vector_hit = {"page_id": "lec01-main-p001", "distance": 0.1}
    neighbor_hit = {"page_id": "lec01-main-p002", "retrieval_source": "neighbor"}
    pages = [
        {"page_id": vector_hit["page_id"], "title": "検索結果"},
        {"page_id": neighbor_hit["page_id"], "title": "追加ページ"},
    ]
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps({
        "nodes": [
            {"id": page["page_id"], "kind": "page", "title": page["title"]}
            for page in pages
        ],
        "edges": [],
    }), encoding="utf-8")
    monkeypatch.setattr(retrieval, "load_pages", lambda *_: pages)
    monkeypatch.setattr(retrieval, "RAG_GRAPH", graph_path)
    monkeypatch.setattr(retrieval, "search", lambda *_: [vector_hit, neighbor_hit])

    results = retrieval.search_graph("質問")

    assert [page["page_id"] for page in results["seeds"]] == [vector_hit["page_id"]]


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True), (None, False)])
def test_rag_neighbors_environment_setting(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("AIE_RAG_NEIGHBORS", raising=False)
    else:
        monkeypatch.setenv("AIE_RAG_NEIGHBORS", value)

    assert reload(config).RAG_NEIGHBORS is expected


def test_existing_index_survives_failed_rebuild(materials):
    source, database = materials
    build_index(source, database, embedder=lambda *_: [1, 0])
    before = database.read_bytes()
    embed = Mock(side_effect=RuntimeError("埋め込みに失敗"))
    with pytest.raises(FileExistsError):
        build_index(source, database, embedder=embed)
    embed.assert_not_called()
    with pytest.raises(RuntimeError, match="埋め込みに失敗"):
        build_index(source, database, rebuild=True, embedder=embed)
    assert database.read_bytes() == before
    assert not list(database.parent.glob(".rag-build-*"))
