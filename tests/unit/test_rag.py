"""Tests for the RAG retrieval tool.

After the SOLID refactor:
- ChromaVectorStore is the concrete backend (imports Chroma lazily inside search()).
- VectorStoreProvider is the abstraction.
- retrieve_care_info is built by _make_rag_tool() around a default retriever.

Patching strategy: inject a mock VectorStoreProvider directly into _make_rag_tool()
or patch the lazy Chroma import inside ChromaVectorStore.search.
"""

from unittest.mock import MagicMock, patch

from src.tools.rag import (
    ChromaVectorStore,
    VectorStoreProvider,
    _make_rag_tool,
)


class MockVectorStore(VectorStoreProvider):
    """In-memory VectorStoreProvider for testing."""

    def __init__(self, results: list[str]) -> None:
        self._results = results

    def search(self, query: str, k: int = 3) -> list[str]:
        return self._results


class TestVectorStoreProvider:
    def test_chroma_vector_store_implements_provider(self):
        assert issubclass(ChromaVectorStore, VectorStoreProvider)

    def test_custom_provider_can_implement_interface(self):
        store = MockVectorStore(["chunk 1"])
        assert store.search("test") == ["chunk 1"]


class TestRAGTool:
    def test_returns_context_when_results_found(self):
        store = MockVectorStore(["Take medication with food.", "Avoid alcohol while on this medication."])
        tool = _make_rag_tool(store)
        result = tool.invoke({"query": "medication instructions"})
        assert "Take medication with food." in result
        assert "Avoid alcohol while on this medication." in result

    def test_returns_fallback_when_no_results(self):
        store = MockVectorStore([])
        tool = _make_rag_tool(store)
        result = tool.invoke({"query": "something unknown"})
        assert "contact your caregiver" in result.lower()

    def test_returns_fallback_on_exception(self):
        class BrokenStore(VectorStoreProvider):
            def search(self, query: str, k: int = 3) -> list[str]:
                raise RuntimeError("DB connection failed")

        tool = _make_rag_tool(BrokenStore())
        result = tool.invoke({"query": "care plan"})
        assert "temporarily unavailable" in result.lower()

    def test_returns_string(self):
        store = MockVectorStore([])
        tool = _make_rag_tool(store)
        result = tool.invoke({"query": "test"})
        assert isinstance(result, str)

    def test_multiple_results_joined_with_newlines(self):
        store = MockVectorStore(["chunk A", "chunk B", "chunk C"])
        tool = _make_rag_tool(store)
        result = tool.invoke({"query": "care plan"})
        assert "chunk A" in result
        assert "chunk B" in result
        assert "chunk C" in result


class TestChromaVectorStoreSearch:
    """Integration-level tests for ChromaVectorStore.search (Chroma patched)."""

    def test_search_returns_page_content_strings(self):
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Take medication with food."
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Avoid alcohol."

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [mock_doc1, mock_doc2]

        with patch("langchain_chroma.Chroma", return_value=mock_store):
            store = ChromaVectorStore()
            results = store.search("medication")

        assert results == ["Take medication with food.", "Avoid alcohol."]

    def test_search_returns_empty_list_when_no_results(self):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []

        with patch("langchain_chroma.Chroma", return_value=mock_store):
            store = ChromaVectorStore()
            results = store.search("unknown query")

        assert results == []
