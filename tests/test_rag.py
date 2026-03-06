from unittest.mock import MagicMock, patch


class TestRAGTool:
    def test_returns_context_when_results_found(self):
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Take medication with food."
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Avoid alcohol while on this medication."

        with patch("src.tools.rag.Chroma") as mock_chroma_cls:
            mock_store = MagicMock()
            mock_store.similarity_search.return_value = [mock_doc1, mock_doc2]
            mock_chroma_cls.return_value = mock_store

            from src.tools.rag import retrieve_care_info
            result = retrieve_care_info.invoke({"query": "medication instructions"})

        assert "Take medication with food." in result
        assert "Avoid alcohol while on this medication." in result

    def test_returns_fallback_when_no_results(self):
        with patch("src.tools.rag.Chroma") as mock_chroma_cls:
            mock_store = MagicMock()
            mock_store.similarity_search.return_value = []
            mock_chroma_cls.return_value = mock_store

            from src.tools.rag import retrieve_care_info
            result = retrieve_care_info.invoke({"query": "something unknown"})

        assert "contact your caregiver" in result.lower()

    def test_returns_fallback_on_exception(self):
        with patch("src.tools.rag.Chroma") as mock_chroma_cls:
            mock_chroma_cls.side_effect = Exception("DB connection failed")

            from src.tools.rag import retrieve_care_info
            result = retrieve_care_info.invoke({"query": "care plan"})

        assert "temporarily unavailable" in result.lower()

    def test_returns_string(self):
        with patch("src.tools.rag.Chroma") as mock_chroma_cls:
            mock_store = MagicMock()
            mock_store.similarity_search.return_value = []
            mock_chroma_cls.return_value = mock_store

            from src.tools.rag import retrieve_care_info
            result = retrieve_care_info.invoke({"query": "test"})

        assert isinstance(result, str)
