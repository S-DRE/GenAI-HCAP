import os
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestLoadDocuments:
    def test_loads_txt_files(self, tmp_path):
        (tmp_path / "patient_a.txt").write_text("Care plan content for patient A.", encoding="utf-8")
        (tmp_path / "patient_b.txt").write_text("Care plan content for patient B.", encoding="utf-8")

        from src.tools.ingest import load_documents
        docs = load_documents(tmp_path, source_type="care_plan")

        assert len(docs) == 2
        contents = [d["content"] for d in docs]
        assert "Care plan content for patient A." in contents
        assert "Care plan content for patient B." in contents

    def test_sets_correct_metadata(self, tmp_path):
        (tmp_path / "test_patient.txt").write_text("Some content.", encoding="utf-8")

        from src.tools.ingest import load_documents
        docs = load_documents(tmp_path, source_type="guideline")

        assert docs[0]["metadata"]["source"] == "test_patient.txt"
        assert docs[0]["metadata"]["type"] == "guideline"

    def test_returns_empty_list_for_empty_directory(self, tmp_path):
        from src.tools.ingest import load_documents
        docs = load_documents(tmp_path, source_type="care_plan")
        assert docs == []

    def test_ignores_non_txt_files(self, tmp_path):
        (tmp_path / "document.txt").write_text("Valid.", encoding="utf-8")
        (tmp_path / "document.pdf").write_bytes(b"%PDF-ignore")
        (tmp_path / "notes.csv").write_text("a,b,c", encoding="utf-8")

        from src.tools.ingest import load_documents
        docs = load_documents(tmp_path, source_type="care_plan")
        assert len(docs) == 1


class TestChunkDocuments:
    def test_splits_long_document_into_multiple_chunks(self):
        long_text = "This is a sentence. " * 100
        docs = [{"content": long_text, "metadata": {"source": "test.txt", "type": "care_plan"}}]

        from src.tools.ingest import chunk_documents
        chunks = chunk_documents(docs)

        assert len(chunks) > 1

    def test_each_chunk_has_content_and_metadata(self):
        docs = [{"content": "Short text. " * 10, "metadata": {"source": "s.txt", "type": "guideline"}}]

        from src.tools.ingest import chunk_documents
        chunks = chunk_documents(docs)

        for chunk in chunks:
            assert "content" in chunk
            assert "metadata" in chunk
            assert chunk["content"].strip() != ""

    def test_chunk_metadata_includes_chunk_index(self):
        long_text = "Word " * 300
        docs = [{"content": long_text, "metadata": {"source": "t.txt", "type": "care_plan"}}]

        from src.tools.ingest import chunk_documents
        chunks = chunk_documents(docs)

        indices = [c["metadata"]["chunk"] for c in chunks]
        assert 0 in indices

    def test_preserves_source_metadata_in_chunks(self):
        docs = [{"content": "Some content. " * 20, "metadata": {"source": "orig.txt", "type": "guideline"}}]

        from src.tools.ingest import chunk_documents
        chunks = chunk_documents(docs)

        for chunk in chunks:
            assert chunk["metadata"]["source"] == "orig.txt"
            assert chunk["metadata"]["type"] == "guideline"

    def test_returns_empty_list_for_empty_input(self):
        from src.tools.ingest import chunk_documents
        assert chunk_documents([]) == []


class TestIngest:
    def test_returns_chunk_count(self, tmp_path):
        mock_vectorstore = MagicMock()

        with patch("src.tools.ingest.CARE_PLANS_DIR", tmp_path / "care_plans"), \
             patch("src.tools.ingest.GUIDELINES_DIR", tmp_path / "guidelines"), \
             patch("src.tools.ingest.Chroma", return_value=mock_vectorstore), \
             patch("src.tools.ingest.SentenceTransformerEmbeddings"):

            (tmp_path / "care_plans").mkdir()
            (tmp_path / "guidelines").mkdir()
            (tmp_path / "care_plans" / "patient.txt").write_text("Care plan. " * 20, encoding="utf-8")
            (tmp_path / "guidelines" / "guide.txt").write_text("Guideline. " * 20, encoding="utf-8")

            from src.tools.ingest import ingest
            count = ingest(reset=False)

        assert count > 0

    def test_returns_zero_when_no_documents(self, tmp_path):
        with patch("src.tools.ingest.CARE_PLANS_DIR", tmp_path / "care_plans"), \
             patch("src.tools.ingest.GUIDELINES_DIR", tmp_path / "guidelines"):

            (tmp_path / "care_plans").mkdir()
            (tmp_path / "guidelines").mkdir()

            from src.tools.ingest import ingest
            count = ingest(reset=False)

        assert count == 0

    def test_calls_vectorstore_add_texts(self, tmp_path):
        mock_vectorstore = MagicMock()

        with patch("src.tools.ingest.CARE_PLANS_DIR", tmp_path / "care_plans"), \
             patch("src.tools.ingest.GUIDELINES_DIR", tmp_path / "guidelines"), \
             patch("src.tools.ingest.Chroma", return_value=mock_vectorstore), \
             patch("src.tools.ingest.SentenceTransformerEmbeddings"):

            (tmp_path / "care_plans").mkdir()
            (tmp_path / "guidelines").mkdir()
            (tmp_path / "care_plans" / "p.txt").write_text("Content. " * 10, encoding="utf-8")

            from src.tools.ingest import ingest
            ingest(reset=False)

        mock_vectorstore.add_texts.assert_called_once()

    def test_reset_deletes_collection_before_ingesting(self, tmp_path):
        mock_vectorstore = MagicMock()
        mock_chroma_client = MagicMock()

        with patch("src.tools.ingest.CARE_PLANS_DIR", tmp_path / "care_plans"), \
             patch("src.tools.ingest.GUIDELINES_DIR", tmp_path / "guidelines"), \
             patch("src.tools.ingest.Chroma", return_value=mock_vectorstore), \
             patch("src.tools.ingest.SentenceTransformerEmbeddings"), \
             patch("chromadb.PersistentClient", return_value=mock_chroma_client):

            (tmp_path / "care_plans").mkdir()
            (tmp_path / "guidelines").mkdir()
            (tmp_path / "care_plans" / "p.txt").write_text("Content. " * 10, encoding="utf-8")

            from src.tools.ingest import ingest
            ingest(reset=True)

        mock_chroma_client.delete_collection.assert_called_once()
