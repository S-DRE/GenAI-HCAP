"""Tests for the document ingestion pipeline.

After the SOLID refactor, each concern is a separate class:
- DocumentLoader  — reads files from disk
- DocumentChunker — splits text into overlapping chunks
- DocumentStore   — abstract backend interface
- ChromaDocumentStore — concrete Chroma implementation
- ingest()        — orchestrator; accepts injected loader/chunker/store
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tools.ingest import (
    ChromaDocumentStore,
    Document,
    DocumentChunker,
    DocumentLoader,
    DocumentStore,
    ingest,
)


# ---------------------------------------------------------------------------
# Helpers / mocks
# ---------------------------------------------------------------------------

class InMemoryDocumentStore(DocumentStore):
    """Test double that captures added chunks in memory."""

    def __init__(self):
        self.added: list[Document] = []
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True
        self.added.clear()

    def add(self, chunks: list[Document]) -> None:
        self.added.extend(chunks)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class TestDocument:
    def test_stores_content_and_metadata(self):
        doc = Document(content="hello", metadata={"source": "file.txt"})
        assert doc.content == "hello"
        assert doc.metadata["source"] == "file.txt"


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------

class TestDocumentLoader:
    def test_loads_txt_files(self, tmp_path):
        (tmp_path / "patient_a.txt").write_text("Care plan A.", encoding="utf-8")
        (tmp_path / "patient_b.txt").write_text("Care plan B.", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load(tmp_path, source_type="care_plan")

        assert len(docs) == 2
        contents = [d.content for d in docs]
        assert "Care plan A." in contents
        assert "Care plan B." in contents

    def test_sets_correct_metadata(self, tmp_path):
        (tmp_path / "test_patient.txt").write_text("Some content.", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load(tmp_path, source_type="guideline")

        assert docs[0].metadata["source"] == "test_patient.txt"
        assert docs[0].metadata["type"] == "guideline"

    def test_returns_empty_list_for_empty_directory(self, tmp_path):
        docs = DocumentLoader().load(tmp_path, source_type="care_plan")
        assert docs == []

    def test_ignores_non_txt_files(self, tmp_path):
        (tmp_path / "document.txt").write_text("Valid.", encoding="utf-8")
        (tmp_path / "document.pdf").write_bytes(b"%PDF-ignore")
        (tmp_path / "notes.csv").write_text("a,b,c", encoding="utf-8")

        docs = DocumentLoader().load(tmp_path, source_type="care_plan")
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# DocumentChunker
# ---------------------------------------------------------------------------

class TestDocumentChunker:
    def test_splits_long_document_into_multiple_chunks(self):
        doc = Document(content="This is a sentence. " * 100, metadata={"source": "test.txt", "type": "care_plan"})
        chunks = DocumentChunker().chunk([doc])
        assert len(chunks) > 1

    def test_each_chunk_has_content_and_metadata(self):
        doc = Document(content="Short text. " * 10, metadata={"source": "s.txt", "type": "guideline"})
        chunks = DocumentChunker().chunk([doc])
        for chunk in chunks:
            assert chunk.content.strip() != ""
            assert "source" in chunk.metadata

    def test_chunk_metadata_includes_chunk_index(self):
        doc = Document(content="Word " * 300, metadata={"source": "t.txt", "type": "care_plan"})
        chunks = DocumentChunker().chunk([doc])
        indices = [c.metadata["chunk"] for c in chunks]
        assert 0 in indices

    def test_preserves_source_metadata_in_chunks(self):
        doc = Document(content="Some content. " * 20, metadata={"source": "orig.txt", "type": "guideline"})
        chunks = DocumentChunker().chunk([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "orig.txt"
            assert chunk.metadata["type"] == "guideline"

    def test_returns_empty_list_for_empty_input(self):
        assert DocumentChunker().chunk([]) == []


# ---------------------------------------------------------------------------
# DocumentStore abstract interface
# ---------------------------------------------------------------------------

class TestDocumentStore:
    def test_chroma_document_store_implements_interface(self):
        assert issubclass(ChromaDocumentStore, DocumentStore)

    def test_in_memory_store_can_implement_interface(self):
        store = InMemoryDocumentStore()
        doc = Document(content="hello", metadata={})
        store.add([doc])
        assert store.added[0].content == "hello"


# ---------------------------------------------------------------------------
# ingest() orchestrator
# ---------------------------------------------------------------------------

class TestIngest:
    def test_returns_chunk_count(self, tmp_path):
        (tmp_path / "care_plans").mkdir()
        (tmp_path / "guidelines").mkdir()
        (tmp_path / "care_plans" / "patient.txt").write_text("Care plan. " * 20, encoding="utf-8")
        (tmp_path / "guidelines" / "guide.txt").write_text("Guideline. " * 20, encoding="utf-8")

        store = InMemoryDocumentStore()

        with patch("src.tools.ingest.CARE_PLANS_DIR", tmp_path / "care_plans"), \
             patch("src.tools.ingest.GUIDELINES_DIR", tmp_path / "guidelines"):
            count = ingest(reset=False, store=store)

        assert count > 0
        assert len(store.added) == count

    def test_returns_zero_when_no_documents(self, tmp_path):
        (tmp_path / "care_plans").mkdir()
        (tmp_path / "guidelines").mkdir()

        store = InMemoryDocumentStore()

        with patch("src.tools.ingest.CARE_PLANS_DIR", tmp_path / "care_plans"), \
             patch("src.tools.ingest.GUIDELINES_DIR", tmp_path / "guidelines"):
            count = ingest(reset=False, store=store)

        assert count == 0

    def test_store_add_is_called_with_chunks(self, tmp_path):
        (tmp_path / "care_plans").mkdir()
        (tmp_path / "guidelines").mkdir()
        (tmp_path / "care_plans" / "p.txt").write_text("Content. " * 10, encoding="utf-8")

        store = InMemoryDocumentStore()

        with patch("src.tools.ingest.CARE_PLANS_DIR", tmp_path / "care_plans"), \
             patch("src.tools.ingest.GUIDELINES_DIR", tmp_path / "guidelines"):
            ingest(reset=False, store=store)

        assert len(store.added) > 0

    def test_reset_calls_store_reset(self, tmp_path):
        (tmp_path / "care_plans").mkdir()
        (tmp_path / "guidelines").mkdir()
        (tmp_path / "care_plans" / "p.txt").write_text("Content. " * 10, encoding="utf-8")

        store = InMemoryDocumentStore()

        with patch("src.tools.ingest.CARE_PLANS_DIR", tmp_path / "care_plans"), \
             patch("src.tools.ingest.GUIDELINES_DIR", tmp_path / "guidelines"):
            ingest(reset=True, store=store)

        assert store.reset_called

    def test_no_reset_does_not_call_store_reset(self, tmp_path):
        (tmp_path / "care_plans").mkdir()
        (tmp_path / "guidelines").mkdir()

        store = InMemoryDocumentStore()

        with patch("src.tools.ingest.CARE_PLANS_DIR", tmp_path / "care_plans"), \
             patch("src.tools.ingest.GUIDELINES_DIR", tmp_path / "guidelines"):
            ingest(reset=False, store=store)

        assert not store.reset_called

    def test_injectable_loader_is_used(self, tmp_path):
        """A custom loader that returns a fixed set of Documents."""
        class FakeLoader(DocumentLoader):
            def load(self, directory: Path, source_type: str) -> list[Document]:
                return [Document(content="Fixed content. " * 10, metadata={"source": "fixed.txt", "type": source_type})]

        store = InMemoryDocumentStore()
        with patch("src.tools.ingest.CARE_PLANS_DIR", tmp_path), \
             patch("src.tools.ingest.GUIDELINES_DIR", tmp_path):
            count = ingest(reset=False, loader=FakeLoader(), store=store)

        assert count > 0
