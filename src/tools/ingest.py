"""Document ingestion pipeline.

Design:
- DocumentLoader:    loads raw .txt files from a directory (SRP).
- DocumentChunker:  splits raw text into overlapping chunks (SRP).
- DocumentStore:    abstract interface for persisting chunks (DIP / OCP).
- ChromaDocumentStore: concrete Chroma-backed implementation.
- ingest():         orchestrates the three steps; depends on abstractions
                    so it can be tested with in-memory mocks.

Usage:
    .venv\\Scripts\\python -m src.tools.ingest
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path

import structlog
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = structlog.get_logger()

DATA_DIR = Path(__file__).parents[2] / "data"
CARE_PLANS_DIR = DATA_DIR / "care_plans"
GUIDELINES_DIR = DATA_DIR / "guidelines"
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
COLLECTION_NAME = "care_plans"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Document representation
# ---------------------------------------------------------------------------

class Document:
    """Plain data object: a piece of text with associated metadata."""

    def __init__(self, content: str, metadata: dict) -> None:
        self.content = content
        self.metadata = metadata


# ---------------------------------------------------------------------------
# Loader — SRP: only reads files from disk
# ---------------------------------------------------------------------------

class DocumentLoader:
    """Reads all .txt files from a directory."""

    def load(self, directory: Path, source_type: str) -> list[Document]:
        docs: list[Document] = []
        for path in sorted(directory.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            docs.append(Document(
                content=text,
                metadata={"source": path.name, "type": source_type},
            ))
            logger.info("document_loaded", file=path.name, chars=len(text))
        return docs


# ---------------------------------------------------------------------------
# Chunker — SRP: only splits text
# ---------------------------------------------------------------------------

class DocumentChunker:
    """Splits documents into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(self, docs: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in docs:
            for i, split in enumerate(self._splitter.split_text(doc.content)):
                chunks.append(Document(
                    content=split,
                    metadata={**doc.metadata, "chunk": i},
                ))
        logger.info("documents_chunked", total_chunks=len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# DocumentStore — abstract interface (DIP / OCP)
# ---------------------------------------------------------------------------

class DocumentStore(ABC):
    """Persists document chunks for later retrieval."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all existing documents in the store."""

    @abstractmethod
    def add(self, chunks: list[Document]) -> None:
        """Persist a list of document chunks."""


# ---------------------------------------------------------------------------
# Concrete Chroma implementation
# ---------------------------------------------------------------------------

class ChromaDocumentStore(DocumentStore):
    """Chroma-backed document store."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str = CHROMA_PERSIST_DIR,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._embedding_model = embedding_model

    def reset(self) -> None:
        import chromadb
        logger.info("resetting_collection", collection=self._collection_name)
        client = chromadb.PersistentClient(path=self._persist_directory)
        try:
            client.delete_collection(self._collection_name)
        except Exception:
            pass

    def add(self, chunks: list[Document]) -> None:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=self._embedding_model)
        vectorstore = Chroma(
            collection_name=self._collection_name,
            embedding_function=embeddings,
            persist_directory=self._persist_directory,
        )
        vectorstore.add_texts(
            texts=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        logger.info(
            "chunks_stored",
            count=len(chunks),
            collection=self._collection_name,
            persist_dir=self._persist_directory,
        )


# ---------------------------------------------------------------------------
# Orchestration function
# ---------------------------------------------------------------------------

def ingest(
    reset: bool = False,
    loader: DocumentLoader | None = None,
    chunker: DocumentChunker | None = None,
    store: DocumentStore | None = None,
) -> int:
    """Ingest all documents into the document store.

    Args:
        reset:   If True, clears the store before ingesting.
        loader:  DocumentLoader to use; defaults to a plain FileLoader.
        chunker: DocumentChunker to use; defaults to RecursiveCharacterTextSplitter.
        store:   DocumentStore to use; defaults to ChromaDocumentStore.

    Returns:
        Number of chunks ingested.
    """
    loader = loader or DocumentLoader()
    chunker = chunker or DocumentChunker()
    store = store or ChromaDocumentStore()

    care_plan_docs = loader.load(CARE_PLANS_DIR, source_type="care_plan")
    guideline_docs = loader.load(GUIDELINES_DIR, source_type="guideline")
    all_docs = care_plan_docs + guideline_docs

    if not all_docs:
        logger.warning("no_documents_found", dirs=[str(CARE_PLANS_DIR), str(GUIDELINES_DIR)])
        return 0

    chunks = chunker.chunk(all_docs)

    if reset:
        store.reset()

    store.add(chunks)

    logger.info("ingestion_complete", chunks=len(chunks))
    return len(chunks)


if __name__ == "__main__":
    ingest(reset=True)
