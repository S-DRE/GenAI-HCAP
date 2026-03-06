"""Document ingestion script.

Reads all .txt files from data/care_plans/ and data/guidelines/,
chunks them into overlapping segments, embeds them, and stores them
in ChromaDB for use by the RAG tool.

Usage:
    .venv\\Scripts\\python -m src.tools.ingest
"""

import os
from pathlib import Path

import structlog
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings

load_dotenv()

logger = structlog.get_logger()

DATA_DIR = Path(__file__).parents[2] / "data"
CARE_PLANS_DIR = DATA_DIR / "care_plans"
GUIDELINES_DIR = DATA_DIR / "guidelines"
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
COLLECTION_NAME = "care_plans"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_documents(directory: Path, source_type: str) -> list[dict]:
    """Read all .txt files from a directory and return as a list of dicts."""
    docs = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append({
            "content": text,
            "metadata": {
                "source": path.name,
                "type": source_type,
            }
        })
        logger.info("document_loaded", file=path.name, chars=len(text))
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["content"])
        for i, split in enumerate(splits):
            chunks.append({
                "content": split,
                "metadata": {**doc["metadata"], "chunk": i},
            })
    logger.info("documents_chunked", total_chunks=len(chunks))
    return chunks


def ingest(reset: bool = False) -> int:
    """Ingest all documents into ChromaDB.

    Args:
        reset: If True, clears the existing collection before ingesting.

    Returns:
        Number of chunks ingested.
    """
    care_plan_docs = load_documents(CARE_PLANS_DIR, source_type="care_plan")
    guideline_docs = load_documents(GUIDELINES_DIR, source_type="guideline")
    all_docs = care_plan_docs + guideline_docs

    if not all_docs:
        logger.warning("no_documents_found", dirs=[str(CARE_PLANS_DIR), str(GUIDELINES_DIR)])
        return 0

    chunks = chunk_documents(all_docs)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    if reset:
        logger.info("resetting_collection", collection=COLLECTION_NAME)
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    vectorstore.add_texts(texts=texts, metadatas=metadatas)

    logger.info(
        "ingestion_complete",
        chunks=len(chunks),
        collection=COLLECTION_NAME,
        persist_dir=CHROMA_PERSIST_DIR,
    )
    return len(chunks)


if __name__ == "__main__":
    ingest(reset=True)
