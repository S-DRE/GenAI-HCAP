"""RAG retrieval tool.

Design:
- VectorStoreProvider: abstract interface for similarity search (DIP).
- ChromaVectorStore: concrete implementation backed by ChromaDB (SRP —
  only responsible for search, not for the tool logic).
- retrieve_care_info: the LangChain tool function. Depends on
  VectorStoreProvider, not on Chroma directly.

The default retriever is a module-level singleton built from env config.
Tests or alternative backends can supply a different VectorStoreProvider
without changing any existing code (OCP).
"""

import os
from abc import ABC, abstractmethod

import structlog
from langchain_core.tools import tool

logger = structlog.get_logger()

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
COLLECTION_NAME = "care_plans"
_DEFAULT_K = 3


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class VectorStoreProvider(ABC):
    """Performs similarity search over a document collection."""

    @abstractmethod
    def search(self, query: str, k: int = _DEFAULT_K) -> list[str]:
        """Return up to k relevant document chunks for the given query."""


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------

class ChromaVectorStore(VectorStoreProvider):
    """ChromaDB-backed vector store provider."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str = CHROMA_PERSIST_DIR,
    ) -> None:
        self._collection_name = collection_name
        self._persist_directory = persist_directory

    def search(self, query: str, k: int = _DEFAULT_K) -> list[str]:
        from langchain_chroma import Chroma
        vectorstore = Chroma(
            collection_name=self._collection_name,
            persist_directory=self._persist_directory,
        )
        results = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]


# ---------------------------------------------------------------------------
# Default retriever instance
# ---------------------------------------------------------------------------

_default_retriever = ChromaVectorStore()


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

def _make_rag_tool(retriever: VectorStoreProvider):
    """Factory that binds a retriever to the tool function.

    Separating the factory from the tool definition keeps the tool itself
    stateless and independently testable.
    """

    @tool
    def retrieve_care_info(query: str) -> str:
        """Retrieve relevant care plan information, medication details, or medical guidelines
        for a given patient query. Use this tool whenever the user asks about their care plan,
        medications, daily routines, or health-related instructions."""
        logger.info("rag_tool_called", query=query)
        try:
            results = retriever.search(query)
            if not results:
                return (
                    "No specific care plan information was found for this query. "
                    "Please contact your caregiver for guidance."
                )
            context = "\n\n".join(results)
            logger.info("rag_results_found", count=len(results))
            return context
        except Exception as exc:
            logger.error("rag_tool_error", error=str(exc))
            return "Care plan information is temporarily unavailable. Please contact your caregiver directly."

    return retrieve_care_info


# Module-level tool instance used by the agent
retrieve_care_info = _make_rag_tool(_default_retriever)
