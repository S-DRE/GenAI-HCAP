import os

import structlog
from langchain_chroma import Chroma
from langchain_core.tools import tool

logger = structlog.get_logger()

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
COLLECTION_NAME = "care_plans"


@tool
def retrieve_care_info(query: str) -> str:
    """Retrieve relevant care plan information, medication details, or medical guidelines
    for a given patient query. Use this tool whenever the user asks about their care plan,
    medications, daily routines, or health-related instructions."""
    logger.info("rag_tool_called", query=query)

    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        results = vectorstore.similarity_search(query, k=3)

        if not results:
            return (
                "No specific care plan information was found for this query. "
                "Please contact your caregiver for guidance."
            )

        context = "\n\n".join(doc.page_content for doc in results)
        logger.info("rag_results_found", count=len(results))
        return context

    except Exception as e:
        logger.error("rag_tool_error", error=str(e))
        return "Care plan information is temporarily unavailable. Please contact your caregiver directly."
