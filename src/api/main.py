import logging

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.graph import run_agent

load_dotenv()

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
logger = structlog.get_logger()

app = FastAPI(
    title="GenAI Home Care Assistance Platform",
    description="Conversational assistant for home care patients and caregivers.",
    version="0.1.0",
)

MAX_MESSAGE_LENGTH = 2_000  # characters — prevents quota exhaustion and prompt-injection spam


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info("chat_request_received", message_length=len(request.message))
    try:
        response = await run_agent(request.message)
    except Exception as exc:
        # Log the full detail internally but never expose it to the caller —
        # raw exception messages can contain API keys, org IDs, or token usage data.
        logger.error("chat_agent_error", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable. Please try again shortly.",
        ) from exc
    logger.info("chat_response_sent", response_length=len(response))
    return ChatResponse(response=response)
