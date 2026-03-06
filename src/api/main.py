import logging

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info("chat_request_received", message=request.message)
    try:
        response = await run_agent(request.message)
    except Exception as exc:
        logger.error("chat_agent_error", error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.info("chat_response_sent", response=response)
    return ChatResponse(response=response)
