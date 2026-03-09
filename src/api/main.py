import logging
import os
import tempfile

import structlog
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

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

MAX_MESSAGE_LENGTH = 2_000
MAX_AUDIO_BYTES = 25 * 1024 * 1024
_UPLOAD_CHUNK_SIZE = 64 * 1024
_ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ChatResponse(BaseModel):
    response: str


class ServiceFactory:
    def __init__(self) -> None:
        self._voice_pipeline = None

    def get_voice_pipeline(self):
        if self._voice_pipeline is None:
            from src.voice.pipeline import VoicePipeline
            from src.voice.stt import WhisperSTT
            from src.voice.tts import CoquiTTS

            self._voice_pipeline = VoicePipeline(
                stt=WhisperSTT(model_size=os.environ.get("WHISPER_MODEL_SIZE", "base")),
                tts=CoquiTTS(model_name=os.environ.get("TTS_MODEL", CoquiTTS.DEFAULT_MODEL)),
            )
        return self._voice_pipeline


_factory = ServiceFactory()


def get_factory() -> ServiceFactory:
    return _factory


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    from src.agent.graph import run_agent

    logger.info("chat_request_received", message_length=len(request.message))
    try:
        response = await run_agent(request.message)
    except Exception as exc:
        # Never expose internal error details to the caller.
        logger.error("chat_agent_error", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable. Please try again shortly.",
        ) from exc
    logger.info("chat_response_sent", response_length=len(response))
    return ChatResponse(response=response)


@app.post("/voice", response_class=FileResponse)
async def voice(
    audio: UploadFile = File(..., description="Audio file (wav, mp3, m4a, etc.)"),
    factory: ServiceFactory = Depends(get_factory),
):
    logger.info("voice_request_received", filename=audio.filename, content_type=audio.content_type)

    # Never trust the client-supplied filename — derive extension from an allowlist.
    raw_ext = os.path.splitext(audio.filename or "")[1].lower()
    suffix = raw_ext if raw_ext in _ALLOWED_AUDIO_EXTENSIONS else ".wav"

    # Read in chunks to enforce a size cap before writing to disk.
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await audio.read(_UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large. Maximum allowed size is {MAX_AUDIO_BYTES // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(b"".join(chunks))
        tmp_path = tmp.name

    try:
        output_path = await factory.get_voice_pipeline().process(tmp_path)
    except ValueError as exc:
        logger.warning("voice_transcription_empty", error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("voice_pipeline_error", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="The voice assistant is temporarily unavailable. Please try again shortly.",
        ) from exc
    finally:
        os.unlink(tmp_path)

    logger.info("voice_response_ready", output=output_path)
    return FileResponse(output_path, media_type="audio/wav", filename="response.wav")
