# GenAI Home Care Assistance Platform (HCAP)

> AI Architectural Kata — Conversational Home Care Assistant

---

## Business Context

A healthcare provider offers home care services for elderly and chronically ill patients. Caregivers and patients often need quick access to medical instructions, medication information, and care plans, but support staff are limited and cannot respond instantly to every request.

## Problem Statement

Design a conversational home-care assistance system that helps patients and caregivers manage daily care activities, answer health-related questions, and escalate issues when necessary.

## Current Limitations (Without GenAI)

- Patients rely on phone calls, printed care instructions, or caregiver visits
- Information is fragmented across patient records, care plans, and medical guidelines
- Response times are slow, especially outside working hours

## Desired Outcome

Build a voice-enabled assistant that can:
- Interact naturally with patients and caregivers
- Retrieve patient-specific care information
- Orchestrate workflows across healthcare services
- Coordinate multiple AI capabilities
- Enforce strong safety guardrails, monitoring, and cost-effective operation

---

## Architecture Overview

```
User (Voice/Text)
       │
       ▼
  Voice Layer (STT / TTS)
       │
       ▼
  Conversational Agent (LangGraph)
       │
  ┌────┴──────────────────────────────────┐
  │                                       │
  ▼                                       ▼
RAG Tool                           Escalation Tool
(Care Plans / Medication DB)       (Notify Caregiver/Staff)
  │
  ▼
Vector Store (ChromaDB)
  │
  ▼
Patient Records / Medical Guidelines
```

---

## Design Constraints

| Constraint | Decision |
|---|---|
| **Kata scope** | This is an architectural kata — a learning and design exercise, not a production system |
| **Zero cost** | All tools, models, and services must operate within free tiers or run fully locally; no paid API keys |
| **No cloud infrastructure** | The system must run on a local machine without provisioning paid cloud resources |
| **Synthetic data only** | All patient data used in demos must be fabricated; no real patient records |
| **Full test coverage** | Every module must have corresponding unit tests; all critical paths must be covered |

These constraints directly shaped every technology choice in the stack below.

---

## Tech Stack

> All components are **free to use** — no paid API keys required.

### Language & Runtime
| Layer | Choice | Rationale |
|---|---|---|
| Language | **Python 3.12** | Richest GenAI/ML ecosystem; first-class SDK support from all major LLM providers |

### AI & Orchestration
| Layer | Choice | Rationale |
|---|---|---|
| LLM Provider | **Groq API — Llama 3.3 70B** | Free tier with generous limits; extremely fast inference; OpenAI-compatible SDK; Llama 3.3 70B is capable enough for healthcare assistant workflows |
| Agent Framework | **LangGraph** | Stateful multi-step agentic workflows; supports conditional edges for escalation logic |
| RAG Framework | **LangChain** | Integrates cleanly with LangGraph; broad retriever and loader ecosystem |

### Voice
| Layer | Choice | Rationale |
|---|---|---|
| Speech-to-Text | **OpenAI Whisper (local)** — `openai-whisper` pip package | Runs fully locally on CPU; no API key or cost; best open-source STT accuracy available |
| Text-to-Speech | **Coqui TTS** — `TTS` pip package | Free, local, high-quality voice synthesis; no API key or cost |

### Knowledge & Storage
| Layer | Choice | Rationale |
|---|---|---|
| Vector Store | **ChromaDB** | Lightweight, embeddable, no external service required for prototyping |
| Document Loaders | **LangChain loaders** | PDF, CSV, and JSON ingestion for care plans and medical guidelines |

### Safety & Guardrails
| Layer | Choice | Rationale |
|---|---|---|
| Output Validation | **Guardrails AI** | Schema-based validation of LLM outputs; supports medical domain rules |
| Content Filtering | **LlamaGuard 3** (via Groq) | Free, open-source safety model for detecting harmful content in inputs and outputs |

### Observability & Cost Management
| Layer | Choice | Rationale |
|---|---|---|
| LLM Tracing | **LangSmith** (free tier) | End-to-end tracing of agent steps, token usage, and latency; free tier is sufficient for a kata |
| Logging | **Python `logging` + structlog** | Structured logs for audit trails (important in healthcare contexts) |

### API & Deployment
| Layer | Choice | Rationale |
|---|---|---|
| API Framework | **FastAPI** | Async-first, auto-generated OpenAPI docs, lightweight |
| Containerization | **Docker + Docker Compose** | Reproducible environments; easy local and cloud deployment |

---

## Key Architectural Decisions

### 1. LangGraph for Agentic Orchestration
LangGraph's graph-based state machine is well-suited for healthcare workflows where the conversation can branch into distinct paths: answering a routine care question, retrieving medication data, or escalating an emergency. The explicit state transitions make the system auditable and predictable — critical in a healthcare context.

### 2. RAG over Patient-Specific Data
Rather than relying solely on the LLM's parametric knowledge, patient care plans and medical guidelines are indexed in a vector store and retrieved at query time. This grounds responses in real, up-to-date patient data and reduces hallucination risk.

### 3. Guardrails as a First-Class Concern
Given the sensitive nature of medical advice, output validation is not optional. Guardrails AI enforces rules (e.g., "never recommend dosage changes", "always defer to a doctor for diagnoses") on every LLM response before it reaches the user.

### 4. Voice as the Primary Interface
Elderly patients and caregivers benefit most from voice interaction. Local Whisper (STT) + Coqui TTS provides a full round-trip voice experience with zero API costs and no data leaving the machine.

### 5. Escalation as a Tool
Emergency or high-severity situations are handled by a dedicated escalation tool in the agent's toolkit. When the agent detects urgency signals (keywords, sentiment, explicit requests), it triggers the escalation workflow to notify on-call staff.

### 6. Groq for Free, Fast LLM Inference
Groq's free tier offers some of the fastest LLM inference available (low-latency responses are important for voice interactions). Llama 3.3 70B is the model of choice — open-weight, instruction-tuned, and capable of tool calling and structured output. The Groq SDK is OpenAI-compatible, minimizing migration effort if the LLM provider changes in the future.

---

## Project Structure (Proposed)

```
GenAI-HCAP/
├── src/
│   ├── agent/           # LangGraph agent definition and graph
│   ├── tools/           # RAG tool, escalation tool, medication lookup
│   ├── voice/           # Local Whisper STT and Coqui TTS wrappers
│   ├── guardrails/      # Guardrails AI validators
│   └── api/             # FastAPI routes
├── data/
│   ├── care_plans/      # Sample patient care plan documents
│   └── guidelines/      # Medical guideline documents
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Getting Started

> Prerequisites: **Python 3.12** (required — 3.13+ not yet supported by the AI/ML ecosystem), Docker, Groq API key (free at [console.groq.com](https://console.groq.com)), LangSmith API key (free at [smith.langchain.com](https://smith.langchain.com))

```bash
# Clone the repo
git clone <repo-url>
cd GenAI-HCAP

# Create virtual environment (must use Python 3.12 explicitly)
py -3.12 -m venv .venv       # Windows with Python Install Manager
# python3.12 -m venv .venv   # Mac/Linux alternative
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the API (--env-file loads .env automatically)
uvicorn src.api.main:app --reload --env-file .env
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for LLM inference (free at console.groq.com) |
| `GROQ_MODEL` | No | Groq model to use. Default: `llama-3.3-70b-versatile`. Switch to `llama-3.1-8b-instant` if the 70B daily token limit (100k TPD on free tier) is exhausted |
| `LANGSMITH_API_KEY` | No | LangSmith tracing key (free tier) |
| `LANGSMITH_PROJECT` | No | LangSmith project name |
| `CHROMA_PERSIST_DIR` | No | Local path for ChromaDB persistence (default: `./data/chroma`) |
| `E2E_BASE_URL` | No | Override the server address for E2E tests (default: `http://127.0.0.1:8000`) |

> No paid API keys required. Whisper and Coqui TTS run fully locally.

---

## Testing

Full test coverage is a core acceptance criterion for this kata. Every module has a corresponding test file under `tests/`.

```
tests/
├── test_validators.py      # Guardrails keyword blocking and escalation detection
├── test_rag.py             # RAG tool: found results, empty results, error handling
├── test_escalation.py      # Escalation tool: message format and logging
├── test_stt.py             # STT wrapper: transcription output
├── test_tts.py             # TTS wrapper: audio file generation
├── test_api.py             # FastAPI routes: /health and /chat (unit)
├── test_agent.py           # LangGraph agent: state, tool wiring, guardrails
├── test_ingest.py          # Ingestion pipeline: loading, chunking, vectorstore
├── test_integration.py     # Integration: full HTTP → agent → tools → guardrails (LLM mocked)
└── test_e2e.py             # E2E: live server + real Groq API (skipped without GROQ_API_KEY)
```

External dependencies (Groq API, ChromaDB, Whisper, Coqui TTS) are mocked in unit and integration tests so they run fully offline with no API keys required.

```bash
# Run all unit + integration tests (default — E2E excluded automatically)
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run only end-to-end tests (requires live server + GROQ_API_KEY)
# 1. Start the server:  uvicorn src.api.main:app
# 2. Then run:
pytest -m e2e

# Run everything including E2E (not recommended for day-to-day use)
pytest -m ""
```

### Test types at a glance

| Type | File | LLM real? | Server needed? | Runs in CI? |
|---|---|---|---|---|
| Unit | `test_*.py` (except integration/e2e) | No | No | Yes (default) |
| Integration | `test_integration.py` | No (mocked) | No | Yes (default) |
| E2E | `test_e2e.py` | Yes | Yes | Opt-in: `pytest -m e2e` |

> **Note:** `TestEscalationE2E` inside `test_e2e.py` is permanently skipped (`@pytest.mark.skip`) because the escalation path requires two sequential LLM calls which consistently exceeds free-tier timeouts. The same behaviour is fully covered by `test_integration.py::TestChatEscalationToolIntegration` where the LLM is mocked. The tests are kept in the file as documentation of the expected live behaviour.

---

## Safety & Compliance Notes

- This is a **kata / prototype** — not intended for production clinical use
- All LLM responses must be validated through guardrails before delivery
- The system should never provide diagnosis, prescriptions, or emergency medical advice without escalation
- Patient data used in demos must be **synthetic / anonymized**

---

## Glossary

| Acronym / Term | Full Name | Description |
|---|---|---|
| **GenAI** | Generative AI | AI systems capable of generating text, audio, or other content in response to prompts |
| **HCAP** | Home Care Assistance Platform | The name of this project |
| **LLM** | Large Language Model | A deep learning model trained on large text datasets to understand and generate human language (e.g. Llama 3.3) |
| **RAG** | Retrieval-Augmented Generation | A technique that retrieves relevant documents from a knowledge base and feeds them to the LLM as context, grounding responses in real data |
| **STT** | Speech-to-Text | Technology that converts spoken audio into a written text transcript (implemented here with local Whisper) |
| **TTS** | Text-to-Speech | Technology that converts written text into synthesised spoken audio (implemented here with local Coqui TTS) |
| **API** | Application Programming Interface | A set of endpoints that allow software systems to communicate; here exposed via FastAPI |
| **LPU** | Language Processing Unit | Groq's custom chip designed specifically for fast LLM inference |
| **GPU** | Graphics Processing Unit | Hardware originally designed for rendering graphics, widely repurposed for AI training and inference |
| **SDK** | Software Development Kit | A set of tools and libraries for building software against a particular platform or API |
| **STM** | State Machine | A model of computation where the system transitions between defined states based on inputs; used here to describe LangGraph's agent flow |
