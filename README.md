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

## Tech Stack

### Language & Runtime
| Layer | Choice | Rationale |
|---|---|---|
| Language | **Python 3.12** | Richest GenAI/ML ecosystem; first-class SDK support from all major LLM providers |

### AI & Orchestration
| Layer | Choice | Rationale |
|---|---|---|
| LLM Provider | **OpenAI GPT-4o** | Strong tool-calling, vision, and structured output support |
| Agent Framework | **LangGraph** | Stateful multi-step agentic workflows; supports conditional edges for escalation logic |
| RAG Framework | **LangChain** | Integrates cleanly with LangGraph; broad retriever and loader ecosystem |

### Voice
| Layer | Choice | Rationale |
|---|---|---|
| Speech-to-Text | **OpenAI Whisper** | High accuracy; handles accented/elderly speech well |
| Text-to-Speech | **OpenAI TTS** | Low-latency, natural-sounding voice output |

### Knowledge & Storage
| Layer | Choice | Rationale |
|---|---|---|
| Vector Store | **ChromaDB** | Lightweight, embeddable, no external service required for prototyping |
| Document Loaders | **LangChain loaders** | PDF, CSV, and JSON ingestion for care plans and medical guidelines |

### Safety & Guardrails
| Layer | Choice | Rationale |
|---|---|---|
| Output Validation | **Guardrails AI** | Schema-based validation of LLM outputs; supports medical domain rules |
| Content Filtering | **OpenAI Moderation API** | Real-time flagging of harmful or unsafe content |

### Observability & Cost Management
| Layer | Choice | Rationale |
|---|---|---|
| LLM Tracing | **LangSmith** | End-to-end tracing of agent steps, token usage, and latency |
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
Elderly patients and caregivers benefit most from voice interaction. Whisper STT + OpenAI TTS provides a full round-trip voice experience without requiring third-party telephony infrastructure for the kata scope.

### 5. Escalation as a Tool
Emergency or high-severity situations are handled by a dedicated escalation tool in the agent's toolkit. When the agent detects urgency signals (keywords, sentiment, explicit requests), it triggers the escalation workflow to notify on-call staff.

---

## Project Structure (Proposed)

```
GenAI-HCAP/
├── src/
│   ├── agent/           # LangGraph agent definition and graph
│   ├── tools/           # RAG tool, escalation tool, medication lookup
│   ├── voice/           # Whisper STT and TTS wrappers
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

> Prerequisites: Python 3.12+, Docker, OpenAI API key, LangSmith API key

```bash
# Clone the repo
git clone <repo-url>
cd GenAI-HCAP

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the API
uvicorn src.api.main:app --reload
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key for LLM, Whisper, and TTS |
| `LANGSMITH_API_KEY` | LangSmith tracing key |
| `LANGSMITH_PROJECT` | LangSmith project name |
| `CHROMA_PERSIST_DIR` | Local path for ChromaDB persistence |

---

## Safety & Compliance Notes

- This is a **kata / prototype** — not intended for production clinical use
- All LLM responses must be validated through guardrails before delivery
- The system should never provide diagnosis, prescriptions, or emergency medical advice without escalation
- Patient data used in demos must be **synthetic / anonymized**
