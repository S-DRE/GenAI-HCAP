import structlog
from langchain_core.tools import tool

logger = structlog.get_logger()


@tool
def escalate(reason: str) -> str:
    """Escalate an urgent situation to on-call caregiving staff or medical personnel.
    Use this tool when the patient reports an emergency, severe symptoms, a fall,
    chest pain, difficulty breathing, or any situation requiring immediate human attention."""
    logger.warning("escalation_triggered", reason=reason)

    # TODO: integrate with a real notification system (e.g. SMS, email, paging system)
    # For the kata this logs the event and returns a reassuring confirmation message.
    return (
        "I have alerted your care team about this situation. "
        "Someone will contact you shortly. "
        f"Escalation reason logged: {reason}. "
        "If this is a life-threatening emergency, please call emergency services immediately."
    )
